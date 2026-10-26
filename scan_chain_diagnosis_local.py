#!/usr/bin/env python3
"""
Unified local scan-chain diagnosis simulator.

This consolidates the recent million-flop experiments into one self-contained
program that can be run locally (Windows/Linux/macOS).

Supported modes
---------------
1) pool-scale
   Reproduces the nested candidate-pool experiment:
       P = 64, 128, 256, 512, 1024
   using one common max-size pattern pool.

2) targeted-sat
   Uses the largest candidate pool, extracts every residual ambiguity class,
   and asks an exact SAT question for every unresolved pair:
       exists pattern p such that R(p, f_i) != R(p, f_j)?
   SAT     -> synthesize and verify a witness pattern.
   UNSAT   -> structurally inseparable under THIS synthetic model.
   UNKNOWN -> SAT conflict budget exhausted.

3) all
   Runs pool scaling and targeted SAT in one pass, reusing the same response
   matrix so the expensive fault simulation is not repeated.

The default full-size configuration is:
    1000 chains x 1000 flops = 1,000,000 scan flops
    10,000,000 synthetic gates
    5 random failed chains
    nested pattern pools 64,128,256,512,1024

Important modeling scope
------------------------
This remains the Phase-1 synthetic architecture used in the recent experiments:
each scan chain owns one local combinational block. It tests the hierarchical
diagnosis concept:
    huge chip -> flush screening -> local failing-chain diagnosis.
It does NOT yet model cross-chain functional coupling, EDT decompression, or
response compaction.

Fault model
-----------
- One permanent scan-shift stuck-at fault per failed chain.
- Scan index 0 is nearest scan-out.
- If Q_n is stuck at v during load, Q_0..Q_n are forced to v.
- Functional capture is not clamped.
- During unload, captured values Q_0..Q_{n-1} can leave; from n onward the
  observed value is stuck at v.
- Alternating flush is used only for chain detection and optional SA0/SA1
  polarity inference; it does not reveal the cell index.

Local-machine features
----------------------
- Pure CLI; no Colab syntax.
- Cross-platform paths.
- Optional Numba thread control.
- Optional design cache to avoid rebuilding the 10M-gate synthetic design.
- Proper response timing checkpoints for each nested pattern-pool size.
- Optional one-chain targeting for debugging.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import numpy as np

try:
    from numba import njit, prange, get_num_threads, set_num_threads
except Exception as exc:
    raise RuntimeError(
        "numba is required. Install dependencies with:\n"
        "    python -m pip install -r requirements_scan_simulator.txt"
    ) from exc


# =============================================================================
# Gate model
# =============================================================================

AND, OR, XOR, NAND, NOR, XNOR, BUF, NOT = range(8)


@njit(cache=True, parallel=True)
def simulate_block(gtype, f1, f2, capsrc, scan_state, gate_values, capture):
    """Bit-parallel gate simulation; one uint64 bit = one fault hypothesis."""
    L, W = scan_state.shape
    G = gtype.size

    # Fault words are independent.
    for w in prange(W):
        for g in range(G):
            s1 = f1[g]
            a = scan_state[s1, w] if s1 < L else gate_values[s1 - L, w]
            t = gtype[g]

            if t == BUF:
                y = a
            elif t == NOT:
                y = ~a
            else:
                s2 = f2[g]
                b = scan_state[s2, w] if s2 < L else gate_values[s2 - L, w]

                if t == AND:
                    y = a & b
                elif t == OR:
                    y = a | b
                elif t == XOR:
                    y = a ^ b
                elif t == NAND:
                    y = ~(a & b)
                elif t == NOR:
                    y = ~(a | b)
                else:  # XNOR
                    y = ~(a ^ b)

            gate_values[g, w] = y

        for i in range(L):
            s = capsrc[i]
            capture[i, w] = (
                scan_state[s, w] if s < L else gate_values[s - L, w]
            )


# =============================================================================
# Synthetic design and faults
# =============================================================================

@dataclass(frozen=True)
class Fault:
    chain: int
    cell: int
    stuck: int


class Design:
    """Compact block-diagonal synthetic circuit."""

    def __init__(self, chains, L, total_gates, seed, allocate=True):
        self.chains = int(chains)
        self.L = int(L)
        self.gpc = int(round(total_gates / chains))
        self.total_gates = self.gpc * self.chains
        self.seed = int(seed)

        if allocate:
            self.gtype = np.empty(self.total_gates, np.uint8)
            self.f1 = np.empty(self.total_gates, np.int32)
            self.f2 = np.empty(self.total_gates, np.int32)
            self.capsrc = np.empty((self.chains, self.L), np.int32)
        else:
            self.gtype = None
            self.f1 = None
            self.f2 = None
            self.capsrc = None

    @property
    def memory_mib(self):
        return (
            self.gtype.nbytes
            + self.f1.nbytes
            + self.f2.nbytes
            + self.capsrc.nbytes
        ) / (1024**2)

    def build(self, progress=100):
        rng = np.random.default_rng(self.seed)
        G, L = self.gpc, self.L
        maxsrc = L + np.arange(G, dtype=np.int64)

        t0 = time.perf_counter()

        for c in range(self.chains):
            lo, hi = c * G, (c + 1) * G

            self.gtype[lo:hi] = rng.integers(
                0, 8, G, dtype=np.uint8
            )
            self.f1[lo:hi] = (
                rng.random(G) * maxsrc
            ).astype(np.int32)
            self.f2[lo:hi] = (
                rng.random(G) * maxsrc
            ).astype(np.int32)

            # D-inputs of scan cells are driven from latter half of block.
            self.capsrc[c] = rng.integers(
                L + G // 2,
                L + G,
                L,
                dtype=np.int32,
            )

            if progress and (c + 1) % progress == 0:
                print(
                    f"[design] {c+1}/{self.chains} blocks, "
                    f"{(c+1)*G:,} gates"
                )

        return time.perf_counter() - t0

    def block(self, chain):
        lo = chain * self.gpc
        hi = (chain + 1) * self.gpc
        return (
            self.gtype[lo:hi],
            self.f1[lo:hi],
            self.f2[lo:hi],
            self.capsrc[chain],
        )

    def save(self, path: Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        np.savez(
            path,
            chains=np.int64(self.chains),
            L=np.int64(self.L),
            total_gates=np.int64(self.total_gates),
            gpc=np.int64(self.gpc),
            seed=np.int64(self.seed),
            gtype=self.gtype,
            f1=self.f1,
            f2=self.f2,
            capsrc=self.capsrc,
        )

    @classmethod
    def load(cls, path: Path, expected=None):
        with np.load(path, allow_pickle=False) as z:
            chains = int(z["chains"])
            L = int(z["L"])
            total_gates = int(z["total_gates"])
            seed = int(z["seed"])

            if expected is not None:
                exp_chains, exp_L, exp_total_gates, exp_seed = expected
                if (
                    chains != int(exp_chains)
                    or L != int(exp_L)
                    or total_gates != int(exp_total_gates)
                    or seed != int(exp_seed)
                ):
                    raise ValueError(
                        "Design-cache metadata does not match this run.\n"
                        f"cache: chains={chains}, L={L}, "
                        f"gates={total_gates}, seed={seed}\n"
                        f"run:   chains={exp_chains}, L={exp_L}, "
                        f"gates={exp_total_gates}, seed={exp_seed}"
                    )

            obj = cls(chains, L, total_gates, seed, allocate=False)
            obj.gtype = np.array(z["gtype"], copy=True)
            obj.f1 = np.array(z["f1"], copy=True)
            obj.f2 = np.array(z["f2"], copy=True)
            obj.capsrc = np.array(z["capsrc"], copy=True)
            obj.gpc = int(z["gpc"])
            obj.total_gates = total_gates
            return obj


def get_or_build_design(args):
    cache = Path(args.design_cache) if args.design_cache else None

    expected = (
        args.chains,
        args.chain_length,
        int(round(args.total_gates / args.chains)) * args.chains,
        args.logic_seed,
    )

    if cache is not None and cache.exists():
        print(f"[design] loading cache: {cache}")
        t0 = time.perf_counter()
        design = Design.load(cache, expected=expected)
        seconds = time.perf_counter() - t0
        print(
            f"[design] loaded {design.total_gates:,} gates in "
            f"{seconds:.2f}s; static arrays ~{design.memory_mib:.1f} MiB"
        )
        return design, seconds, True

    design = Design(
        args.chains,
        args.chain_length,
        args.total_gates,
        args.logic_seed,
    )
    print(
        f"[design] building {design.total_gates:,} gates; "
        f"static arrays ~{design.memory_mib:.1f} MiB"
    )
    seconds = design.build(args.design_progress)
    print(f"[design] build time: {seconds:.2f}s")

    if cache is not None:
        print(f"[design] saving cache: {cache}")
        t0 = time.perf_counter()
        design.save(cache)
        print(
            f"[design] cache saved in {time.perf_counter()-t0:.2f}s"
        )

    return design, seconds, False


def inject_faults(chains, L, nfail, rng):
    if nfail > chains:
        raise ValueError("--failed-chains cannot exceed --chains")

    ids = rng.choice(chains, nfail, replace=False)
    cells = rng.integers(0, L, nfail)
    vals = rng.integers(0, 2, nfail)

    return [
        Fault(int(c), int(n), int(v))
        for c, n, v in zip(ids, cells, vals)
    ]


def flush_screen(chains, L, faults, complementary=True):
    """
    Chain-level screening only.

    The synthetic permanent shift-stuck fault collapses the faulty shift stream
    to the stuck value. We intentionally do not use mismatch position.
    """
    expected = 1 - (np.arange(L, dtype=np.uint8) & 1)  # 1010...
    patterns = [expected, 1 - expected] if complementary else [expected]

    byc = {f.chain: f for f in faults}
    detected = []
    stuck = {}

    for c in range(chains):
        f = byc.get(c)
        fail = False

        for p in patterns:
            obs = p if f is None else np.full_like(p, f.stuck)
            if not np.array_equal(obs, p):
                fail = True

        if fail:
            detected.append(c)
            stuck[c] = f.stuck

    return detected, stuck


# =============================================================================
# Fault-hypothesis packing and response database
# =============================================================================

def pack_rows_u64(bits):
    bits = np.asarray(bits, np.uint8)
    rows, nbits = bits.shape

    pad_bits = ((nbits + 63) // 64) * 64
    if pad_bits != nbits:
        tmp = np.zeros((rows, pad_bits), np.uint8)
        tmp[:, :nbits] = bits
        bits = tmp

    packed = np.packbits(bits, axis=1, bitorder="little")
    return np.ascontiguousarray(packed).view(np.uint64)


def hypotheses(L, known_stuck):
    if known_stuck is None:
        cell = np.r_[
            np.arange(L, dtype=np.int32),
            np.arange(L, dtype=np.int32),
        ]
        val = np.r_[
            np.zeros(L, np.uint8),
            np.ones(L, np.uint8),
        ]
    else:
        cell = np.arange(L, dtype=np.int32)
        val = np.full(L, int(known_stuck), np.uint8)

    return cell, val


def force_masks(L, hcell, hval):
    j = np.arange(L, dtype=np.int32)[:, None]
    affects = hcell[None, :] >= j

    f0 = pack_rows_u64(
        affects & (hval[None, :] == 0)
    )
    f1 = pack_rows_u64(
        affects & (hval[None, :] == 1)
    )

    H, W = hcell.size, f0.shape[1]

    valid = np.full(
        W,
        np.uint64(0xFFFFFFFFFFFFFFFF),
        dtype=np.uint64,
    )
    if H % 64:
        valid[-1] = np.uint64((1 << (H % 64)) - 1)

    return f0, f1, valid


def packed_loaded_state(pattern, f0, f1, valid):
    one = pattern.astype(bool)
    out = np.empty_like(f0)

    # Intended 0 -> only SA1-forced hypotheses are 1.
    out[~one] = f1[~one]

    # Intended 1 -> all valid hypotheses are 1 except SA0-forced ones.
    if np.any(one):
        out[one] = valid[None, :] & (~f0[one])

    return np.ascontiguousarray(out)


def salts(nwords, seed):
    rng = np.random.default_rng(seed)
    hi = np.iinfo(np.uint64).max

    a = rng.integers(
        1, hi, nwords, dtype=np.uint64
    ) | np.uint64(1)
    b = rng.integers(
        1, hi, nwords, dtype=np.uint64
    ) | np.uint64(1)

    return a, b


def response_hash(capture, hcell, hval, salt1, salt2):
    """Two uint64 fingerprints of each full faulty unload response."""
    L, W = capture.shape
    H = hcell.size

    raw = np.ascontiguousarray(capture).view(np.uint8)
    cap = np.unpackbits(
        raw,
        axis=1,
        bitorder="little",
    )[:, :H].T  # [H,L]

    tail = (
        np.arange(L, dtype=np.int32)[None, :]
        >= hcell[:, None]
    )

    resp = np.where(
        tail,
        hval[:, None],
        cap,
    ).astype(np.uint8, copy=False)

    rb = np.packbits(
        resp,
        axis=1,
        bitorder="little",
    )

    need = ((rb.shape[1] + 7) // 8) * 8
    if need != rb.shape[1]:
        tmp = np.zeros((H, need), np.uint8)
        tmp[:, :rb.shape[1]] = rb
        rb = tmp

    words = np.ascontiguousarray(rb).view(np.uint64)

    h1 = np.bitwise_xor.reduce(
        words * salt1[None, :],
        axis=1,
    )

    mix = (
        words
        ^ (words >> np.uint64(23))
        ^ (words << np.uint64(17))
    )
    h2 = np.bitwise_xor.reduce(
        mix * salt2[None, :],
        axis=1,
    )

    return h1, h2


def candidate_patterns(P, L, seed):
    return np.random.default_rng(seed).integers(
        0,
        2,
        (P, L),
        dtype=np.uint8,
    )


def simulate_db(
    block,
    patterns,
    hcell,
    hval,
    hash_seed,
    progress=32,
    timing_checkpoints=None,
):
    """
    Generate response fingerprints for every pattern x hypothesis.

    Returns
    -------
    R : uint64 [P,H,2]
    total_seconds : float
    checkpoint_times : dict[int,float]
        Cumulative timed response-generation time after P patterns.
    """
    gt, fi1, fi2, capsrc = block
    P, L = patterns.shape
    H = hcell.size

    f0, f1, valid = force_masks(L, hcell, hval)
    W = valid.size

    s1, s2 = salts((L + 63) // 64, hash_seed)

    out = np.empty((P, H, 2), np.uint64)
    gv = np.empty((gt.size, W), np.uint64)
    cap = np.empty((L, W), np.uint64)

    checkpoints = set(timing_checkpoints or [])
    checkpoint_times = {}

    # Compile the Numba kernel before starting the timer.
    warm_state = packed_loaded_state(
        patterns[0],
        f0,
        f1,
        valid,
    )
    simulate_block(
        gt,
        fi1,
        fi2,
        capsrc,
        warm_state,
        gv,
        cap,
    )

    t0 = time.perf_counter()

    for p in range(P):
        state = packed_loaded_state(
            patterns[p],
            f0,
            f1,
            valid,
        )

        simulate_block(
            gt,
            fi1,
            fi2,
            capsrc,
            state,
            gv,
            cap,
        )

        cap[:, -1] &= valid[-1]

        a, b = response_hash(
            cap,
            hcell,
            hval,
            s1,
            s2,
        )

        out[p, :, 0] = a
        out[p, :, 1] = b

        completed = p + 1
        if completed in checkpoints:
            checkpoint_times[completed] = (
                time.perf_counter() - t0
            )

        if progress and completed % progress == 0:
            dt = time.perf_counter() - t0
            print(
                f"    [response] {completed}/{P}, "
                f"{completed/max(dt,1e-12):.2f} patterns/s"
            )

    total_seconds = time.perf_counter() - t0
    return out, total_seconds, checkpoint_times


# =============================================================================
# Partitioning and greedy selectors
# =============================================================================

C1 = np.uint64(0x9E3779B185EBCA87)
C2 = np.uint64(0xC2B2AE3D27D4EB4F)
C3 = np.uint64(0x165667B19E3779F9)
C4 = np.uint64(0x85EBCA77C2B2AE63)


def mix64(x):
    with np.errstate(over="ignore"):
        x = x.astype(np.uint64, copy=False)
        x = x ^ (x >> np.uint64(30))
        x = x * C1
        x = x ^ (x >> np.uint64(27))
        x = x * C2
        return x ^ (x >> np.uint64(31))


def combine(a, b, r1, r2):
    return (
        mix64(a ^ mix64(r1 + C3)),
        mix64(b ^ mix64(r2 + C4)),
    )


def counts2(a, b):
    key = np.empty(
        a.size,
        dtype=[("a", "<u8"), ("b", "<u8")],
    )
    key["a"] = a
    key["b"] = b
    return np.unique(
        key,
        return_counts=True,
    )[1].astype(np.int64)


def unresolved_pairs(counts):
    return int(
        np.sum(counts * (counts - 1) // 2)
    )


def entropy(counts, H):
    return float(
        np.sum((counts / H) * np.log2(counts))
    )


def max_partition(R):
    P, H, _ = R.shape
    a = np.zeros(H, np.uint64)
    b = np.zeros(H, np.uint64)

    for p in range(P):
        a, b = combine(
            a,
            b,
            R[p, :, 0],
            R[p, :, 1],
        )

    c = counts2(a, b)
    return len(c), unresolved_pairs(c)


def greedy(R, objective, max_selected=50):
    """
    Greedy fixed-set selector.

    objective='ig':
        maximize reduction in expected log2 ambiguity.

    objective='pair':
        maximize number of currently unresolved fault pairs separated.
    """
    P, H, _ = R.shape

    used = np.zeros(P, bool)
    selected = []

    a = np.zeros(H, np.uint64)
    b = np.zeros(H, np.uint64)

    current_counts = np.array([H], np.int64)
    current_unresolved = unresolved_pairs(current_counts)
    current_entropy = entropy(current_counts, H)

    t0 = time.perf_counter()

    for _ in range(max_selected):
        best = -1
        best_score = -1.0
        keep = None

        for p in range(P):
            if used[p]:
                continue

            x, y = combine(
                a,
                b,
                R[p, :, 0],
                R[p, :, 1],
            )
            c = counts2(x, y)

            if objective == "pair":
                score = float(
                    current_unresolved
                    - unresolved_pairs(c)
                )
            elif objective == "ig":
                score = (
                    current_entropy
                    - entropy(c, H)
                )
            else:
                raise ValueError(
                    "objective must be 'ig' or 'pair'"
                )

            if score > best_score + 1e-15:
                best_score = score
                best = p
                keep = (x, y, c)

        if best < 0 or best_score <= 0:
            break

        used[best] = True
        selected.append(best)

        a, b, c = keep
        current_unresolved = unresolved_pairs(c)
        current_entropy = entropy(c, H)

        if current_unresolved == 0:
            break

    c = counts2(a, b)

    return {
        "selected": selected,
        "seconds": time.perf_counter() - t0,
        "classes": len(c),
        "unresolved": unresolved_pairs(c),
        "a": a,
        "b": b,
    }


def true_index(fault, L, known_stuck):
    if known_stuck is not None:
        return fault.cell
    return (
        fault.cell
        if fault.stuck == 0
        else L + fault.cell
    )


def class_size(selection, idx):
    return int(
        np.sum(
            (selection["a"] == selection["a"][idx])
            & (selection["b"] == selection["b"][idx])
        )
    )


# =============================================================================
# Ambiguity extraction
# =============================================================================

def full_signature(R):
    P, H, _ = R.shape
    a = np.zeros(H, dtype=np.uint64)
    b = np.zeros(H, dtype=np.uint64)

    for p in range(P):
        a, b = combine(
            a,
            b,
            R[p, :, 0],
            R[p, :, 1],
        )

    return a, b


def ambiguity_classes(R):
    """Return only equivalence classes with size > 1."""
    a, b = full_signature(R)

    key = np.empty(
        a.size,
        dtype=[("a", "<u8"), ("b", "<u8")],
    )
    key["a"] = a
    key["b"] = b

    _, inv = np.unique(
        key,
        return_inverse=True,
    )

    groups = {}
    for i, gid in enumerate(inv):
        groups.setdefault(
            int(gid),
            [],
        ).append(int(i))

    ambiguous = [
        group
        for group in groups.values()
        if len(group) > 1
    ]
    ambiguous.sort(
        key=lambda g: (g[0], len(g))
    )
    return ambiguous


def all_pairs_from_classes(classes):
    pairs = []
    for group in classes:
        pairs.extend(
            itertools.combinations(group, 2)
        )
    return [
        (int(i), int(j))
        for i, j in pairs
    ]


# =============================================================================
# SAT support (lazy dependency)
# =============================================================================

def import_sat():
    try:
        from pysat.formula import CNF
        from pysat.solvers import Glucose4
    except Exception as exc:
        raise RuntimeError(
            "targeted-sat/all mode requires python-sat.\n"
            "Install dependencies with:\n"
            "    python -m pip install -r "
            "requirements_scan_simulator.txt"
        ) from exc

    return CNF, Glucose4


def add_equiv(cnf, a, b):
    """a <-> b"""
    cnf.append([-a, b])
    cnf.append([a, -b])


def add_gate(cnf, typ, a, b, y):
    """CNF for y = gate(a,b)."""
    if typ == AND:
        cnf.append([-a, -b, y])
        cnf.append([a, -y])
        cnf.append([b, -y])

    elif typ == OR:
        cnf.append([a, b, -y])
        cnf.append([-a, y])
        cnf.append([-b, y])

    elif typ == XOR:
        cnf.append([-a, -b, -y])
        cnf.append([a, b, -y])
        cnf.append([a, -b, y])
        cnf.append([-a, b, y])

    elif typ == NAND:
        cnf.append([-a, -b, -y])
        cnf.append([a, y])
        cnf.append([b, y])

    elif typ == NOR:
        cnf.append([a, b, y])
        cnf.append([-a, -y])
        cnf.append([-b, -y])

    elif typ == XNOR:
        cnf.append([-a, -b, y])
        cnf.append([a, b, y])
        cnf.append([a, -b, -y])
        cnf.append([-a, b, -y])

    elif typ == BUF:
        cnf.append([-a, y])
        cnf.append([a, -y])

    elif typ == NOT:
        cnf.append([a, y])
        cnf.append([-a, -y])

    else:
        raise ValueError(
            f"unknown gate type {typ}"
        )


def add_xor_var(cnf, a, b, d):
    """d <-> (a XOR b)."""
    cnf.append([-a, -b, -d])
    cnf.append([a, b, -d])
    cnf.append([a, -b, d])
    cnf.append([-a, b, d])


def alloc_vars(start, count):
    arr = np.arange(
        start,
        start + count,
        dtype=np.int32,
    )
    return arr, start + count


def build_pair_cnf(block, hcell, hval, i, j):
    CNF, _ = import_sat()

    gtype, f1, f2, capsrc = block
    L = int(capsrc.size)
    G = int(gtype.size)

    ni = int(hcell[i])
    vi = int(hval[i])
    nj = int(hcell[j])
    vj = int(hval[j])

    cnf = CNF()
    nxt = 1

    # Shared intended scan-load bits.
    x, nxt = alloc_vars(nxt, L)

    # Faulty loaded scan states.
    si, nxt = alloc_vars(nxt, L)
    sj, nxt = alloc_vars(nxt, L)

    # Two copies of functional logic.
    gi, nxt = alloc_vars(nxt, G)
    gj, nxt = alloc_vars(nxt, G)

    # Scan-load fault model.
    for q in range(L):
        if q <= ni:
            cnf.append(
                [int(si[q])]
                if vi
                else [-int(si[q])]
            )
        else:
            add_equiv(
                cnf,
                int(si[q]),
                int(x[q]),
            )

        if q <= nj:
            cnf.append(
                [int(sj[q])]
                if vj
                else [-int(sj[q])]
            )
        else:
            add_equiv(
                cnf,
                int(sj[q]),
                int(x[q]),
            )

    def source_var(source, scan_vars, gate_vars):
        if source < L:
            return int(scan_vars[source])
        return int(gate_vars[source - L])

    # Functional capture network.
    for g in range(G):
        typ = int(gtype[g])

        ai = source_var(
            int(f1[g]),
            si,
            gi,
        )
        aj = source_var(
            int(f1[g]),
            sj,
            gj,
        )

        if typ in (BUF, NOT):
            bi = None
            bj = None
        else:
            bi = source_var(
                int(f2[g]),
                si,
                gi,
            )
            bj = source_var(
                int(f2[g]),
                sj,
                gj,
            )

        add_gate(
            cnf,
            typ,
            ai,
            bi,
            int(gi[g]),
        )
        add_gate(
            cnf,
            typ,
            aj,
            bj,
            int(gj[g]),
        )

    # Require at least one tester-visible unload bit to differ.
    diff_literals = []
    always_different = False

    for k in range(L):
        if k >= ni:
            ri_const = vi
            ri_var = None
        else:
            ri_const = None
            ri_var = source_var(
                int(capsrc[k]),
                si,
                gi,
            )

        if k >= nj:
            rj_const = vj
            rj_var = None
        else:
            rj_const = None
            rj_var = source_var(
                int(capsrc[k]),
                sj,
                gj,
            )

        if ri_var is None and rj_var is None:
            if ri_const != rj_const:
                always_different = True
                break
            continue

        if ri_var is None:
            diff_literals.append(
                int(rj_var)
                if ri_const == 0
                else -int(rj_var)
            )
            continue

        if rj_var is None:
            diff_literals.append(
                int(ri_var)
                if rj_const == 0
                else -int(ri_var)
            )
            continue

        if ri_var == rj_var:
            continue

        d = nxt
        nxt += 1
        add_xor_var(
            cnf,
            int(ri_var),
            int(rj_var),
            int(d),
        )
        diff_literals.append(int(d))

    if not always_different:
        if not diff_literals:
            cnf.append([])
        else:
            cnf.append(diff_literals)

    meta = {
        "x_vars": x,
        "nvars": nxt - 1,
        "nclauses": len(cnf.clauses),
        "fault_i": (ni, vi),
        "fault_j": (nj, vj),
        "always_different": always_different,
    }

    return cnf, meta


def solve_pair(block, hcell, hval, i, j, conf_budget):
    _, Glucose4 = import_sat()

    cnf, meta = build_pair_cnf(
        block,
        hcell,
        hval,
        i,
        j,
    )

    t0 = time.perf_counter()

    with Glucose4(
        bootstrap_with=cnf.clauses
    ) as solver:

        if conf_budget and conf_budget > 0:
            solver.conf_budget(
                int(conf_budget)
            )
            sat = solver.solve_limited(
                expect_interrupt=True
            )
        else:
            sat = solver.solve()

        elapsed = (
            time.perf_counter() - t0
        )

        if sat is True:
            model = solver.get_model()
            positives = {
                lit
                for lit in model
                if lit > 0
            }

            pattern = np.fromiter(
                (
                    1
                    if int(v) in positives
                    else 0
                    for v in meta["x_vars"]
                ),
                dtype=np.uint8,
                count=len(meta["x_vars"]),
            )
            status = "SAT"

        elif sat is False:
            pattern = None
            status = "UNSAT"

        else:
            pattern = None
            status = "UNKNOWN"

    return (
        status,
        pattern,
        elapsed,
        meta,
    )


def verify_witness(
    block,
    pattern,
    hcell,
    hval,
    i,
    j,
    hash_seed,
):
    pair_cell = np.asarray(
        [hcell[i], hcell[j]],
        dtype=np.int32,
    )
    pair_val = np.asarray(
        [hval[i], hval[j]],
        dtype=np.uint8,
    )

    R, _, _ = simulate_db(
        block,
        pattern[None, :],
        pair_cell,
        pair_val,
        hash_seed,
        progress=0,
    )

    same = (
        R[0, 0, 0] == R[0, 1, 0]
        and R[0, 0, 1] == R[0, 1, 1]
    )
    return not same


# =============================================================================
# Output helpers
# =============================================================================

def write_csv(path, rows):
    if not rows:
        return

    with Path(path).open(
        "w",
        newline="",
        encoding="utf-8",
    ) as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)


def save_ambiguity_classes(
    path,
    classes,
    hcell,
    hval,
):
    payload = []

    for class_id, group in enumerate(classes):
        cells = [
            int(hcell[idx])
            for idx in group
        ]
        contiguous = (
            cells
            == list(
                range(
                    min(cells),
                    max(cells) + 1,
                )
            )
            if cells
            else False
        )

        payload.append({
            "class_id": class_id,
            "size": len(group),
            "hypothesis_indices": group,
            "cells": cells,
            "contiguous_cells": contiguous,
            "faults": [
                {
                    "index": idx,
                    "cell": int(hcell[idx]),
                    "stuck": int(hval[idx]),
                }
                for idx in group
            ],
        })

    Path(path).write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )


# =============================================================================
# Per-chain experiment pieces
# =============================================================================

def run_pool_scaling_for_chain(
    ch,
    fault,
    true_idx,
    hcell,
    hval,
    R,
    pattern_counts,
    checkpoint_times,
    max_selected,
    outdir,
):
    rows = []

    total_pairs = (
        hcell.size * (hcell.size - 1) // 2
    )

    for P in pattern_counts:
        Rp = R[:P]

        max_classes, intrinsic = (
            max_partition(Rp)
        )
        frac = (
            100.0 * intrinsic / total_pairs
            if total_pairs
            else 0.0
        )

        ig = greedy(
            Rp,
            "ig",
            max_selected,
        )
        pc = greedy(
            Rp,
            "pair",
            max_selected,
        )

        row = {
            "chain": ch,
            "true_cell": fault.cell,
            "true_stuck": fault.stuck,
            "hypotheses": int(hcell.size),
            "pattern_pool_size": int(P),
            "max_classes": int(max_classes),
            "intrinsic_unresolved_pairs": int(intrinsic),
            "unresolved_pair_fraction_pct": float(frac),
            "response_seconds": float(
                checkpoint_times.get(P, np.nan)
            ),
            "ig_patterns": len(ig["selected"]),
            "ig_seconds": float(ig["seconds"]),
            "ig_classes": int(ig["classes"]),
            "ig_unresolved_pairs": int(ig["unresolved"]),
            "pair_patterns": len(pc["selected"]),
            "pair_seconds": float(pc["seconds"]),
            "pair_classes": int(pc["classes"]),
            "pair_unresolved_pairs": int(pc["unresolved"]),
            "true_class_size_ig": class_size(
                ig,
                true_idx,
            ),
            "true_class_size_pair": class_size(
                pc,
                true_idx,
            ),
        }
        rows.append(row)

        result_json = {
            **row,
            "ig_selected_pattern_ids": ig["selected"],
            "pair_selected_pattern_ids": pc["selected"],
        }

        (
            Path(outdir)
            / f"chain_{ch:04d}_pool_{P:04d}.json"
        ).write_text(
            json.dumps(
                result_json,
                indent=2,
            ),
            encoding="utf-8",
        )

        print(
            f"[pool P={P:4d}] "
            f"max={max_classes}/{hcell.size}, "
            f"unresolved={intrinsic}, "
            f"IG={len(ig['selected'])}, "
            f"pair={len(pc['selected'])}"
        )

    return rows


def run_targeted_sat_for_chain(
    ch,
    fault,
    block,
    hcell,
    hval,
    R,
    base_patterns,
    args,
    outdir,
):
    classes = ambiguity_classes(R)
    pairs = all_pairs_from_classes(classes)

    active = sorted(
        {
            x
            for pair in pairs
            for x in pair
        }
    )

    base_classes, base_unresolved = (
        max_partition(R)
    )

    save_ambiguity_classes(
        Path(outdir)
        / f"chain_{ch:04d}_ambiguity_classes.json",
        classes,
        hcell,
        hval,
    )

    print(
        f"[SAT] chain {ch}: "
        f"base={base_classes}/{hcell.size}, "
        f"ambiguity classes={len(classes)}, "
        f"active hypotheses={len(active)}, "
        f"pair queries={len(pairs)}"
    )

    pair_rows = []
    witnesses = []
    witness_id_by_key = {}

    sat_count = 0
    unsat_count = 0
    unknown_count = 0
    verified_sat = 0
    sat_total_seconds = 0.0

    for pair_idx, (i, j) in enumerate(
        pairs,
        start=1,
    ):
        (
            status,
            witness,
            sat_seconds,
            meta,
        ) = solve_pair(
            block,
            hcell,
            hval,
            i,
            j,
            args.sat_conf_budget,
        )

        sat_total_seconds += sat_seconds
        verified = False
        witness_id = None

        if status == "SAT":
            sat_count += 1

            verified = verify_witness(
                block,
                witness,
                hcell,
                hval,
                i,
                j,
                args.hash_seed
                + ch
                + 10_000_000
                + pair_idx,
            )

            if verified:
                verified_sat += 1
                key = np.packbits(
                    witness,
                    bitorder="little",
                ).tobytes()

                if key not in witness_id_by_key:
                    witness_id_by_key[key] = (
                        len(witnesses)
                    )
                    witnesses.append(
                        witness.copy()
                    )

                witness_id = (
                    witness_id_by_key[key]
                )

        elif status == "UNSAT":
            unsat_count += 1
        else:
            unknown_count += 1

        pair_rows.append({
            "chain": ch,
            "pair_query": pair_idx,
            "hypothesis_i": i,
            "cell_i": int(hcell[i]),
            "stuck_i": int(hval[i]),
            "hypothesis_j": j,
            "cell_j": int(hcell[j]),
            "stuck_j": int(hval[j]),
            "status": status,
            "verified": int(verified),
            "witness_id": witness_id,
            "sat_seconds": float(sat_seconds),
            "cnf_vars": int(meta["nvars"]),
            "cnf_clauses": int(meta["nclauses"]),
        })

        if (
            args.pair_progress
            and (
                pair_idx % args.pair_progress == 0
                or pair_idx == len(pairs)
            )
        ):
            print(
                f"    [SAT] {pair_idx}/{len(pairs)} "
                f"SAT={sat_count}, "
                f"UNSAT={unsat_count}, "
                f"UNKNOWN={unknown_count}, "
                f"unique witnesses={len(witnesses)}"
            )

    write_csv(
        Path(outdir)
        / f"chain_{ch:04d}_pair_sat.csv",
        pair_rows,
    )

    witness_sim_seconds = 0.0

    if witnesses:
        W = np.asarray(
            witnesses,
            dtype=np.uint8,
        )

        np.savez_compressed(
            Path(outdir)
            / f"chain_{ch:04d}_sat_witnesses.npz",
            patterns=W,
        )

        witness_R, witness_sim_seconds, _ = (
            simulate_db(
                block,
                W,
                hcell,
                hval,
                args.hash_seed
                + ch
                + 20_000_000,
                progress=0,
            )
        )

        augmented = np.concatenate(
            [R, witness_R],
            axis=0,
        )
        final_classes, final_unresolved = (
            max_partition(augmented)
        )
    else:
        final_classes = base_classes
        final_unresolved = base_unresolved

    print(
        f"[SAT] result chain {ch}: "
        f"SAT={sat_count}, "
        f"UNSAT={unsat_count}, "
        f"UNKNOWN={unknown_count}; "
        f"resolution {base_classes}->{final_classes}, "
        f"unresolved {base_unresolved}->{final_unresolved}"
    )

    return {
        "chain": ch,
        "true_cell": fault.cell,
        "true_stuck": fault.stuck,
        "hypotheses": int(hcell.size),
        "base_patterns": int(base_patterns),
        "base_classes": int(base_classes),
        "base_unresolved_pairs": int(base_unresolved),
        "ambiguous_classes": int(len(classes)),
        "active_ambiguous_hypotheses": int(len(active)),
        "pair_queries": int(len(pairs)),
        "sat_pairs": int(sat_count),
        "unsat_pairs": int(unsat_count),
        "unknown_pairs": int(unknown_count),
        "verified_sat_pairs": int(verified_sat),
        "unique_witness_patterns": int(len(witnesses)),
        "final_classes": int(final_classes),
        "final_unresolved_pairs": int(final_unresolved),
        "sat_total_seconds": float(sat_total_seconds),
        "witness_sim_seconds": float(witness_sim_seconds),
    }


# =============================================================================
# Main
# =============================================================================

def run(args):
    outdir = Path(args.outdir)
    outdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    if args.threads:
        current_max = get_num_threads()
        chosen = min(
            int(args.threads),
            int(current_max),
        )
        set_num_threads(chosen)

    print("=" * 80)
    print("UNIFIED LOCAL SCAN-CHAIN DIAGNOSIS SIMULATOR")
    print("=" * 80)
    print(f"mode              : {args.mode}")
    print(f"Numba threads     : {get_num_threads()}")
    print(f"scan chains       : {args.chains:,}")
    print(f"flops/chain       : {args.chain_length:,}")
    print(
        f"total scan flops  : "
        f"{args.chains*args.chain_length:,}"
    )
    print(f"requested gates   : {args.total_gates:,}")
    print(f"failed chains     : {args.failed_chains}")
    print(
        f"pattern pools     : "
        f"{args.pattern_counts}"
    )
    print()

    # SAT dependency is checked before long work starts.
    if args.mode in ("targeted-sat", "all"):
        import_sat()

    design, design_seconds, cache_used = (
        get_or_build_design(args)
    )

    rng = np.random.default_rng(
        args.seed
    )
    faults = inject_faults(
        args.chains,
        args.chain_length,
        args.failed_chains,
        rng,
    )
    truth = {
        f.chain: f
        for f in faults
    }

    print("\n[injected faults]")
    for f in sorted(
        faults,
        key=lambda z: z.chain,
    ):
        print(
            f"  chain {f.chain:4d}: "
            f"Q{f.cell}:SA{f.stuck}"
        )

    t0 = time.perf_counter()
    detected, polarity = flush_screen(
        args.chains,
        args.chain_length,
        faults,
        complementary=not args.single_flush,
    )
    flush_seconds = (
        time.perf_counter() - t0
    )

    print(
        f"\n[flush] detected "
        f"{len(detected)}/{len(faults)} "
        f"in {flush_seconds:.6f}s: "
        f"{detected}"
    )

    if args.only_chain is not None:
        if args.only_chain not in detected:
            raise ValueError(
                f"--only-chain {args.only_chain} "
                f"is not in detected chains "
                f"{detected}"
            )
        detected = [args.only_chain]
        print(
            f"[run] restricting to chain "
            f"{args.only_chain}"
        )

    pattern_counts = sorted(
        set(args.pattern_counts)
    )

    if not pattern_counts:
        raise ValueError(
            "--pattern-counts cannot be empty"
        )

    if any(P <= 0 for P in pattern_counts):
        raise ValueError(
            "all --pattern-counts must be > 0"
        )

    max_patterns = max(pattern_counts)

    patterns = candidate_patterns(
        max_patterns,
        args.chain_length,
        args.pattern_seed,
    )

    pool_rows = []
    sat_rows = []

    for chain_number, ch in enumerate(
        detected,
        start=1,
    ):
        fault = truth[ch]
        known = (
            polarity[ch]
            if args.infer_stuck_from_flush
            else None
        )

        hcell, hval = hypotheses(
            args.chain_length,
            known,
        )

        print("\n" + "-" * 80)
        print(
            f"chain {ch} "
            f"({chain_number}/{len(detected)}): "
            f"truth Q{fault.cell}:SA{fault.stuck}; "
            f"hypotheses={hcell.size:,}"
        )

        block = design.block(ch)

        R, response_total_seconds, checkpoints = (
            simulate_db(
                block,
                patterns,
                hcell,
                hval,
                args.hash_seed + ch,
                progress=args.response_progress,
                timing_checkpoints=pattern_counts,
            )
        )

        print(
            f"[response] full {max_patterns}-pattern "
            f"database: {response_total_seconds:.3f}s"
        )

        if args.mode in ("pool-scale", "all"):
            chain_pool_rows = (
                run_pool_scaling_for_chain(
                    ch,
                    fault,
                    true_index(
                        fault,
                        args.chain_length,
                        known,
                    ),
                    hcell,
                    hval,
                    R,
                    pattern_counts,
                    checkpoints,
                    args.max_selected,
                    outdir,
                )
            )
            pool_rows.extend(
                chain_pool_rows
            )

        if args.mode in (
            "targeted-sat",
            "all",
        ):
            sat_row = (
                run_targeted_sat_for_chain(
                    ch,
                    fault,
                    block,
                    hcell,
                    hval,
                    R,
                    max_patterns,
                    args,
                    outdir,
                )
            )
            sat_row["response_seconds"] = (
                float(response_total_seconds)
            )
            sat_rows.append(sat_row)

        del R

    if pool_rows:
        write_csv(
            outdir / "pool_scaling_summary.csv",
            pool_rows,
        )

    if sat_rows:
        write_csv(
            outdir / "targeted_pair_sat_summary.csv",
            sat_rows,
        )

    metadata = {
        "config": vars(args),
        "python": sys.version,
        "platform": sys.platform,
        "cpu_count": os.cpu_count(),
        "numba_threads": get_num_threads(),
        "actual_total_gates": design.total_gates,
        "gates_per_chain": design.gpc,
        "design_memory_mib": design.memory_mib,
        "design_setup_seconds": design_seconds,
        "design_cache_used": cache_used,
        "flush_seconds": flush_seconds,
        "faults": [
            asdict(f)
            for f in faults
        ],
        "detected_chains": detected,
    }

    (
        outdir / "run_metadata.json"
    ).write_text(
        json.dumps(
            metadata,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)
    print(f"results -> {outdir.resolve()}")


# =============================================================================
# CLI
# =============================================================================

def cli():
    p = argparse.ArgumentParser(
        description=(
            "Unified local simulator for the recent "
            "scan-chain diagnosis experiments."
        )
    )

    p.add_argument(
        "--mode",
        choices=[
            "pool-scale",
            "targeted-sat",
            "all",
        ],
        default="all",
    )

    p.add_argument(
        "--chains",
        type=int,
        default=1000,
    )
    p.add_argument(
        "--chain-length",
        type=int,
        default=1000,
    )
    p.add_argument(
        "--total-gates",
        type=int,
        default=10_000_000,
    )
    p.add_argument(
        "--failed-chains",
        type=int,
        default=5,
    )

    p.add_argument(
        "--pattern-counts",
        type=int,
        nargs="+",
        default=[64, 128, 256, 512, 1024],
        help=(
            "Nested candidate-pattern pool sizes. "
            "One common max-size pool is generated."
        ),
    )

    p.add_argument(
        "--infer-stuck-from-flush",
        action="store_true",
        help=(
            "Use flush polarity so each 1000-cell "
            "failed chain has L rather than 2L hypotheses."
        ),
    )
    p.add_argument(
        "--single-flush",
        action="store_true",
        help=(
            "Use only 1010... rather than both "
            "1010... and 0101...."
        ),
    )

    p.add_argument(
        "--only-chain",
        type=int,
        default=None,
        help=(
            "Analyze only one detected failing chain."
        ),
    )

    p.add_argument(
        "--max-selected",
        type=int,
        default=50,
    )

    p.add_argument(
        "--sat-conf-budget",
        type=int,
        default=500_000,
        help=(
            "Glucose conflict budget per unresolved pair. "
            "0 = unlimited."
        ),
    )

    p.add_argument(
        "--threads",
        type=int,
        default=0,
        help=(
            "Numba threads. 0 keeps Numba's default."
        ),
    )

    p.add_argument(
        "--design-cache",
        default=None,
        help=(
            "Optional .npz path. Existing matching cache "
            "is loaded; otherwise the design is built and saved."
        ),
    )

    p.add_argument(
        "--seed",
        type=int,
        default=20261008,
    )
    p.add_argument(
        "--logic-seed",
        type=int,
        default=12345,
    )
    p.add_argument(
        "--pattern-seed",
        type=int,
        default=20260929,
    )
    p.add_argument(
        "--hash-seed",
        type=int,
        default=918273,
    )

    p.add_argument(
        "--design-progress",
        type=int,
        default=100,
    )
    p.add_argument(
        "--response-progress",
        type=int,
        default=64,
    )
    p.add_argument(
        "--pair-progress",
        type=int,
        default=10,
    )

    p.add_argument(
        "--outdir",
        default="local_scan_results",
    )

    return p.parse_args()


if __name__ == "__main__":
    run(cli())
