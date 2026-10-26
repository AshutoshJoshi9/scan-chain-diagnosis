#!/usr/bin/env python3
"""
Shared-logic extension of scan_chain_diagnosis_local.py.

New capability
--------------
Allows the functional combinational block of one scan chain to depend on
scan-cell Q values belonging to other scan chains.

The main coupling parameter is

    --shared-logic-fraction F

where F is the probability that an eligible two-input gate has ONE of its
fan-ins replaced by a scan-cell Q from another chain.

Example:
    F = 0.01

means approximately 1% of eligible two-input gates contain one cross-chain
scan-cell source.

A second parameter,

    --shared-partners K

controls how many other chains each destination block is allowed to receive
cross-chain scan-cell inputs from.

Important properties
--------------------
1. Cross-chain connections originate ONLY from scan-cell Q outputs.
   Cross-block gate-output connections are intentionally not allowed.
   Therefore no combinational cycles can be introduced.

2. The original local combinational network is generated first using the
   original logic seed. Shared edges are added afterward using an independent
   --shared-seed. Thus the underlying local logic is controlled.

3. Shared-edge placement is nested across shared fractions when the same
   --shared-seed is used. For example, the connections present at F=0.001 are
   a subset of those present at F=0.01, which are a subset of F=0.10.

4. Test patterns are global: every relevant scan chain receives its own
   deterministic random scan-load vector for every pattern ID.

5. When diagnosing target chain T, the response signature contains:
       - target-chain unload response, and
       - every other chain whose logic directly receives Q values from T.
   These are called observer chains.

6. Only the target chain contains the candidate scan fault in this experiment.
   Other chains are fault-free. This isolates the effect of shared functional
   logic from multiple-failure interference.

7. The SAT miter also includes all observer blocks and all relevant external
   scan-load variables, so SAT/UNSAT remains an exact result under this
   synthetic shared-logic model.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

try:
    from numba import njit, prange, get_num_threads, set_num_threads
except Exception as exc:
    raise RuntimeError(
        "numba is required.\n"
        "Install with:\n"
        "    python -m pip install numba"
    ) from exc

# Reuse the already validated diagnosis machinery.
import scan_chain_diagnosis_local as base


AND = base.AND
OR = base.OR
XOR = base.XOR
NAND = base.NAND
NOR = base.NOR
XNOR = base.XNOR
BUF = base.BUF
NOT = base.NOT


# ============================================================================
# Utilities
# ============================================================================

def encode_external_source(chain: int, cell: int, L: int) -> int:
    """
    Local sources are >= 0.

    Cross-chain scan-cell Q sources are represented as negative integers:

        encoded = -(global_scan_index + 1)

    where
        global_scan_index = chain * L + cell
    """
    return -(int(chain) * int(L) + int(cell) + 1)


def decode_external_source(source: int, L: int) -> tuple[int, int]:
    if source >= 0:
        raise ValueError("source is not an external source")

    idx = -int(source) - 1
    return idx // int(L), idx % int(L)


def numeric_fraction_tag(x: float) -> str:
    return (
        f"{float(x):.6f}"
        .rstrip("0")
        .rstrip(".")
        .replace(".", "p")
    )


# ============================================================================
# Shared synthetic design
# ============================================================================

class SharedDesign(base.Design):
    """
    Synthetic scan design with controlled cross-chain Q dependencies.

    Local gate-generation is identical to the existing simulator. Shared
    connections are introduced only after the local blocks have been built.
    """

    def __init__(
        self,
        chains,
        L,
        total_gates,
        seed,
        shared_fraction,
        shared_partners,
        shared_seed,
        allocate=True,
    ):
        super().__init__(
            chains,
            L,
            total_gates,
            seed,
            allocate=allocate,
        )

        self.shared_fraction = float(shared_fraction)
        self.shared_partners = int(shared_partners)
        self.shared_seed = int(shared_seed)

        if allocate:
            self.partners = np.empty(
                (self.chains, self.shared_partners),
                dtype=np.int32,
            )
        else:
            self.partners = None

        self.shared_edges = 0
        self.eligible_binary_gates = 0
        self._observer_map = None

    @property
    def memory_mib(self):
        extra = 0
        if self.partners is not None:
            extra = self.partners.nbytes

        return (
            self.gtype.nbytes
            + self.f1.nbytes
            + self.f2.nbytes
            + self.capsrc.nbytes
            + extra
        ) / (1024**2)

    def _choose_partner_lists(self, rng):
        if self.shared_partners == 0:
            return

        for c in range(self.chains):
            # Choose from [0, chains-2], then skip c.
            raw = rng.choice(
                self.chains - 1,
                size=self.shared_partners,
                replace=False,
            )

            raw = raw.astype(np.int32)
            raw[raw >= c] += 1
            self.partners[c] = raw

    def build(self, progress=100):
        if not (0.0 <= self.shared_fraction <= 1.0):
            raise ValueError(
                "--shared-logic-fraction must be between 0 and 1"
            )

        if self.shared_partners < 0:
            raise ValueError(
                "--shared-partners must be >= 0"
            )

        if (
            self.shared_fraction > 0
            and self.shared_partners == 0
        ):
            raise ValueError(
                "shared fraction > 0 requires --shared-partners > 0"
            )

        if (
            self.shared_partners
            > max(0, self.chains - 1)
        ):
            raise ValueError(
                "--shared-partners cannot exceed chains - 1"
            )

        # ------------------------------------------------------------
        # 1. Build the original block-diagonal circuit first.
        # ------------------------------------------------------------
        t0 = time.perf_counter()

        super().build(progress=progress)

        if self.shared_fraction == 0.0:
            if self.shared_partners:
                rng = np.random.default_rng(self.shared_seed)
                self._choose_partner_lists(rng)

            self.shared_edges = 0
            self.eligible_binary_gates = int(
                np.count_nonzero(
                    (self.gtype != BUF)
                    & (self.gtype != NOT)
                )
            )
            self._build_observer_map()

            return time.perf_counter() - t0

        # ------------------------------------------------------------
        # 2. Generate cross-chain topology independently.
        # ------------------------------------------------------------
        rng = np.random.default_rng(
            self.shared_seed
        )

        self._choose_partner_lists(rng)

        G = self.gpc
        L = self.L

        total_shared = 0
        total_eligible = 0

        for c in range(self.chains):
            lo = c * G
            hi = (c + 1) * G

            gt = self.gtype[lo:hi]

            eligible = np.flatnonzero(
                (gt != BUF)
                & (gt != NOT)
            ).astype(np.int32)

            n = eligible.size
            total_eligible += int(n)

            if n == 0:
                continue

            # --------------------------------------------------------
            # IMPORTANT:
            # Generate ALL random topology choices before thresholding.
            # This makes the topology nested across shared fractions.
            # --------------------------------------------------------
            threshold = rng.random(n)

            chosen_input = rng.integers(
                0,
                2,
                size=n,
                dtype=np.uint8,
            )

            partner_slot = rng.integers(
                0,
                self.shared_partners,
                size=n,
                dtype=np.int32,
            )

            external_cell = rng.integers(
                0,
                L,
                size=n,
                dtype=np.int32,
            )

            use_shared = (
                threshold < self.shared_fraction
            )

            selected = np.flatnonzero(
                use_shared
            )

            for pos in selected:
                local_gate = int(eligible[pos])
                global_gate = lo + local_gate

                src_chain = int(
                    self.partners[
                        c,
                        partner_slot[pos],
                    ]
                )

                src_cell = int(
                    external_cell[pos]
                )

                encoded = encode_external_source(
                    src_chain,
                    src_cell,
                    L,
                )

                if chosen_input[pos] == 0:
                    self.f1[global_gate] = encoded
                else:
                    self.f2[global_gate] = encoded

            total_shared += int(
                selected.size
            )

            if (
                progress
                and (c + 1) % progress == 0
            ):
                print(
                    f"[shared] {c+1}/{self.chains} blocks; "
                    f"shared edges so far={total_shared:,}"
                )

        self.shared_edges = int(
            total_shared
        )
        self.eligible_binary_gates = int(
            total_eligible
        )

        self._build_observer_map()

        return time.perf_counter() - t0

    def _build_observer_map(self):
        """
        observer_map[t] contains every chain whose functional block
        directly reads at least one Q value from target chain t.

        The target chain itself is always included.
        """
        observers = [
            {c}
            for c in range(self.chains)
        ]

        G = self.gpc
        L = self.L

        for dest in range(self.chains):
            lo = dest * G
            hi = (dest + 1) * G

            s1 = self.f1[lo:hi]
            s2 = self.f2[lo:hi]

            external = np.concatenate(
                [
                    s1[s1 < 0],
                    s2[s2 < 0],
                ]
            )

            if external.size == 0:
                continue

            codes = (
                -external.astype(np.int64)
                - 1
            )
            source_chains = np.unique(
                codes // L
            )

            for src in source_chains:
                observers[int(src)].add(
                    dest
                )

        self._observer_map = [
            sorted(x)
            for x in observers
        ]

    def observer_chains(self, target):
        if self._observer_map is None:
            self._build_observer_map()

        return list(
            self._observer_map[int(target)]
        )

    def source_chains_for_block(
        self,
        dest,
    ):
        G = self.gpc
        L = self.L

        lo = int(dest) * G
        hi = (int(dest) + 1) * G

        external = np.concatenate(
            [
                self.f1[lo:hi][
                    self.f1[lo:hi] < 0
                ],
                self.f2[lo:hi][
                    self.f2[lo:hi] < 0
                ],
            ]
        )

        if external.size == 0:
            return []

        codes = (
            -external.astype(np.int64)
            - 1
        )

        return sorted(
            int(x)
            for x in np.unique(
                codes // L
            )
        )

    def relevant_chains(self, target):
        """
        Chains whose intended scan-load values may be needed when
        simulating the target and its direct observer blocks.
        """
        relevant = {
            int(target)
        }

        for obs in self.observer_chains(
            target
        ):
            relevant.add(
                int(obs)
            )

            for src in (
                self.source_chains_for_block(
                    obs
                )
            ):
                relevant.add(
                    int(src)
                )

        return sorted(relevant)

    def topology_stats(self):
        realized = (
            self.shared_edges
            / self.eligible_binary_gates
            if self.eligible_binary_gates
            else 0.0
        )

        obs_counts = np.asarray(
            [
                len(self.observer_chains(c))
                for c in range(
                    self.chains
                )
            ],
            dtype=np.int32,
        )

        return {
            "requested_shared_fraction":
                float(self.shared_fraction),
            "realized_shared_fraction":
                float(realized),
            "shared_edges":
                int(self.shared_edges),
            "eligible_binary_gates":
                int(
                    self.eligible_binary_gates
                ),
            "shared_partners":
                int(self.shared_partners),
            "observer_chains_mean":
                float(obs_counts.mean()),
            "observer_chains_min":
                int(obs_counts.min()),
            "observer_chains_max":
                int(obs_counts.max()),
        }

    def save(self, path: Path):
        path = Path(path)
        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        np.savez(
            path,
            format_version=np.int64(2),
            chains=np.int64(self.chains),
            L=np.int64(self.L),
            total_gates=np.int64(
                self.total_gates
            ),
            gpc=np.int64(self.gpc),
            seed=np.int64(self.seed),
            shared_fraction=np.float64(
                self.shared_fraction
            ),
            shared_partners=np.int64(
                self.shared_partners
            ),
            shared_seed=np.int64(
                self.shared_seed
            ),
            shared_edges=np.int64(
                self.shared_edges
            ),
            eligible_binary_gates=np.int64(
                self.eligible_binary_gates
            ),
            gtype=self.gtype,
            f1=self.f1,
            f2=self.f2,
            capsrc=self.capsrc,
            partners=self.partners,
        )

    @classmethod
    def load(
        cls,
        path: Path,
        expected=None,
    ):
        with np.load(
            path,
            allow_pickle=False,
        ) as z:
            chains = int(z["chains"])
            L = int(z["L"])
            total_gates = int(
                z["total_gates"]
            )
            seed = int(z["seed"])

            shared_fraction = float(
                z["shared_fraction"]
            )
            shared_partners = int(
                z["shared_partners"]
            )
            shared_seed = int(
                z["shared_seed"]
            )

            if expected is not None:
                (
                    exp_chains,
                    exp_L,
                    exp_gates,
                    exp_seed,
                    exp_frac,
                    exp_partners,
                    exp_shared_seed,
                ) = expected

                if (
                    chains != int(exp_chains)
                    or L != int(exp_L)
                    or total_gates
                    != int(exp_gates)
                    or seed != int(exp_seed)
                    or not np.isclose(
                        shared_fraction,
                        float(exp_frac),
                        rtol=0.0,
                        atol=1e-15,
                    )
                    or shared_partners
                    != int(exp_partners)
                    or shared_seed
                    != int(
                        exp_shared_seed
                    )
                ):
                    raise ValueError(
                        "Shared-design cache metadata "
                        "does not match current run."
                    )

            obj = cls(
                chains,
                L,
                total_gates,
                seed,
                shared_fraction,
                shared_partners,
                shared_seed,
                allocate=False,
            )

            obj.gtype = np.array(
                z["gtype"],
                copy=True,
            )
            obj.f1 = np.array(
                z["f1"],
                copy=True,
            )
            obj.f2 = np.array(
                z["f2"],
                copy=True,
            )
            obj.capsrc = np.array(
                z["capsrc"],
                copy=True,
            )
            obj.partners = np.array(
                z["partners"],
                copy=True,
            )

            obj.gpc = int(z["gpc"])
            obj.total_gates = (
                total_gates
            )

            obj.shared_edges = int(
                z["shared_edges"]
            )
            obj.eligible_binary_gates = (
                int(
                    z[
                        "eligible_binary_gates"
                    ]
                )
            )

            obj._build_observer_map()

            return obj


def get_or_build_shared_design(args):
    actual_gates = (
        int(
            round(
                args.total_gates
                / args.chains
            )
        )
        * args.chains
    )

    expected = (
        args.chains,
        args.chain_length,
        actual_gates,
        args.logic_seed,
        args.shared_logic_fraction,
        args.shared_partners,
        args.shared_seed,
    )

    cache = (
        Path(args.design_cache)
        if args.design_cache
        else None
    )

    if (
        cache is not None
        and cache.exists()
    ):
        print(
            f"[design] loading shared cache: "
            f"{cache}"
        )

        t0 = time.perf_counter()

        design = SharedDesign.load(
            cache,
            expected=expected,
        )

        dt = (
            time.perf_counter()
            - t0
        )

        print(
            f"[design] loaded "
            f"{design.total_gates:,} gates "
            f"in {dt:.2f}s"
        )

        return design, dt, True

    design = SharedDesign(
        args.chains,
        args.chain_length,
        args.total_gates,
        args.logic_seed,
        args.shared_logic_fraction,
        args.shared_partners,
        args.shared_seed,
    )

    print(
        f"[design] building "
        f"{design.total_gates:,} gates"
    )

    dt = design.build(
        args.design_progress
    )

    stats = design.topology_stats()

    print(
        "[shared] "
        f"requested="
        f"{stats['requested_shared_fraction']:.6f}, "
        f"realized="
        f"{stats['realized_shared_fraction']:.6f}, "
        f"edges="
        f"{stats['shared_edges']:,}"
    )

    if cache is not None:
        print(
            f"[design] saving cache: "
            f"{cache}"
        )
        design.save(cache)

    return design, dt, False


# ============================================================================
# Global deterministic scan-pattern bank
# ============================================================================

def chain_pattern_pool(
    P,
    L,
    global_seed,
    chain,
):
    """
    Every physical scan chain gets its own deterministic random pattern
    sequence. Recreating the same chain with the same seed returns exactly
    the same data.

    Therefore candidate pattern ID p corresponds to one consistent global
    scan-load state across the chip.
    """
    ss = np.random.SeedSequence(
        [
            int(global_seed),
            int(chain),
            0x51A7C0DE,
        ]
    )

    rng = np.random.default_rng(ss)

    return rng.integers(
        0,
        2,
        size=(int(P), int(L)),
        dtype=np.uint8,
    )


@dataclass
class SharedContext:
    design: SharedDesign
    target: int
    observers: list[int]
    relevant: list[int]
    chain_to_row: np.ndarray
    pattern_bank: np.ndarray


def make_shared_context(
    design,
    target,
    P,
    pattern_seed,
):
    observers = (
        design.observer_chains(
            target
        )
    )

    relevant = (
        design.relevant_chains(
            target
        )
    )

    chain_to_row = np.full(
        design.chains,
        -1,
        dtype=np.int32,
    )

    bank = np.empty(
        (
            len(relevant),
            P,
            design.L,
        ),
        dtype=np.uint8,
    )

    for row, c in enumerate(
        relevant
    ):
        chain_to_row[c] = row

        bank[row] = (
            chain_pattern_pool(
                P,
                design.L,
                pattern_seed,
                c,
            )
        )

    return SharedContext(
        design=design,
        target=int(target),
        observers=observers,
        relevant=relevant,
        chain_to_row=chain_to_row,
        pattern_bank=bank,
    )


# ============================================================================
# Shared functional simulation
# ============================================================================

@njit(cache=True, parallel=True)
def simulate_shared_block(
    gtype,
    f1,
    f2,
    capsrc,
    local_state,
    target_state,
    target_chain,
    chain_to_row,
    global_pattern_bits,
    L,
    valid,
    gate_values,
    capture,
):
    """
    Evaluate one destination block for all fault hypotheses in parallel.

    Positive source:
        local scan cell or local gate.

    Negative source:
        Q of another scan chain.
    """
    _, W = local_state.shape
    G = gtype.size

    for w in prange(W):
        vw = valid[w]

        for g in range(G):
            s1 = f1[g]

            if s1 >= 0:
                if s1 < L:
                    a = local_state[
                        s1, w
                    ]
                else:
                    a = gate_values[
                        s1 - L, w
                    ]

            else:
                code = -s1 - 1
                src_chain = code // L
                src_cell = (
                    code
                    - src_chain * L
                )

                if (
                    src_chain
                    == target_chain
                ):
                    a = target_state[
                        src_cell,
                        w,
                    ]
                else:
                    row = chain_to_row[
                        src_chain
                    ]
                    bit = (
                        global_pattern_bits[
                            row,
                            src_cell,
                        ]
                    )

                    a = (
                        vw
                        if bit
                        else np.uint64(0)
                    )

            typ = gtype[g]

            if typ == BUF:
                y = a

            elif typ == NOT:
                y = ~a

            else:
                s2 = f2[g]

                if s2 >= 0:
                    if s2 < L:
                        b = local_state[
                            s2,
                            w,
                        ]
                    else:
                        b = gate_values[
                            s2 - L,
                            w,
                        ]

                else:
                    code = -s2 - 1
                    src_chain = (
                        code // L
                    )
                    src_cell = (
                        code
                        - src_chain * L
                    )

                    if (
                        src_chain
                        == target_chain
                    ):
                        b = target_state[
                            src_cell,
                            w,
                        ]
                    else:
                        row = (
                            chain_to_row[
                                src_chain
                            ]
                        )
                        bit = (
                            global_pattern_bits[
                                row,
                                src_cell,
                            ]
                        )

                        b = (
                            vw
                            if bit
                            else np.uint64(0)
                        )

                if typ == AND:
                    y = a & b
                elif typ == OR:
                    y = a | b
                elif typ == XOR:
                    y = a ^ b
                elif typ == NAND:
                    y = ~(a & b)
                elif typ == NOR:
                    y = ~(a | b)
                else:
                    y = ~(a ^ b)

            gate_values[g, w] = y

        for q in range(L):
            src = capsrc[q]

            if src < L:
                capture[q, w] = (
                    local_state[
                        src,
                        w,
                    ]
                )
            else:
                capture[q, w] = (
                    gate_values[
                        src - L,
                        w,
                    ]
                )


def fault_free_packed_state(
    pattern,
    valid,
):
    """
    Repeat one fault-free scan state across every hypothesis bit.
    """
    one = (
        np.asarray(
            pattern,
            dtype=np.uint8,
        )
        != 0
    )

    L = one.size
    W = valid.size

    out = np.zeros(
        (L, W),
        dtype=np.uint64,
    )

    if np.any(one):
        out[one] = valid[None, :]

    return out


def capture_only_hash(
    capture,
    H,
    L,
    salt1,
    salt2,
):
    """
    Hash complete fault-free unload response.

    hcell=L ensures response_hash never activates the stuck tail.
    """
    dummy_cell = np.full(
        H,
        L,
        dtype=np.int32,
    )

    dummy_val = np.zeros(
        H,
        dtype=np.uint8,
    )

    return base.response_hash(
        capture,
        dummy_cell,
        dummy_val,
        salt1,
        salt2,
    )


def simulate_shared_db(
    context,
    hcell,
    hval,
    hash_seed,
    progress=32,
    timing_checkpoints=None,
):
    """
    Generate multi-chain response fingerprints.

    One signature includes the target-chain unload and every observer-chain
    unload.
    """
    design = context.design
    target = context.target

    bank = context.pattern_bank

    P = bank.shape[1]
    L = design.L
    H = hcell.size

    f0, ff1, valid = (
        base.force_masks(
            L,
            hcell,
            hval,
        )
    )

    W = valid.size

    out = np.empty(
        (P, H, 2),
        dtype=np.uint64,
    )

    # Per-observer working buffers.
    gate_buffers = {}
    capture_buffers = {}
    observer_salts = {}

    for obs in context.observers:
        gt, _, _, _ = (
            design.block(obs)
        )

        gate_buffers[obs] = (
            np.empty(
                (gt.size, W),
                dtype=np.uint64,
            )
        )

        capture_buffers[obs] = (
            np.empty(
                (L, W),
                dtype=np.uint64,
            )
        )

        observer_salts[obs] = (
            base.salts(
                (L + 63) // 64,
                int(hash_seed)
                + 1_000_003
                * (int(obs) + 1),
            )
        )

    checkpoints = set(
        timing_checkpoints or []
    )
    checkpoint_times = {}

    target_row = (
        context.chain_to_row[
            target
        ]
    )

    # Compile before timing.
    target_state = (
        base.packed_loaded_state(
            bank[target_row, 0],
            f0,
            ff1,
            valid,
        )
    )

    first_obs = (
        context.observers[0]
    )

    gt, s1, s2, capsrc = (
        design.block(first_obs)
    )

    if first_obs == target:
        local_state = target_state
    else:
        row = (
            context.chain_to_row[
                first_obs
            ]
        )
        local_state = (
            fault_free_packed_state(
                bank[row, 0],
                valid,
            )
        )

    simulate_shared_block(
        gt,
        s1,
        s2,
        capsrc,
        local_state,
        target_state,
        target,
        context.chain_to_row,
        bank[:, 0, :],
        L,
        valid,
        gate_buffers[first_obs],
        capture_buffers[first_obs],
    )

    t0 = time.perf_counter()

    for p in range(P):
        target_state = (
            base.packed_loaded_state(
                bank[
                    target_row,
                    p,
                ],
                f0,
                ff1,
                valid,
            )
        )

        sig_a = np.zeros(
            H,
            dtype=np.uint64,
        )
        sig_b = np.zeros(
            H,
            dtype=np.uint64,
        )

        for obs in context.observers:
            (
                gt,
                s1,
                s2,
                capsrc,
            ) = design.block(obs)

            if obs == target:
                local_state = (
                    target_state
                )
            else:
                row = (
                    context.chain_to_row[
                        obs
                    ]
                )

                local_state = (
                    fault_free_packed_state(
                        bank[
                            row,
                            p,
                        ],
                        valid,
                    )
                )

            gv = gate_buffers[obs]
            cap = capture_buffers[obs]

            simulate_shared_block(
                gt,
                s1,
                s2,
                capsrc,
                local_state,
                target_state,
                target,
                context.chain_to_row,
                bank[:, p, :],
                L,
                valid,
                gv,
                cap,
            )

            cap[:, -1] &= (
                valid[-1]
            )

            salt1, salt2 = (
                observer_salts[obs]
            )

            if obs == target:
                a, b = (
                    base.response_hash(
                        cap,
                        hcell,
                        hval,
                        salt1,
                        salt2,
                    )
                )
            else:
                a, b = (
                    capture_only_hash(
                        cap,
                        H,
                        L,
                        salt1,
                        salt2,
                    )
                )

            sig_a, sig_b = (
                base.combine(
                    sig_a,
                    sig_b,
                    a,
                    b,
                )
            )

        out[p, :, 0] = sig_a
        out[p, :, 1] = sig_b

        completed = p + 1

        if completed in checkpoints:
            checkpoint_times[
                completed
            ] = (
                time.perf_counter()
                - t0
            )

        if (
            progress
            and completed % progress
            == 0
        ):
            dt = (
                time.perf_counter()
                - t0
            )

            print(
                f"    [shared response] "
                f"{completed}/{P}, "
                f"{completed/max(dt,1e-12):.2f} "
                f"patterns/s"
            )

    total_seconds = (
        time.perf_counter()
        - t0
    )

    return (
        out,
        total_seconds,
        checkpoint_times,
    )


# ============================================================================
# Shared-logic SAT miter
# ============================================================================

def shared_source_var(
    source,
    observer,
    target,
    L,
    x_by_chain,
    target_state,
    gate_vars,
):
    """
    Convert one encoded circuit source into the corresponding SAT variable.
    """
    source = int(source)

    if source >= 0:
        if source < L:
            if observer == target:
                return int(
                    target_state[
                        source
                    ]
                )

            return int(
                x_by_chain[
                    observer
                ][source]
            )

        return int(
            gate_vars[
                source - L
            ]
        )

    src_chain, src_cell = (
        decode_external_source(
            source,
            L,
        )
    )

    if src_chain == target:
        return int(
            target_state[
                src_cell
            ]
        )

    return int(
        x_by_chain[
            src_chain
        ][src_cell]
    )


def build_shared_pair_cnf(
    context,
    hcell,
    hval,
    i,
    j,
):
    CNF, _ = base.import_sat()

    design = context.design
    target = context.target
    L = design.L
    G = design.gpc

    ni = int(hcell[i])
    vi = int(hval[i])

    nj = int(hcell[j])
    vj = int(hval[j])

    cnf = CNF()
    nxt = 1

    # ------------------------------------------------------------
    # Intended scan-load variables for every relevant chain.
    # ------------------------------------------------------------
    x_by_chain = {}

    for c in context.relevant:
        arr, nxt = base.alloc_vars(
            nxt,
            L,
        )
        x_by_chain[c] = arr

    # ------------------------------------------------------------
    # Two faulty loaded states for target chain.
    # ------------------------------------------------------------
    si, nxt = base.alloc_vars(
        nxt,
        L,
    )
    sj, nxt = base.alloc_vars(
        nxt,
        L,
    )

    for q in range(L):
        if q <= ni:
            cnf.append(
                [int(si[q])]
                if vi
                else [-int(si[q])]
            )
        else:
            base.add_equiv(
                cnf,
                int(si[q]),
                int(
                    x_by_chain[
                        target
                    ][q]
                ),
            )

        if q <= nj:
            cnf.append(
                [int(sj[q])]
                if vj
                else [-int(sj[q])]
            )
        else:
            base.add_equiv(
                cnf,
                int(sj[q]),
                int(
                    x_by_chain[
                        target
                    ][q]
                ),
            )

    # ------------------------------------------------------------
    # Functional logic for every observer chain, duplicated for
    # the two candidate target faults.
    # ------------------------------------------------------------
    gi_by_obs = {}
    gj_by_obs = {}

    for obs in context.observers:
        gi, nxt = base.alloc_vars(
            nxt,
            G,
        )
        gj, nxt = base.alloc_vars(
            nxt,
            G,
        )

        gi_by_obs[obs] = gi
        gj_by_obs[obs] = gj

        (
            gtype,
            f1,
            f2,
            _,
        ) = design.block(obs)

        for g in range(G):
            typ = int(gtype[g])

            ai = shared_source_var(
                f1[g],
                obs,
                target,
                L,
                x_by_chain,
                si,
                gi,
            )

            aj = shared_source_var(
                f1[g],
                obs,
                target,
                L,
                x_by_chain,
                sj,
                gj,
            )

            if typ in (
                BUF,
                NOT,
            ):
                bi = None
                bj = None
            else:
                bi = shared_source_var(
                    f2[g],
                    obs,
                    target,
                    L,
                    x_by_chain,
                    si,
                    gi,
                )

                bj = shared_source_var(
                    f2[g],
                    obs,
                    target,
                    L,
                    x_by_chain,
                    sj,
                    gj,
                )

            base.add_gate(
                cnf,
                typ,
                ai,
                bi,
                int(gi[g]),
            )

            base.add_gate(
                cnf,
                typ,
                aj,
                bj,
                int(gj[g]),
            )

    # ------------------------------------------------------------
    # Require at least one visible unload bit, in any observer
    # chain, to differ.
    # ------------------------------------------------------------
    diff_literals = []
    always_different = False

    for obs in context.observers:
        _, _, _, capsrc = (
            design.block(obs)
        )

        for k in range(L):
            if obs == target:
                # Existing target-chain unload fault semantics.
                if k >= ni:
                    ri_const = vi
                    ri_var = None
                else:
                    ri_const = None
                    ri_var = (
                        shared_source_var(
                            capsrc[k],
                            obs,
                            target,
                            L,
                            x_by_chain,
                            si,
                            gi_by_obs[
                                obs
                            ],
                        )
                    )

                if k >= nj:
                    rj_const = vj
                    rj_var = None
                else:
                    rj_const = None
                    rj_var = (
                        shared_source_var(
                            capsrc[k],
                            obs,
                            target,
                            L,
                            x_by_chain,
                            sj,
                            gj_by_obs[
                                obs
                            ],
                        )
                    )

            else:
                ri_const = None
                rj_const = None

                ri_var = (
                    shared_source_var(
                        capsrc[k],
                        obs,
                        target,
                        L,
                        x_by_chain,
                        si,
                        gi_by_obs[
                            obs
                        ],
                    )
                )

                rj_var = (
                    shared_source_var(
                        capsrc[k],
                        obs,
                        target,
                        L,
                        x_by_chain,
                        sj,
                        gj_by_obs[
                            obs
                        ],
                    )
                )

            if (
                ri_var is None
                and rj_var is None
            ):
                if (
                    ri_const
                    != rj_const
                ):
                    always_different = (
                        True
                    )
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

            base.add_xor_var(
                cnf,
                int(ri_var),
                int(rj_var),
                int(d),
            )

            diff_literals.append(
                int(d)
            )

        if always_different:
            break

    if not always_different:
        if not diff_literals:
            # Explicit contradiction.
            cnf.append([])
        else:
            cnf.append(
                diff_literals
            )

    ordered_relevant = list(
        context.relevant
    )

    return cnf, {
        "nvars": nxt - 1,
        "nclauses": len(
            cnf.clauses
        ),
        "fault_i": (ni, vi),
        "fault_j": (nj, vj),
        "relevant_chains":
            ordered_relevant,
        "x_by_chain":
            x_by_chain,
        "always_different":
            always_different,
    }


def solve_shared_pair(
    context,
    hcell,
    hval,
    i,
    j,
    conf_budget,
):
    _, Glucose4 = (
        base.import_sat()
    )

    cnf, meta = (
        build_shared_pair_cnf(
            context,
            hcell,
            hval,
            i,
            j,
        )
    )

    t0 = time.perf_counter()

    with Glucose4(
        bootstrap_with=cnf.clauses
    ) as solver:

        if (
            conf_budget
            and conf_budget > 0
        ):
            solver.conf_budget(
                int(conf_budget)
            )

            sat = solver.solve_limited(
                expect_interrupt=True
            )
        else:
            sat = solver.solve()

        elapsed = (
            time.perf_counter()
            - t0
        )

        if sat is True:
            model = solver.get_model()

            positives = {
                lit
                for lit in model
                if lit > 0
            }

            witness = {}

            for c in (
                meta[
                    "relevant_chains"
                ]
            ):
                x = (
                    meta[
                        "x_by_chain"
                    ][c]
                )

                witness[c] = (
                    np.fromiter(
                        (
                            1
                            if int(v)
                            in positives
                            else 0
                            for v in x
                        ),
                        dtype=np.uint8,
                        count=len(x),
                    )
                )

            status = "SAT"

        elif sat is False:
            witness = None
            status = "UNSAT"

        else:
            witness = None
            status = "UNKNOWN"

    return (
        status,
        witness,
        elapsed,
        meta,
    )


def explicit_witness_context(
    original_context,
    witness,
):
    """
    Construct a one-pattern context directly from SAT's complete relevant
    scan-load assignment.
    """
    relevant = (
        original_context.relevant
    )

    L = (
        original_context.design.L
    )

    bank = np.empty(
        (
            len(relevant),
            1,
            L,
        ),
        dtype=np.uint8,
    )

    for row, c in enumerate(
        relevant
    ):
        bank[row, 0] = (
            witness[c]
        )

    return SharedContext(
        design=(
            original_context.design
        ),
        target=(
            original_context.target
        ),
        observers=list(
            original_context.observers
        ),
        relevant=list(relevant),
        chain_to_row=(
            original_context
            .chain_to_row.copy()
        ),
        pattern_bank=bank,
    )


def verify_shared_witness(
    context,
    witness,
    hcell,
    hval,
    i,
    j,
    hash_seed,
):
    pair_cell = np.asarray(
        [
            hcell[i],
            hcell[j],
        ],
        dtype=np.int32,
    )

    pair_val = np.asarray(
        [
            hval[i],
            hval[j],
        ],
        dtype=np.uint8,
    )

    wctx = explicit_witness_context(
        context,
        witness,
    )

    R, _, _ = (
        simulate_shared_db(
            wctx,
            pair_cell,
            pair_val,
            hash_seed,
            progress=0,
        )
    )

    return not (
        R[0, 0, 0]
        == R[0, 1, 0]
        and
        R[0, 0, 1]
        == R[0, 1, 1]
    )


def witness_key(
    context,
    witness,
):
    payload = bytearray()

    for c in context.relevant:
        payload.extend(
            np.packbits(
                witness[c],
                bitorder="little",
            ).tobytes()
        )

    return bytes(payload)


# ============================================================================
# Shared targeted SAT stage
# ============================================================================

def run_shared_targeted_sat(
    ch,
    fault,
    context,
    hcell,
    hval,
    R,
    base_patterns,
    args,
    outdir,
):
    classes = (
        base.ambiguity_classes(
            R
        )
    )

    pairs = (
        base.all_pairs_from_classes(
            classes
        )
    )

    active = sorted(
        {
            x
            for pair in pairs
            for x in pair
        }
    )

    (
        base_classes,
        base_unresolved,
    ) = base.max_partition(R)

    base.save_ambiguity_classes(
        Path(outdir)
        / (
            f"chain_{ch:04d}_"
            "ambiguity_classes.json"
        ),
        classes,
        hcell,
        hval,
    )

    print(
        f"[SAT] chain {ch}: "
        f"base={base_classes}/"
        f"{hcell.size}, "
        f"observers="
        f"{context.observers}, "
        f"pair queries={len(pairs)}"
    )

    pair_rows = []

    witnesses = []
    witness_id_by_key = {}

    sat_count = 0
    unsat_count = 0
    unknown_count = 0
    verified_count = 0

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
        ) = solve_shared_pair(
            context,
            hcell,
            hval,
            i,
            j,
            args.sat_conf_budget,
        )

        sat_total_seconds += (
            sat_seconds
        )

        verified = False
        witness_id = None

        if status == "SAT":
            sat_count += 1

            verified = (
                verify_shared_witness(
                    context,
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
            )

            if verified:
                verified_count += 1

                key = witness_key(
                    context,
                    witness,
                )

                if (
                    key
                    not in witness_id_by_key
                ):
                    witness_id_by_key[
                        key
                    ] = len(
                        witnesses
                    )

                    witnesses.append(
                        {
                            c:
                            witness[c].copy()
                            for c in (
                                context.relevant
                            )
                        }
                    )

                witness_id = (
                    witness_id_by_key[
                        key
                    ]
                )

        elif status == "UNSAT":
            unsat_count += 1

        else:
            unknown_count += 1

        pair_rows.append(
            {
                "chain": ch,
                "pair_query":
                    pair_idx,
                "hypothesis_i": i,
                "cell_i":
                    int(hcell[i]),
                "stuck_i":
                    int(hval[i]),
                "hypothesis_j": j,
                "cell_j":
                    int(hcell[j]),
                "stuck_j":
                    int(hval[j]),
                "status": status,
                "verified":
                    int(verified),
                "witness_id":
                    witness_id,
                "sat_seconds":
                    float(
                        sat_seconds
                    ),
                "cnf_vars":
                    int(
                        meta[
                            "nvars"
                        ]
                    ),
                "cnf_clauses":
                    int(
                        meta[
                            "nclauses"
                        ]
                    ),
                "observer_chains":
                    len(
                        context.observers
                    ),
                "relevant_chains":
                    len(
                        context.relevant
                    ),
            }
        )

        if (
            args.pair_progress
            and (
                pair_idx
                % args.pair_progress
                == 0
                or pair_idx
                == len(pairs)
            )
        ):
            print(
                f"    [SAT] "
                f"{pair_idx}/"
                f"{len(pairs)} "
                f"SAT={sat_count}, "
                f"UNSAT={unsat_count}, "
                f"UNKNOWN="
                f"{unknown_count}"
            )

    base.write_csv(
        Path(outdir)
        / (
            f"chain_{ch:04d}_"
            "pair_sat.csv"
        ),
        pair_rows,
    )

    witness_sim_seconds = 0.0

    if witnesses:
        witness_responses = []

        t0 = time.perf_counter()

        for wid, witness in enumerate(
            witnesses
        ):
            wctx = (
                explicit_witness_context(
                    context,
                    witness,
                )
            )

            WR, _, _ = (
                simulate_shared_db(
                    wctx,
                    hcell,
                    hval,
                    args.hash_seed
                    + ch
                    + 20_000_000
                    + wid,
                    progress=0,
                )
            )

            witness_responses.append(
                WR
            )

        witness_sim_seconds = (
            time.perf_counter()
            - t0
        )

        augmented = np.concatenate(
            [R]
            + witness_responses,
            axis=0,
        )

        (
            final_classes,
            final_unresolved,
        ) = base.max_partition(
            augmented
        )

        # Store complete witness assignments.
        save_payload = {
            "relevant_chains":
                np.asarray(
                    context.relevant,
                    dtype=np.int32,
                )
        }

        for wid, witness in enumerate(
            witnesses
        ):
            stacked = np.stack(
                [
                    witness[c]
                    for c in (
                        context.relevant
                    )
                ],
                axis=0,
            )

            save_payload[
                f"witness_{wid:04d}"
            ] = stacked

        np.savez_compressed(
            Path(outdir)
            / (
                f"chain_{ch:04d}_"
                "sat_witnesses.npz"
            ),
            **save_payload,
        )

    else:
        final_classes = (
            base_classes
        )
        final_unresolved = (
            base_unresolved
        )

    print(
        f"[SAT] result chain {ch}: "
        f"{base_classes}"
        f"->{final_classes} classes, "
        f"SAT={sat_count}, "
        f"UNSAT={unsat_count}, "
        f"UNKNOWN={unknown_count}"
    )

    return {
        "chain": ch,
        "true_cell":
            fault.cell,
        "true_stuck":
            fault.stuck,
        "hypotheses":
            int(hcell.size),
        "base_patterns":
            int(base_patterns),
        "base_classes":
            int(base_classes),
        "base_unresolved_pairs":
            int(base_unresolved),
        "ambiguous_classes":
            int(len(classes)),
        "active_ambiguous_hypotheses":
            int(len(active)),
        "pair_queries":
            int(len(pairs)),
        "sat_pairs":
            int(sat_count),
        "unsat_pairs":
            int(unsat_count),
        "unknown_pairs":
            int(unknown_count),
        "verified_sat_pairs":
            int(verified_count),
        "unique_witness_patterns":
            int(len(witnesses)),
        "final_classes":
            int(final_classes),
        "final_unresolved_pairs":
            int(final_unresolved),
        "sat_total_seconds":
            float(
                sat_total_seconds
            ),
        "witness_sim_seconds":
            float(
                witness_sim_seconds
            ),
        "observer_chains":
            int(
                len(
                    context.observers
                )
            ),
        "relevant_chains":
            int(
                len(
                    context.relevant
                )
            ),
        "observer_chain_ids":
            " ".join(
                str(x)
                for x in (
                    context.observers
                )
            ),
    }


# ============================================================================
# Topology CSV
# ============================================================================

def write_topology_csv(
    design,
    path,
):
    rows = []

    G = design.gpc
    L = design.L

    for dest in range(
        design.chains
    ):
        lo = dest * G
        hi = (dest + 1) * G

        sources = []

        for arr in (
            design.f1[lo:hi],
            design.f2[lo:hi],
        ):
            ext = arr[arr < 0]

            for src in ext:
                c, q = (
                    decode_external_source(
                        src,
                        L,
                    )
                )
                sources.append(
                    (c, q)
                )

        by_source = {}

        for c, q in sources:
            by_source.setdefault(
                c,
                [],
            ).append(q)

        for partner in (
            design.partners[dest]
            if design.shared_partners
            else []
        ):
            partner = int(partner)

            cells = by_source.get(
                partner,
                [],
            )

            rows.append(
                {
                    "destination_chain":
                        dest,
                    "source_chain":
                        partner,
                    "shared_edges":
                        len(cells),
                    "unique_source_cells":
                        len(
                            set(cells)
                        ),
                }
            )

    base.write_csv(
        path,
        rows,
    )


# ============================================================================
# Main experiment
# ============================================================================

def run(args):
    outdir = Path(args.outdir)
    outdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    if args.threads:
        set_num_threads(
            min(
                int(args.threads),
                int(
                    get_num_threads()
                ),
            )
        )

    print("=" * 80)
    print(
        "SHARED-LOGIC SCAN-CHAIN "
        "DIAGNOSIS SIMULATOR"
    )
    print("=" * 80)

    print(
        f"mode                  : "
        f"{args.mode}"
    )
    print(
        f"chains                : "
        f"{args.chains}"
    )
    print(
        f"chain length          : "
        f"{args.chain_length}"
    )
    print(
        f"total scan flops      : "
        f"{args.chains * args.chain_length:,}"
    )
    print(
        f"requested gates       : "
        f"{args.total_gates:,}"
    )
    print(
        f"failed chains         : "
        f"{args.failed_chains}"
    )
    print(
        f"shared fraction       : "
        f"{args.shared_logic_fraction}"
    )
    print(
        f"shared partners       : "
        f"{args.shared_partners}"
    )
    print(
        f"logic seed            : "
        f"{args.logic_seed}"
    )
    print(
        f"shared seed           : "
        f"{args.shared_seed}"
    )
    print(
        f"pattern seed          : "
        f"{args.pattern_seed}"
    )

    if args.mode in (
        "targeted-sat",
        "all",
    ):
        base.import_sat()

    design, design_seconds, cache_used = (
        get_or_build_shared_design(
            args
        )
    )

    topology_stats = (
        design.topology_stats()
    )

    write_topology_csv(
        design,
        outdir
        / "shared_logic_topology.csv",
    )

    rng = np.random.default_rng(
        args.seed
    )

    faults = base.inject_faults(
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
        key=lambda x: x.chain,
    ):
        print(
            f"  chain {f.chain:4d}: "
            f"Q{f.cell}:SA{f.stuck}"
        )

    t0 = time.perf_counter()

    detected, polarity = (
        base.flush_screen(
            args.chains,
            args.chain_length,
            faults,
            complementary=(
                not args.single_flush
            ),
        )
    )

    flush_seconds = (
        time.perf_counter()
        - t0
    )

    print(
        f"\n[flush] detected "
        f"{len(detected)}/"
        f"{len(faults)}: "
        f"{detected}"
    )

    if args.only_chain is not None:
        if (
            args.only_chain
            not in detected
        ):
            raise ValueError(
                "--only-chain must be "
                "one of the detected "
                "failing chains"
            )

        detected = [
            args.only_chain
        ]

    pattern_counts = sorted(
        set(
            int(x)
            for x in (
                args.pattern_counts
            )
        )
    )

    if not pattern_counts:
        raise ValueError(
            "--pattern-counts cannot "
            "be empty"
        )

    if any(
        P <= 0
        for P in pattern_counts
    ):
        raise ValueError(
            "pattern counts must be > 0"
        )

    max_patterns = max(
        pattern_counts
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
            if (
                args.infer_stuck_from_flush
            )
            else None
        )

        hcell, hval = (
            base.hypotheses(
                args.chain_length,
                known,
            )
        )

        context = (
            make_shared_context(
                design,
                ch,
                max_patterns,
                args.pattern_seed,
            )
        )

        print(
            "\n"
            + "-" * 80
        )

        print(
            f"chain {ch} "
            f"({chain_number}/"
            f"{len(detected)}): "
            f"truth "
            f"Q{fault.cell}:SA"
            f"{fault.stuck}; "
            f"H={hcell.size}; "
            f"observers="
            f"{context.observers}; "
            f"relevant="
            f"{len(context.relevant)}"
        )

        (
            R,
            response_total_seconds,
            checkpoints,
        ) = simulate_shared_db(
            context,
            hcell,
            hval,
            args.hash_seed + ch,
            progress=(
                args.response_progress
            ),
            timing_checkpoints=(
                pattern_counts
            ),
        )

        print(
            f"[response] "
            f"{max_patterns} patterns "
            f"in "
            f"{response_total_seconds:.3f}s"
        )

        if args.mode in (
            "pool-scale",
            "all",
        ):
            rows = (
                base.run_pool_scaling_for_chain(
                    ch,
                    fault,
                    base.true_index(
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

            for row in rows:
                row[
                    "shared_logic_fraction"
                ] = float(
                    args.shared_logic_fraction
                )

                row[
                    "realized_shared_fraction"
                ] = float(
                    topology_stats[
                        "realized_shared_fraction"
                    ]
                )

                row[
                    "observer_chains"
                ] = len(
                    context.observers
                )

                row[
                    "relevant_chains"
                ] = len(
                    context.relevant
                )

            pool_rows.extend(rows)

        if args.mode in (
            "targeted-sat",
            "all",
        ):
            sat_row = (
                run_shared_targeted_sat(
                    ch,
                    fault,
                    context,
                    hcell,
                    hval,
                    R,
                    max_patterns,
                    args,
                    outdir,
                )
            )

            sat_row[
                "response_seconds"
            ] = float(
                response_total_seconds
            )

            sat_row[
                "shared_logic_fraction"
            ] = float(
                args.shared_logic_fraction
            )

            sat_row[
                "realized_shared_fraction"
            ] = float(
                topology_stats[
                    "realized_shared_fraction"
                ]
            )

            sat_rows.append(
                sat_row
            )

        del R
        del context

    if pool_rows:
        base.write_csv(
            outdir
            / "pool_scaling_summary.csv",
            pool_rows,
        )

    if sat_rows:
        base.write_csv(
            outdir
            / (
                "targeted_pair_"
                "sat_summary.csv"
            ),
            sat_rows,
        )

    metadata = {
        "config": vars(args),
        "python": sys.version,
        "platform": sys.platform,
        "cpu_count": os.cpu_count(),
        "numba_threads":
            get_num_threads(),
        "actual_total_gates":
            design.total_gates,
        "gates_per_chain":
            design.gpc,
        "design_memory_mib":
            design.memory_mib,
        "design_setup_seconds":
            design_seconds,
        "design_cache_used":
            cache_used,
        "flush_seconds":
            flush_seconds,
        "faults": [
            asdict(f)
            for f in faults
        ],
        "detected_chains":
            detected,
        "shared_logic":
            topology_stats,
    }

    (
        outdir
        / "run_metadata.json"
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

    print(
        f"results -> "
        f"{outdir.resolve()}"
    )


# ============================================================================
# CLI
# ============================================================================

def cli():
    p = argparse.ArgumentParser(
        description=(
            "Shared-logic extension of "
            "scan_chain_diagnosis_local.py"
        )
    )

    p.add_argument(
        "--mode",
        choices=[
            "pool-scale",
            "targeted-sat",
            "all",
        ],
        default="pool-scale",
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
        default=10,
    )

    p.add_argument(
        "--pattern-counts",
        type=int,
        nargs="+",
        default=[
            64,
            128,
            256,
            512,
            1024,
        ],
    )

    p.add_argument(
        "--shared-logic-fraction",
        type=float,
        default=0.0,
        help=(
            "Fraction of eligible binary "
            "gates whose one fan-in is "
            "replaced by a Q value from "
            "another scan chain."
        ),
    )

    p.add_argument(
        "--shared-partners",
        type=int,
        default=4,
        help=(
            "Maximum source-chain partner "
            "set for each destination block."
        ),
    )

    p.add_argument(
        "--shared-seed",
        type=int,
        default=314159,
        help=(
            "Controls partner lists and "
            "cross-chain topology."
        ),
    )

    p.add_argument(
        "--infer-stuck-from-flush",
        action="store_true",
    )

    p.add_argument(
        "--single-flush",
        action="store_true",
    )

    p.add_argument(
        "--only-chain",
        type=int,
        default=None,
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
    )

    p.add_argument(
        "--threads",
        type=int,
        default=0,
    )

    p.add_argument(
        "--design-cache",
        default=None,
    )

    p.add_argument(
        "--seed",
        type=int,
        default=20261008,
        help="Fault seed.",
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
        default=(
            "shared_logic_results"
        ),
    )

    return p.parse_args()


if __name__ == "__main__":
    run(cli())