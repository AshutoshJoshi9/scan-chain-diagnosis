#!/usr/bin/env python3
"""
Circuit visualization helpers for the scan-chain diagnosis simulator — focused viewer v5.

Uses the open-source Graphviz engine to turn the simulator's actual synthetic
logic arrays into local schematics.

The viewer intentionally renders LOCAL logic, not the entire 10-million-gate
design at once. Useful views:
  - scan-chain neighborhood
  - fan-in cone driving D(Q_n)
  - fan-out cone starting at Q_n
  - combined fan-in + fan-out
  - ambiguity-class comparison

Index convention:
  Q0 is nearest scan-out.
  Scan shift direction is:
      Scan-In -> Q(L-1) -> ... -> Q1 -> Q0 -> Scan-Out
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from collections import defaultdict, deque
import shutil
import os

import numpy as np

import scan_chain_diagnosis_local as sim


GATE_LABELS = {
    sim.AND: "AND",
    sim.OR: "OR",
    sim.XOR: "XOR",
    sim.NAND: "NAND",
    sim.NOR: "NOR",
    sim.XNOR: "XNOR",
    sim.BUF: "BUF",
    sim.NOT: "NOT",
}


@dataclass
class CircuitViewResult:
    dot_path: Path
    png_path: Path | None
    svg_path: Path | None
    node_count: int
    gate_count: int
    scan_count: int


def locate_dot() -> str | None:
    """Find Graphviz dot, including common Windows install locations."""
    found = shutil.which("dot")
    if found:
        return found

    if os.name == "nt":
        candidates = [
            Path(os.environ.get("ProgramFiles", r"C:\Program Files"))
            / "Graphviz" / "bin" / "dot.exe",
            Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"))
            / "Graphviz" / "bin" / "dot.exe",
        ]

        local = os.environ.get("LOCALAPPDATA")
        if local:
            candidates.extend([
                Path(local) / "Programs" / "Graphviz" / "bin" / "dot.exe",
                Path(local) / "Graphviz" / "bin" / "dot.exe",
            ])

        for candidate in candidates:
            if candidate.exists():
                os.environ["PATH"] = (
                    str(candidate.parent)
                    + os.pathsep
                    + os.environ.get("PATH", "")
                )
                return str(candidate)

    return None


def graphviz_available() -> bool:
    return locate_dot() is not None


def load_or_build_design(
    *,
    chains: int,
    chain_length: int,
    total_gates: int,
    logic_seed: int,
    cache_path: str | Path | None,
    build_progress: int = 100,
):
    """Load the same synthetic design cache, or build/save it if absent."""
    cache = Path(cache_path).expanduser() if cache_path else None
    expected_total = int(round(total_gates / chains)) * chains

    expected = (
        int(chains),
        int(chain_length),
        int(expected_total),
        int(logic_seed),
    )

    if cache is not None and cache.exists():
        design = sim.Design.load(cache, expected=expected)
        return design, True

    design = sim.Design(
        chains,
        chain_length,
        total_gates,
        logic_seed,
    )
    design.build(progress=build_progress)

    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        design.save(cache)

    return design, False


def _source_name(src: int, L: int) -> str:
    return f"Q{src}" if src < L else f"G{src-L}"


def _is_unary_gate(typ: int) -> bool:
    return typ in (sim.BUF, sim.NOT)


def _fanin_cone(block, target_cell: int, depth: int, max_gates: int):
    """Return nodes/functional edges feeding D(Q_target)."""
    gtype, f1, f2, capsrc = block
    L = int(capsrc.size)

    target_src = int(capsrc[target_cell])

    scan_nodes = set()
    gate_nodes = set()
    edges = set()

    # (source-id, remaining gate depth)
    stack = [(target_src, max(0, int(depth)))]

    while stack:
        src, remaining = stack.pop()

        if src < L:
            scan_nodes.add(src)
            continue

        g = src - L
        if g in gate_nodes:
            continue

        if len(gate_nodes) >= max_gates:
            continue

        gate_nodes.add(g)

        if remaining <= 0:
            continue

        inputs = [int(f1[g])]
        if not _is_unary_gate(int(gtype[g])):
            inputs.append(int(f2[g]))

        for inp in inputs:
            edges.add((inp, src, "functional"))
            if inp < L:
                scan_nodes.add(inp)
            else:
                stack.append((inp, remaining - 1))

    # Capture edge into target scan flop.
    edges.add((target_src, target_cell, "capture"))

    return scan_nodes, gate_nodes, edges


def _fanout_cone(block, start_cell: int, depth: int, max_gates: int):
    """Return nodes/functional edges reachable from Q_start."""
    gtype, f1, f2, capsrc = block
    L = int(capsrc.size)
    G = int(gtype.size)

    rev_gate = defaultdict(list)
    for g in range(G):
        out_src = L + g
        rev_gate[int(f1[g])].append(out_src)
        if not _is_unary_gate(int(gtype[g])):
            rev_gate[int(f2[g])].append(out_src)

    capture_targets = defaultdict(list)
    for q, src in enumerate(capsrc):
        capture_targets[int(src)].append(int(q))

    scan_nodes = {int(start_cell)}
    gate_nodes = set()
    edges = set()

    q = deque([(int(start_cell), 0)])
    visited_sources = {int(start_cell)}

    while q:
        src, d = q.popleft()

        # Any scan flops capturing this source.
        for dst_q in capture_targets.get(src, []):
            scan_nodes.add(dst_q)
            edges.add((src, dst_q, "capture"))

        if d >= depth:
            continue

        for out_src in rev_gate.get(src, []):
            g = out_src - L
            if g not in gate_nodes:
                if len(gate_nodes) >= max_gates:
                    continue
                gate_nodes.add(g)

            edges.add((src, out_src, "functional"))

            if out_src not in visited_sources:
                visited_sources.add(out_src)
                q.append((out_src, d + 1))

    return scan_nodes, gate_nodes, edges


def _scan_neighborhood(L: int, cells: list[int], radius: int):
    if not cells:
        return set(), set()

    lo = max(0, min(cells) - radius)
    hi = min(L - 1, max(cells) + radius)

    scan_nodes = set(range(lo, hi + 1))
    edges = set()

    # Shift direction: higher index -> lower index.
    for i in range(hi, lo, -1):
        edges.add((i, i - 1, "scan"))

    return scan_nodes, edges


def collect_view(
    block,
    *,
    mode: str,
    focus_cell: int,
    depth: int,
    max_gates: int,
    scan_radius: int,
    ambiguous_cells: list[int] | None = None,
):
    """
    Collect graph content for the requested view mode.
    """
    gtype, f1, f2, capsrc = block
    L = int(capsrc.size)

    ambiguous_cells = sorted(set(ambiguous_cells or []))
    focus_cell = int(focus_cell)

    for c in [focus_cell, *ambiguous_cells]:
        if c < 0 or c >= L:
            raise ValueError(f"scan cell Q{c} is outside 0..{L-1}")

    scan_nodes = set()
    gate_nodes = set()
    edges = set()

    if mode == "scan-neighborhood":
        s, e = _scan_neighborhood(
            L,
            ambiguous_cells or [focus_cell],
            scan_radius,
        )
        scan_nodes |= s
        edges |= e

    elif mode == "fanin":
        s, g, e = _fanin_cone(
            block,
            focus_cell,
            depth,
            max_gates,
        )
        scan_nodes |= s
        gate_nodes |= g
        edges |= e

        s2, e2 = _scan_neighborhood(
            L,
            [focus_cell],
            scan_radius,
        )
        scan_nodes |= s2
        edges |= e2

    elif mode == "fanout":
        s, g, e = _fanout_cone(
            block,
            focus_cell,
            depth,
            max_gates,
        )
        scan_nodes |= s
        gate_nodes |= g
        edges |= e

        s2, e2 = _scan_neighborhood(
            L,
            [focus_cell],
            scan_radius,
        )
        scan_nodes |= s2
        edges |= e2

    elif mode == "both":
        s1, g1, e1 = _fanin_cone(
            block,
            focus_cell,
            depth,
            max_gates,
        )
        s2, g2, e2 = _fanout_cone(
            block,
            focus_cell,
            depth,
            max_gates,
        )
        scan_nodes |= s1 | s2
        gate_nodes |= g1 | g2
        edges |= e1 | e2

        s3, e3 = _scan_neighborhood(
            L,
            [focus_cell],
            scan_radius,
        )
        scan_nodes |= s3
        edges |= e3

    elif mode == "ambiguity":
        if not ambiguous_cells:
            raise ValueError(
                "ambiguity mode requires one or more ambiguous cells"
            )

        # Always show their local scan positions.
        s0, e0 = _scan_neighborhood(
            L,
            ambiguous_cells,
            scan_radius,
        )
        scan_nodes |= s0
        edges |= e0

        # Compare the capture fan-in cones of every ambiguous cell.
        per_cell_budget = max(
            1,
            max_gates // max(1, len(ambiguous_cells)),
        )
        for cell in ambiguous_cells:
            s, g, e = _fanin_cone(
                block,
                cell,
                depth,
                per_cell_budget,
            )
            scan_nodes |= s
            gate_nodes |= g
            edges |= e
    else:
        raise ValueError(f"unknown circuit view mode: {mode}")

    return scan_nodes, gate_nodes, edges


def render_view(
    design,
    *,
    chain: int,
    focus_cell: int,
    mode: str,
    depth: int,
    max_gates: int,
    scan_radius: int,
    ambiguous_cells: list[int] | None,
    output_dir: str | Path,
    basename: str | None = None,
):
    """
    Render DOT + PNG + SVG using Graphviz.

    The DOT file is always written. PNG/SVG require Graphviz's `dot` executable.
    """
    try:
        from graphviz import Digraph
    except Exception as exc:
        raise RuntimeError(
            "Python package 'graphviz' is missing. "
            "Install requirements_scan_simulator_v5.txt"
        ) from exc

    if chain < 0 or chain >= design.chains:
        raise ValueError(
            f"chain {chain} is outside 0..{design.chains-1}"
        )

    block = design.block(chain)
    gtype, f1, f2, capsrc = block
    L = design.L

    scan_nodes, gate_nodes, edges = collect_view(
        block,
        mode=mode,
        focus_cell=focus_cell,
        depth=depth,
        max_gates=max_gates,
        scan_radius=scan_radius,
        ambiguous_cells=ambiguous_cells,
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    amb = sorted(set(ambiguous_cells or []))
    if basename is None:
        if mode == "ambiguity" and amb:
            suffix = "_".join(str(c) for c in amb[:6])
            if len(amb) > 6:
                suffix += "_etc"
            basename = f"chain_{chain:04d}_ambiguity_Q{suffix}"
        else:
            basename = f"chain_{chain:04d}_Q{focus_cell:04d}_{mode}"

    base = output_dir / basename

    dot = Digraph(
        name="scan_circuit",
        comment=(
            f"Chain {chain}, mode={mode}, focus=Q{focus_cell}"
        ),
    )
    dot.attr(
        rankdir="LR",
        splines="spline",
        overlap="false",
        bgcolor="white",
        pad="0.2",
        nodesep="0.28",
        ranksep="0.45",
    )
    dot.attr(
        "node",
        fontname="Arial",
        fontsize="10",
        style="filled",
        color="#404040",
    )
    dot.attr(
        "edge",
        fontname="Arial",
        fontsize="8",
        color="#404040",
        arrowsize="0.7",
    )

    ambiguous = set(amb)

    # Scan flops.
    for q in sorted(scan_nodes):
        if q == focus_cell:
            fill = "#ffb3b3"
            penwidth = "2.2"
        elif q in ambiguous:
            fill = "#ffd59a"
            penwidth = "2.0"
        else:
            fill = "#d9ecff"
            penwidth = "1.0"

        dot.node(
            f"Q{q}",
            label=f"Q{q}",
            shape="box",
            fillcolor=fill,
            penwidth=penwidth,
            tooltip=f"Scan flop Q{q}",
        )

    # Logic gates.
    for g in sorted(gate_nodes):
        typ = int(gtype[g])
        label = GATE_LABELS.get(typ, str(typ))

        if typ in (sim.XOR, sim.XNOR):
            shape = "diamond"
            fill = "#efe1ff"
        elif typ in (sim.NOT, sim.BUF):
            shape = "triangle" if typ == sim.NOT else "box"
            fill = "#fff2cc"
        else:
            shape = "ellipse"
            fill = "#eeeeee"

        dot.node(
            f"G{g}",
            label=f"{label}\nG{g}",
            shape=shape,
            fillcolor=fill,
            tooltip=f"{label} gate G{g}",
        )

    # Functional / capture / scan edges.
    for src, dst, kind in sorted(
        edges,
        key=lambda x: (x[2], x[0], x[1]),
    ):
        src_name = _source_name(int(src), L)

        # Capture/scan destinations are scan-cell IDs. Functional destinations
        # are source IDs L+gate.
        if kind in ("capture", "scan"):
            dst_name = f"Q{int(dst)}"
        else:
            dst_name = _source_name(int(dst), L)

        if kind == "scan":
            dot.edge(
                src_name,
                dst_name,
                style="dashed",
                color="#4f81bd",
                penwidth="1.3",
                label="shift",
            )
        elif kind == "capture":
            dot.edge(
                src_name,
                dst_name,
                color="#2e8b57",
                penwidth="1.7",
                label="D",
            )
        else:
            dot.edge(
                src_name,
                dst_name,
                color="#444444",
                penwidth="1.0",
            )

    # Label the scan direction only on the explicitly rendered shift segment.
    scan_edge_nodes = set()
    for src, dst, kind in edges:
        if kind == "scan":
            scan_edge_nodes.add(int(src))
            scan_edge_nodes.add(int(dst))

    if scan_edge_nodes:
        qmin = min(scan_edge_nodes)
        qmax = max(scan_edge_nodes)

        dot.node(
            "SCAN_IN_NOTE",
            label="toward Scan-In",
            shape="plaintext",
            fontcolor="#4f81bd",
        )
        dot.node(
            "SCAN_OUT_NOTE",
            label="toward Scan-Out",
            shape="plaintext",
            fontcolor="#4f81bd",
        )
        dot.edge(
            "SCAN_IN_NOTE",
            f"Q{qmax}",
            style="dashed",
            color="#4f81bd",
            arrowhead="normal",
        )
        dot.edge(
            f"Q{qmin}",
            "SCAN_OUT_NOTE",
            style="dashed",
            color="#4f81bd",
            arrowhead="normal",
        )

    # Always write DOT source ourselves.
    dot_path = base.with_suffix(".dot")
    dot_path.write_text(dot.source, encoding="utf-8")

    png_path = None
    svg_path = None

    if graphviz_available():
        png_path = base.with_suffix(".png")
        svg_path = base.with_suffix(".svg")

        png_path.write_bytes(dot.pipe(format="png"))
        svg_path.write_bytes(dot.pipe(format="svg"))

    return CircuitViewResult(
        dot_path=dot_path,
        png_path=png_path,
        svg_path=svg_path,
        node_count=len(scan_nodes) + len(gate_nodes),
        gate_count=len(gate_nodes),
        scan_count=len(scan_nodes),
    )
