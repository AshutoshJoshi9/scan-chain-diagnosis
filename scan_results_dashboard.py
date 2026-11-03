#!/usr/bin/env python3
"""
Readable experiment-results dashboard for the scan-chain diagnosis simulator.

The dashboard reads the current output folder and summarizes:
- experiment configuration
- injected/detected faults
- pattern-pool scaling
- IG and pair-cover pattern counts
- achievable diagnosis resolution
- exact localization of the injected faults
- SAT/UNSAT residual-pair analysis
- ambiguity-class structure
- recorded runtimes

Only chains listed in the CURRENT run_metadata.json are used. This prevents
stale files from older runs in the same directory from contaminating the
dashboard.
"""

from __future__ import annotations

import csv
import json
import math
import os
import statistics
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
import tkinter as tk
from tkinter import ttk, messagebox, filedialog

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure


def _safe_mean(values):
    vals = [float(v) for v in values if v is not None and not _is_nan(v)]
    return statistics.mean(vals) if vals else None


def _is_nan(x):
    try:
        return math.isnan(float(x))
    except Exception:
        return False


def _fmt_num(v, nd=2):
    if v is None or _is_nan(v):
        return "—"
    return f"{float(v):,.{nd}f}"


def _fmt_int(v):
    if v is None or _is_nan(v):
        return "—"
    return f"{int(round(float(v))):,}"


def _fmt_seconds(sec):
    if sec is None or _is_nan(sec):
        return "—"
    sec = float(sec)
    if sec < 1:
        return f"{sec*1000:.0f} ms"
    if sec < 60:
        return f"{sec:.2f} s"
    m, s = divmod(sec, 60)
    if m < 60:
        return f"{int(m)}m {s:.1f}s"
    h, m = divmod(m, 60)
    return f"{int(h)}h {int(m)}m"


def _open_path(path: Path):
    path = Path(path)
    if os.name == "nt":
        os.startfile(str(path))
    elif sys.platform == "darwin":
        subprocess.Popen(["open", str(path)])
    else:
        subprocess.Popen(["xdg-open", str(path)])


class ResultsData:
    def __init__(self, output_dir):
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.metadata = {}
        self.pool_rows = []
        self.sat_rows = []
        self.ambiguity = {}
        self.current_chains = []

        self._load()

    def _load_json(self, path):
        with Path(path).open("r", encoding="utf-8") as f:
            return json.load(f)

    def _load_csv(self, path):
        path = Path(path)
        if not path.exists():
            return []
        with path.open("r", newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))

    @staticmethod
    def _convert_row(row):
        out = {}
        for k, v in row.items():
            if v is None:
                out[k] = None
                continue
            s = str(v).strip()
            if s == "":
                out[k] = None
                continue
            try:
                if any(ch in s.lower() for ch in [".", "e"]):
                    out[k] = float(s)
                else:
                    out[k] = int(s)
            except Exception:
                out[k] = v
        return out

    def _load(self):
        meta_path = self.output_dir / "run_metadata.json"
        if not meta_path.exists():
            raise FileNotFoundError(
                f"run_metadata.json was not found in:\n{self.output_dir}"
            )

        self.metadata = self._load_json(meta_path)

        detected = self.metadata.get("detected_chains", [])
        self.current_chains = sorted(int(x) for x in detected)
        current = set(self.current_chains)

        self.pool_rows = [
            self._convert_row(r)
            for r in self._load_csv(self.output_dir / "pool_scaling_summary.csv")
        ]
        self.pool_rows = [
            r for r in self.pool_rows
            if not current or int(r.get("chain", -1)) in current
        ]

        self.sat_rows = [
            self._convert_row(r)
            for r in self._load_csv(
                self.output_dir / "targeted_pair_sat_summary.csv"
            )
        ]
        self.sat_rows = [
            r for r in self.sat_rows
            if not current or int(r.get("chain", -1)) in current
        ]

        for ch in self.current_chains:
            p = self.output_dir / f"chain_{ch:04d}_ambiguity_classes.json"
            if p.exists():
                try:
                    self.ambiguity[ch] = self._load_json(p)
                except Exception:
                    self.ambiguity[ch] = []

    @property
    def config(self):
        return self.metadata.get("config", {})

    @property
    def faults(self):
        current = set(self.current_chains)
        faults = self.metadata.get("faults", [])
        return sorted(
            [
                f for f in faults
                if not current or int(f.get("chain", -1)) in current
            ],
            key=lambda x: int(x["chain"]),
        )

    @property
    def pool_sizes(self):
        return sorted(
            {int(r["pattern_pool_size"]) for r in self.pool_rows}
        )

    @property
    def max_pool(self):
        return max(self.pool_sizes) if self.pool_sizes else None

    @property
    def max_pool_rows(self):
        if self.max_pool is None:
            return []
        return [
            r for r in self.pool_rows
            if int(r["pattern_pool_size"]) == self.max_pool
        ]

    def pool_aggregate(self):
        by_p = defaultdict(list)
        for r in self.pool_rows:
            by_p[int(r["pattern_pool_size"])].append(r)

        result = []
        for p in sorted(by_p):
            rows = by_p[p]
            result.append({
                "P": p,
                "max_classes": _safe_mean(r.get("max_classes") for r in rows),
                "unresolved": _safe_mean(
                    r.get("intrinsic_unresolved_pairs") for r in rows
                ),
                "ig_patterns": _safe_mean(r.get("ig_patterns") for r in rows),
                "pair_patterns": _safe_mean(r.get("pair_patterns") for r in rows),
                "response_seconds": _safe_mean(
                    r.get("response_seconds") for r in rows
                ),
                "ig_seconds": _safe_mean(r.get("ig_seconds") for r in rows),
                "pair_seconds": _safe_mean(r.get("pair_seconds") for r in rows),
                "true_exact_ig": sum(
                    1 for r in rows if int(r.get("true_class_size_ig", 999)) == 1
                ),
                "true_exact_pair": sum(
                    1 for r in rows if int(r.get("true_class_size_pair", 999)) == 1
                ),
                "n": len(rows),
            })
        return result

    def ambiguity_size_distribution(self):
        c = Counter()
        contiguous = 0
        total = 0
        for classes in self.ambiguity.values():
            for rec in classes:
                size = int(rec.get("size", len(rec.get("faults", []))))
                c[size] += 1
                total += 1
                if rec.get("contiguous_cells") is True:
                    contiguous += 1
        return c, contiguous, total

    def recorded_compute_seconds(self):
        total = float(self.metadata.get("design_setup_seconds", 0) or 0)
        total += float(self.metadata.get("flush_seconds", 0) or 0)

        # The DB is simulated once per chain up to max P.
        if self.max_pool is not None:
            total += sum(
                float(r.get("response_seconds", 0) or 0)
                for r in self.max_pool_rows
            )

        # Every pool size runs both selectors.
        total += sum(
            float(r.get("ig_seconds", 0) or 0)
            + float(r.get("pair_seconds", 0) or 0)
            for r in self.pool_rows
        )

        total += sum(
            float(r.get("sat_total_seconds", 0) or 0)
            + float(r.get("witness_sim_seconds", 0) or 0)
            for r in self.sat_rows
        )
        return total

    def headline_metrics(self):
        cfg = self.config
        max_rows = self.max_pool_rows

        total_flops = (
            int(cfg.get("chains", 0))
            * int(cfg.get("chain_length", 0))
        )

        avg_classes = _safe_mean(r.get("max_classes") for r in max_rows)
        avg_h = _safe_mean(r.get("hypotheses") for r in max_rows)
        resolution_pct = (
            100.0 * avg_classes / avg_h
            if avg_classes is not None and avg_h
            else None
        )

        avg_pair = _safe_mean(r.get("pair_patterns") for r in max_rows)
        avg_ig = _safe_mean(r.get("ig_patterns") for r in max_rows)

        retained_pct = (
            100.0 * avg_pair / self.max_pool
            if avg_pair is not None and self.max_pool
            else None
        )

        exact_pair = sum(
            1 for r in max_rows
            if int(r.get("true_class_size_pair", 999)) == 1
        )
        exact_total = len(max_rows)

        sat = sum(int(r.get("sat_pairs", 0) or 0) for r in self.sat_rows)
        unsat = sum(int(r.get("unsat_pairs", 0) or 0) for r in self.sat_rows)
        unknown = sum(int(r.get("unknown_pairs", 0) or 0) for r in self.sat_rows)

        return {
            "total_flops": total_flops,
            "total_gates": int(self.metadata.get("actual_total_gates", 0) or 0),
            "failed": int(cfg.get("failed_chains", len(self.faults)) or 0),
            "detected": len(self.current_chains),
            "max_pool": self.max_pool,
            "avg_classes": avg_classes,
            "avg_hypotheses": avg_h,
            "resolution_pct": resolution_pct,
            "avg_pair_patterns": avg_pair,
            "avg_ig_patterns": avg_ig,
            "pair_retained_pct": retained_pct,
            "exact_pair": exact_pair,
            "exact_total": exact_total,
            "sat": sat,
            "unsat": unsat,
            "unknown": unknown,
            "compute_seconds": self.recorded_compute_seconds(),
        }

    def readable_findings(self):
        m = self.headline_metrics()
        parts = []

        parts.append(
            f"{m['detected']}/{m['failed']} injected failing chains were detected "
            f"by the flush screen."
        )

        if m["max_pool"] is not None and m["resolution_pct"] is not None:
            parts.append(
                f"With the full {m['max_pool']}-pattern pool, the average maximum "
                f"diagnostic resolution was {m['avg_classes']:.1f}/"
                f"{m['avg_hypotheses']:.0f} classes "
                f"({m['resolution_pct']:.2f}%)."
            )

        if m["avg_pair_patterns"] is not None:
            removed = (
                100.0 - m["pair_retained_pct"]
                if m["pair_retained_pct"] is not None
                else None
            )
            parts.append(
                f"Pair-cover preserved that pool resolution using only "
                f"{m['avg_pair_patterns']:.1f} patterns on average"
                + (
                    f" ({m['pair_retained_pct']:.2f}% of the pool; "
                    f"{removed:.2f}% discarded)."
                    if removed is not None else "."
                )
            )

        if m["exact_total"]:
            parts.append(
                f"The actual injected fault was uniquely localized in "
                f"{m['exact_pair']}/{m['exact_total']} failing chains at the "
                f"largest pool."
            )

        sat_total = m["sat"] + m["unsat"] + m["unknown"]
        if sat_total:
            parts.append(
                f"Exact targeted SAT checked {sat_total} residual fault pairs: "
                f"{m['sat']} SAT, {m['unsat']} UNSAT, and "
                f"{m['unknown']} UNKNOWN."
            )
            if m["sat"] == 0 and m["unknown"] == 0 and m["unsat"] > 0:
                parts.append(
                    "Therefore the remaining ambiguity is structural under the "
                    "current synthetic load/capture/unload model, rather than a "
                    "failure to search enough random patterns."
                )

        dist, contiguous, total = self.ambiguity_size_distribution()
        if total:
            parts.append(
                f"The run contains {total} residual ambiguity classes; "
                f"{contiguous}/{total} are contiguous in scan position."
            )

        parts.append(
            "Model caveat: the current Phase-1 architecture diagnoses failing "
            "chains as independent local logic blocks; it does not yet model "
            "cross-chain shared logic, decompression, or compactor aliasing."
        )
        return "\n\n".join(parts)


class DashboardWindow(tk.Toplevel):
    def __init__(self, parent, output_dir):
        super().__init__(parent)
        self.parent = parent
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.data = None
        self.figure_canvases = []

        self.title("Experiment Results Dashboard")
        self.geometry("1320x860")
        self.minsize(1100, 720)

        self._build_shell()
        self.reload()

    def _build_shell(self):
        root = ttk.Frame(self, padding=10)
        root.pack(fill="both", expand=True)

        top = ttk.Frame(root)
        top.pack(fill="x", pady=(0, 8))

        ttk.Label(
            top,
            text="Current Experiment Results",
            font=("Segoe UI", 18, "bold"),
        ).pack(side="left")

        self.folder_var = tk.StringVar(value=str(self.output_dir))
        ttk.Label(
            top,
            textvariable=self.folder_var,
            foreground="#555555",
        ).pack(side="left", padx=(14, 0))

        ttk.Button(
            top, text="Reload", command=self.reload
        ).pack(side="right")
        ttk.Button(
            top, text="Open Results Folder", command=self._open_folder
        ).pack(side="right", padx=6)
        ttk.Button(
            top, text="Export Readable Summary", command=self._export_summary
        ).pack(side="right", padx=6)

        self.nb = ttk.Notebook(root)
        self.nb.pack(fill="both", expand=True)

        self.tab_overview = ttk.Frame(self.nb, padding=10)
        self.tab_pool = ttk.Frame(self.nb, padding=10)
        self.tab_chain = ttk.Frame(self.nb, padding=10)
        self.tab_sat = ttk.Frame(self.nb, padding=10)
        self.tab_runtime = ttk.Frame(self.nb, padding=10)
        self.tab_setup = ttk.Frame(self.nb, padding=10)

        self.nb.add(self.tab_overview, text="Overview")
        self.nb.add(self.tab_pool, text="Pattern Scaling")
        self.nb.add(self.tab_chain, text="Per-Chain Results")
        self.nb.add(self.tab_sat, text="SAT & Ambiguity")
        self.nb.add(self.tab_runtime, text="Runtime")
        self.nb.add(self.tab_setup, text="Experiment Setup")

    def _clear_tab(self, tab):
        for child in tab.winfo_children():
            child.destroy()

    def reload(self):
        try:
            self.data = ResultsData(self.output_dir)
        except Exception as exc:
            messagebox.showerror("Results Dashboard", str(exc), parent=self)
            return

        for tab in [
            self.tab_overview, self.tab_pool, self.tab_chain,
            self.tab_sat, self.tab_runtime, self.tab_setup,
        ]:
            self._clear_tab(tab)

        self._build_overview()
        self._build_pool()
        self._build_per_chain()
        self._build_sat()
        self._build_runtime()
        self._build_setup()

    # ------------------------------------------------------------------
    # General widgets
    # ------------------------------------------------------------------
    def _metric_card(self, parent, title, value, subtitle=""):
        f = ttk.LabelFrame(parent, text=title, padding=8)
        ttk.Label(
            f,
            text=value,
            font=("Segoe UI", 16, "bold"),
        ).pack(anchor="center", pady=(2, 1))
        if subtitle:
            ttk.Label(
                f,
                text=subtitle,
                foreground="#555555",
                justify="center",
            ).pack(anchor="center")
        return f

    def _tree(self, parent, columns, headings=None, height=12):
        frame = ttk.Frame(parent)
        tree = ttk.Treeview(
            frame,
            columns=columns,
            show="headings",
            height=height,
        )
        ys = ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
        xs = ttk.Scrollbar(frame, orient="horizontal", command=tree.xview)
        tree.configure(yscrollcommand=ys.set, xscrollcommand=xs.set)

        tree.grid(row=0, column=0, sticky="nsew")
        ys.grid(row=0, column=1, sticky="ns")
        xs.grid(row=1, column=0, sticky="ew")
        frame.rowconfigure(0, weight=1)
        frame.columnconfigure(0, weight=1)

        headings = headings or {}
        for c in columns:
            tree.heading(c, text=headings.get(c, c))
            tree.column(c, width=110, anchor="center")

        return frame, tree

    def _embed_figure(self, parent, fig):
        canvas = FigureCanvasTkAgg(fig, master=parent)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)
        self.figure_canvases.append(canvas)
        return canvas

    # ------------------------------------------------------------------
    # Overview
    # ------------------------------------------------------------------
    def _build_overview(self):
        d = self.data
        m = d.headline_metrics()

        cards = ttk.Frame(self.tab_overview)
        cards.pack(fill="x", pady=(0, 10))

        card_specs = [
            (
                "Design size",
                f"{m['total_flops']:,}",
                f"scan flops • {m['total_gates']:,} gates",
            ),
            (
                "Failing chains",
                f"{m['detected']}/{m['failed']}",
                "detected / injected",
            ),
            (
                "Largest pattern pool",
                _fmt_int(m["max_pool"]),
                "candidate patterns",
            ),
            (
                "Average resolution",
                (
                    f"{m['resolution_pct']:.2f}%"
                    if m["resolution_pct"] is not None else "—"
                ),
                (
                    f"{m['avg_classes']:.1f}/{m['avg_hypotheses']:.0f} classes"
                    if m["avg_classes"] is not None else ""
                ),
            ),
            (
                "Pair-cover patterns",
                (
                    f"{m['avg_pair_patterns']:.1f}"
                    if m["avg_pair_patterns"] is not None else "—"
                ),
                (
                    f"{m['pair_retained_pct']:.2f}% of full pool"
                    if m["pair_retained_pct"] is not None else ""
                ),
            ),
            (
                "True faults exact",
                f"{m['exact_pair']}/{m['exact_total']}",
                "singleton class at largest pool",
            ),
        ]

        for i, spec in enumerate(card_specs):
            card = self._metric_card(cards, *spec)
            card.grid(
                row=i // 3,
                column=i % 3,
                sticky="nsew",
                padx=4,
                pady=4,
            )

        for c in range(3):
            cards.columnconfigure(c, weight=1)

        lower = ttk.Panedwindow(self.tab_overview, orient="horizontal")
        lower.pack(fill="both", expand=True)

        left = ttk.LabelFrame(lower, text="What this run says", padding=10)
        right = ttk.LabelFrame(lower, text="Pipeline represented by this experiment", padding=10)
        lower.add(left, weight=3)
        lower.add(right, weight=2)

        txt = tk.Text(left, wrap="word", font=("Segoe UI", 10))
        txt.pack(fill="both", expand=True)
        txt.insert("1.0", d.readable_findings())
        txt.configure(state="disabled")

        self._draw_pipeline(right)

    def _draw_pipeline(self, parent):
        canvas = tk.Canvas(parent, background="white", height=500)
        canvas.pack(fill="both", expand=True)

        labels = [
            ("Synthetic design", "scan chains + local logic"),
            ("Alternating flush", "identify failing chains"),
            ("Local hypotheses", "candidate cell positions"),
            ("Response database", "candidate patterns × faults"),
            ("IG / Pair cover", "small diagnostic set"),
            ("Exact SAT", "test residual pairs"),
            ("Final diagnosis", "resolution + ambiguity"),
        ]

        def draw():
            canvas.delete("all")
            w = max(700, canvas.winfo_width())
            h = max(450, canvas.winfo_height())
            x = w / 2
            top = 35
            box_w = min(430, w - 100)
            box_h = 52
            gap = 17

            y = top
            prev_bottom = None
            for title, sub in labels:
                x0 = x - box_w / 2
                x1 = x + box_w / 2
                y0 = y
                y1 = y + box_h

                canvas.create_rectangle(
                    x0, y0, x1, y1,
                    outline="#555555",
                    width=1.5,
                    fill="#f5f5f5",
                )
                canvas.create_text(
                    x, y0 + 18,
                    text=title,
                    font=("Segoe UI", 11, "bold"),
                )
                canvas.create_text(
                    x, y0 + 37,
                    text=sub,
                    font=("Segoe UI", 9),
                    fill="#555555",
                )

                if prev_bottom is not None:
                    canvas.create_line(
                        x, prev_bottom,
                        x, y0,
                        arrow=tk.LAST,
                        width=1.5,
                    )

                prev_bottom = y1
                y += box_h + gap

        canvas.bind("<Configure>", lambda e: draw())
        self.after(100, draw)

    # ------------------------------------------------------------------
    # Pattern scaling
    # ------------------------------------------------------------------
    def _build_pool(self):
        d = self.data
        agg = d.pool_aggregate()

        if not agg:
            ttk.Label(
                self.tab_pool,
                text="No pool_scaling_summary.csv was generated for this run.",
                font=("Segoe UI", 12),
            ).pack(pady=30)
            return

        charts = ttk.Panedwindow(self.tab_pool, orient="horizontal")
        charts.pack(fill="both", expand=True, pady=(0, 8))

        f1 = ttk.LabelFrame(charts, text="Achievable resolution vs candidate pool", padding=4)
        f2 = ttk.LabelFrame(charts, text="Selected diagnostic patterns", padding=4)
        charts.add(f1, weight=1)
        charts.add(f2, weight=1)

        fig1 = Figure(figsize=(5.5, 3.5), dpi=100)
        ax1 = fig1.add_subplot(111)
        x = [r["P"] for r in agg]
        y = [r["max_classes"] for r in agg]
        ax1.plot(x, y, marker="o")
        ax1.set_xlabel("Candidate pool size")
        ax1.set_ylabel("Average distinguishable classes")
        ax1.set_xscale("log", base=2)
        ax1.grid(True, alpha=0.25)
        fig1.tight_layout()
        self._embed_figure(f1, fig1)

        fig2 = Figure(figsize=(5.5, 3.5), dpi=100)
        ax2 = fig2.add_subplot(111)
        ax2.plot(x, [r["ig_patterns"] for r in agg], marker="o", label="Information gain")
        ax2.plot(x, [r["pair_patterns"] for r in agg], marker="o", label="Pair cover")
        ax2.set_xlabel("Candidate pool size")
        ax2.set_ylabel("Average selected patterns")
        ax2.set_xscale("log", base=2)
        ax2.grid(True, alpha=0.25)
        ax2.legend()
        fig2.tight_layout()
        self._embed_figure(f2, fig2)

        columns = (
            "P", "classes", "unresolved", "ig", "pair",
            "response", "igtime", "pairtime", "exact",
        )
        headings = {
            "P": "Pool",
            "classes": "Avg classes",
            "unresolved": "Avg unresolved pairs",
            "ig": "Avg IG patterns",
            "pair": "Avg pair patterns",
            "response": "Avg response time",
            "igtime": "Avg IG time",
            "pairtime": "Avg pair time",
            "exact": "Exact true faults",
        }
        frame, tree = self._tree(
            self.tab_pool, columns, headings, height=7
        )
        frame.pack(fill="x")

        for r in agg:
            tree.insert(
                "", "end",
                values=(
                    r["P"],
                    _fmt_num(r["max_classes"], 1),
                    _fmt_num(r["unresolved"], 1),
                    _fmt_num(r["ig_patterns"], 1),
                    _fmt_num(r["pair_patterns"], 1),
                    _fmt_seconds(r["response_seconds"]),
                    _fmt_seconds(r["ig_seconds"]),
                    _fmt_seconds(r["pair_seconds"]),
                    f"{r['true_exact_pair']}/{r['n']}",
                ),
            )

    # ------------------------------------------------------------------
    # Per-chain
    # ------------------------------------------------------------------
    def _build_per_chain(self):
        d = self.data
        rows = sorted(d.max_pool_rows, key=lambda r: int(r["chain"]))

        if not rows:
            ttk.Label(
                self.tab_chain,
                text="No per-chain pool results were generated.",
            ).pack(pady=30)
            return

        top = ttk.Panedwindow(self.tab_chain, orient="horizontal")
        top.pack(fill="both", expand=True, pady=(0, 8))

        chart = ttk.LabelFrame(top, text="Final full-pool resolution by failing chain", padding=4)
        tablebox = ttk.LabelFrame(top, text="Current failing chains", padding=4)
        top.add(chart, weight=2)
        top.add(tablebox, weight=3)

        fig = Figure(figsize=(6, 4), dpi=100)
        ax = fig.add_subplot(111)
        chains = [str(int(r["chain"])) for r in rows]
        vals = [
            100.0 * float(r["max_classes"]) / float(r["hypotheses"])
            for r in rows
        ]
        ax.bar(chains, vals)
        ax.set_xlabel("Failing chain")
        ax.set_ylabel("Maximum resolution (%)")
        ax.set_ylim(
            max(0, min(vals) - 2),
            min(100, max(vals) + 1),
        )
        ax.grid(True, axis="y", alpha=0.25)
        fig.tight_layout()
        self._embed_figure(chart, fig)

        fault_map = {
            int(f["chain"]): f for f in d.faults
        }
        sat_map = {
            int(r["chain"]): r for r in d.sat_rows
        }

        columns = (
            "chain", "fault", "classes", "resolution",
            "ig", "pair", "unresolved", "exact", "final",
        )
        headings = {
            "chain": "Chain",
            "fault": "Injected fault",
            "classes": "Full-pool classes",
            "resolution": "Resolution",
            "ig": "IG patterns",
            "pair": "Pair patterns",
            "unresolved": "Unresolved pairs",
            "exact": "True fault exact?",
            "final": "After SAT classes",
        }
        frame, tree = self._tree(
            tablebox, columns, headings, height=18
        )
        frame.pack(fill="both", expand=True)

        for r in rows:
            ch = int(r["chain"])
            f = fault_map.get(ch, {})
            fault = (
                f"Q{f.get('cell')}:SA{f.get('stuck')}"
                if f else "—"
            )
            sat = sat_map.get(ch)
            resolution = (
                100.0 * float(r["max_classes"]) / float(r["hypotheses"])
            )
            tree.insert(
                "", "end",
                values=(
                    ch,
                    fault,
                    f"{int(r['max_classes'])}/{int(r['hypotheses'])}",
                    f"{resolution:.2f}%",
                    int(r["ig_patterns"]),
                    int(r["pair_patterns"]),
                    int(r["intrinsic_unresolved_pairs"]),
                    "Yes" if int(r["true_class_size_pair"]) == 1 else "No",
                    (
                        int(sat["final_classes"])
                        if sat is not None else "—"
                    ),
                ),
            )

    # ------------------------------------------------------------------
    # SAT and ambiguity
    # ------------------------------------------------------------------
    def _build_sat(self):
        d = self.data
        rows = sorted(d.sat_rows, key=lambda r: int(r["chain"]))

        if not rows:
            ttk.Label(
                self.tab_sat,
                text=(
                    "No targeted_pair_sat_summary.csv was generated. "
                    "Run targeted-sat or the full pipeline to populate this page."
                ),
                font=("Segoe UI", 11),
            ).pack(pady=30)
            return

        upper = ttk.Panedwindow(self.tab_sat, orient="horizontal")
        upper.pack(fill="both", expand=True, pady=(0, 8))

        satbox = ttk.LabelFrame(upper, text="Exact residual-pair SAT outcome", padding=4)
        ambbox = ttk.LabelFrame(upper, text="Ambiguity-class size distribution", padding=4)
        upper.add(satbox, weight=1)
        upper.add(ambbox, weight=1)

        sat_totals = {
            "SAT": sum(int(r.get("sat_pairs", 0)) for r in rows),
            "UNSAT": sum(int(r.get("unsat_pairs", 0)) for r in rows),
            "UNKNOWN": sum(int(r.get("unknown_pairs", 0)) for r in rows),
        }

        fig1 = Figure(figsize=(5, 3.5), dpi=100)
        ax1 = fig1.add_subplot(111)
        ax1.bar(list(sat_totals.keys()), list(sat_totals.values()))
        ax1.set_ylabel("Residual fault pairs")
        ax1.grid(True, axis="y", alpha=0.25)
        for i, v in enumerate(sat_totals.values()):
            ax1.text(i, v, str(v), ha="center", va="bottom")
        fig1.tight_layout()
        self._embed_figure(satbox, fig1)

        dist, contiguous, total = d.ambiguity_size_distribution()
        fig2 = Figure(figsize=(5, 3.5), dpi=100)
        ax2 = fig2.add_subplot(111)
        sizes = sorted(dist)
        counts = [dist[s] for s in sizes]
        ax2.bar([str(s) for s in sizes], counts)
        ax2.set_xlabel("Cells / hypotheses in ambiguity class")
        ax2.set_ylabel("Number of classes")
        ax2.grid(True, axis="y", alpha=0.25)
        fig2.tight_layout()
        self._embed_figure(ambbox, fig2)

        ttk.Label(
            self.tab_sat,
            text=(
                f"Ambiguity classes: {total:,} total • "
                f"{contiguous:,}/{total:,} contiguous in scan position"
                if total else "No ambiguity-class JSON files found for current chains."
            ),
            font=("Segoe UI", 10, "bold"),
        ).pack(anchor="w", pady=(0, 6))

        columns = (
            "chain", "base", "ambclasses", "pairs",
            "sat", "unsat", "unknown", "final", "time",
        )
        headings = {
            "chain": "Chain",
            "base": "Base classes",
            "ambclasses": "Ambiguous classes",
            "pairs": "Pair queries",
            "sat": "SAT",
            "unsat": "UNSAT",
            "unknown": "UNKNOWN",
            "final": "Final classes",
            "time": "SAT time",
        }
        frame, tree = self._tree(
            self.tab_sat, columns, headings, height=8
        )
        frame.pack(fill="x")

        for r in rows:
            tree.insert(
                "", "end",
                values=(
                    int(r["chain"]),
                    f"{int(r['base_classes'])}/{int(r['hypotheses'])}",
                    int(r["ambiguous_classes"]),
                    int(r["pair_queries"]),
                    int(r["sat_pairs"]),
                    int(r["unsat_pairs"]),
                    int(r["unknown_pairs"]),
                    f"{int(r['final_classes'])}/{int(r['hypotheses'])}",
                    _fmt_seconds(r["sat_total_seconds"]),
                ),
            )

    # ------------------------------------------------------------------
    # Runtime
    # ------------------------------------------------------------------
    def _build_runtime(self):
        d = self.data
        agg = d.pool_aggregate()

        if not agg:
            ttk.Label(
                self.tab_runtime,
                text="No pool timing data available.",
            ).pack(pady=30)
            return

        top = ttk.Panedwindow(self.tab_runtime, orient="horizontal")
        top.pack(fill="both", expand=True)

        poolbox = ttk.LabelFrame(top, text="Average runtime vs candidate pool", padding=4)
        satbox = ttk.LabelFrame(top, text="Exact SAT runtime by chain", padding=4)
        top.add(poolbox, weight=1)
        top.add(satbox, weight=1)

        x = [r["P"] for r in agg]
        fig1 = Figure(figsize=(5.5, 3.8), dpi=100)
        ax1 = fig1.add_subplot(111)
        ax1.plot(x, [r["response_seconds"] for r in agg], marker="o", label="Response DB")
        ax1.plot(x, [r["ig_seconds"] for r in agg], marker="o", label="IG selector")
        ax1.plot(x, [r["pair_seconds"] for r in agg], marker="o", label="Pair selector")
        ax1.set_xscale("log", base=2)
        ax1.set_xlabel("Candidate pool size")
        ax1.set_ylabel("Average seconds / failing chain")
        ax1.grid(True, alpha=0.25)
        ax1.legend()
        fig1.tight_layout()
        self._embed_figure(poolbox, fig1)

        fig2 = Figure(figsize=(5.5, 3.8), dpi=100)
        ax2 = fig2.add_subplot(111)
        satrows = sorted(d.sat_rows, key=lambda r: int(r["chain"]))
        if satrows:
            chains = [str(int(r["chain"])) for r in satrows]
            times = [float(r["sat_total_seconds"]) for r in satrows]
            ax2.bar(chains, times)
            ax2.set_xlabel("Failing chain")
            ax2.set_ylabel("SAT seconds")
            ax2.grid(True, axis="y", alpha=0.25)
        else:
            ax2.text(
                0.5, 0.5,
                "No SAT stage in this run",
                ha="center", va="center",
                transform=ax2.transAxes,
            )
            ax2.set_axis_off()
        fig2.tight_layout()
        self._embed_figure(satbox, fig2)

        ttk.Label(
            self.tab_runtime,
            text=(
                "Recorded compute time across the timed stages: "
                f"{_fmt_seconds(d.recorded_compute_seconds())}. "
                "This is a sum of recorded stage timings, not wall-clock time."
            ),
            font=("Segoe UI", 10, "bold"),
        ).pack(anchor="w", pady=(8, 0))

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------
    def _build_setup(self):
        d = self.data

        pan = ttk.Panedwindow(self.tab_setup, orient="horizontal")
        pan.pack(fill="both", expand=True)

        configbox = ttk.LabelFrame(pan, text="Run configuration", padding=6)
        faultbox = ttk.LabelFrame(pan, text="Injected / detected faults", padding=6)
        pan.add(configbox, weight=2)
        pan.add(faultbox, weight=1)

        text = tk.Text(configbox, wrap="none", font=("Consolas", 10))
        ys = ttk.Scrollbar(configbox, orient="vertical", command=text.yview)
        xs = ttk.Scrollbar(configbox, orient="horizontal", command=text.xview)
        text.configure(yscrollcommand=ys.set, xscrollcommand=xs.set)

        text.grid(row=0, column=0, sticky="nsew")
        ys.grid(row=0, column=1, sticky="ns")
        xs.grid(row=1, column=0, sticky="ew")
        configbox.rowconfigure(0, weight=1)
        configbox.columnconfigure(0, weight=1)

        payload = {
            "config": d.metadata.get("config", {}),
            "machine": {
                "python": d.metadata.get("python"),
                "platform": d.metadata.get("platform"),
                "cpu_count": d.metadata.get("cpu_count"),
                "numba_threads": d.metadata.get("numba_threads"),
            },
            "design": {
                "actual_total_gates": d.metadata.get("actual_total_gates"),
                "gates_per_chain": d.metadata.get("gates_per_chain"),
                "design_memory_mib": d.metadata.get("design_memory_mib"),
                "design_setup_seconds": d.metadata.get("design_setup_seconds"),
                "design_cache_used": d.metadata.get("design_cache_used"),
                "flush_seconds": d.metadata.get("flush_seconds"),
            },
        }
        text.insert("1.0", json.dumps(payload, indent=2))
        text.configure(state="disabled")

        columns = ("chain", "cell", "stuck", "detected")
        headings = {
            "chain": "Chain",
            "cell": "Cell",
            "stuck": "Fault",
            "detected": "Detected?",
        }
        frame, tree = self._tree(
            faultbox, columns, headings, height=20
        )
        frame.pack(fill="both", expand=True)

        detected = set(d.current_chains)
        for f in d.faults:
            ch = int(f["chain"])
            tree.insert(
                "", "end",
                values=(
                    ch,
                    f"Q{int(f['cell'])}",
                    f"SA{int(f['stuck'])}",
                    "Yes" if ch in detected else "No",
                ),
            )

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------
    def _open_folder(self):
        try:
            _open_path(self.output_dir)
        except Exception as exc:
            messagebox.showerror(
                "Open results folder", str(exc), parent=self
            )

    def _export_summary(self):
        d = self.data
        default = self.output_dir / "experiment_readable_summary.txt"
        fn = filedialog.asksaveasfilename(
            parent=self,
            title="Export readable summary",
            initialdir=str(self.output_dir),
            initialfile=default.name,
            defaultextension=".txt",
            filetypes=[("Text file", "*.txt"), ("All files", "*.*")],
        )
        if not fn:
            return

        m = d.headline_metrics()
        lines = [
            "SCAN-CHAIN DIAGNOSIS EXPERIMENT SUMMARY",
            "=" * 42,
            "",
            d.readable_findings(),
            "",
            "HEADLINE METRICS",
            "-" * 20,
            f"Total scan flops: {m['total_flops']:,}",
            f"Total gates: {m['total_gates']:,}",
            f"Detected / injected failing chains: {m['detected']}/{m['failed']}",
            f"Largest candidate pool: {_fmt_int(m['max_pool'])}",
            f"Average full-pool classes: {_fmt_num(m['avg_classes'], 1)}",
            f"Average resolution: {_fmt_num(m['resolution_pct'], 2)}%",
            f"Average pair-cover patterns: {_fmt_num(m['avg_pair_patterns'], 1)}",
            f"Exact true faults: {m['exact_pair']}/{m['exact_total']}",
            f"SAT / UNSAT / UNKNOWN: {m['sat']} / {m['unsat']} / {m['unknown']}",
            f"Recorded compute time: {_fmt_seconds(m['compute_seconds'])}",
            "",
            f"Source results folder: {self.output_dir}",
        ]

        Path(fn).write_text("\n".join(lines), encoding="utf-8")
        messagebox.showinfo(
            "Results Dashboard",
            f"Summary exported:\n{fn}",
            parent=self,
        )


def open_dashboard(parent, output_dir):
    return DashboardWindow(parent, output_dir)
