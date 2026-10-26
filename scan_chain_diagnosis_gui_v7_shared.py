#!/usr/bin/env python3
"""
Scan-Chain Diagnosis Research Simulator - GUI v2

A practical desktop front-end for scan_chain_diagnosis_local.py.

Key changes from v1:
- RUN / STOP / OPEN RESULTS buttons are always visible in a fixed bottom bar.
- Parameters are split into Basic and Advanced tabs so nothing important is
  hidden below the window.
- Dedicated Results tab lists generated CSV/JSON/NPZ files and previews CSV/JSON.
- One-click presets reproduce the recent experiments.
- The simulator writes all result files automatically into the chosen output folder.
"""

from __future__ import annotations

import csv
import json
import os
import queue
import shlex
import subprocess
import sys
import threading
import time
from pathlib import Path
import shutil

from PIL import Image, ImageTk

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

SIMULATOR_NAME = "scan_chain_diagnosis_local.py"
SHARED_SIMULATOR_NAME = "scan_chain_diagnosis_shared.py"


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Scan-Chain Diagnosis Research Simulator + Shared Logic + Results Dashboard")
        self.geometry("1240x820")
        self.minsize(1050, 700)

        self.proc = None
        self.reader = None
        self.q = queue.Queue()
        self.start_time = None

        # Circuit-viewer state.
        self.viewer_thread = None
        self.viewer_queue = queue.Queue()
        self.viewer_design = None
        self.viewer_design_key = None
        self.viewer_png_path = None
        self.viewer_original_image = None
        self.viewer_photo = None
        self.viewer_zoom = 1.0
        self.ambiguity_records = []

        self.vars = {}
        self._style()
        self._vars()
        self._ui()
        self._preset_full()

        self.after(100, self._poll)
        self.after(120, self._poll_viewer)
        self.after(500, self._refresh_results_silent)

    # ------------------------------------------------------------------ #
    # setup
    # ------------------------------------------------------------------ #
    def _style(self):
        s = ttk.Style(self)
        try:
            s.theme_use("vista")
        except tk.TclError:
            pass
        s.configure("Title.TLabel", font=("Segoe UI", 17, "bold"))
        s.configure("H.TLabel", font=("Segoe UI", 10, "bold"))
        s.configure("Run.TButton", font=("Segoe UI", 11, "bold"), padding=(16, 8))
        s.configure("Big.TButton", font=("Segoe UI", 10, "bold"), padding=(10, 6))

    def _vars(self):
        defaults = {
            "mode": "all",
            "chains": "1000",
            "chain_length": "1000",
            "total_gates": "10000000",
            "failed_chains": "5",
            "pattern_counts": "64 128 256 512 1024",
            "max_selected": "50",
            "sat_conf_budget": "500000",
            "only_chain": "",
            "threads": "0",
            "design_cache": "scan_design_1m_10m_seed12345.npz",
            "outdir": "local_full_run",
            "seed": "20261008",
            "logic_seed": "12345",
            "pattern_seed": "20260929",
            "hash_seed": "918273",
            "design_progress": "100",
            "response_progress": "64",
            "pair_progress": "10",
            "shared_logic_fraction": "0.01",
            "shared_partners": "4",
            "shared_seed": "314159",
        }
        for k, v in defaults.items():
            self.vars[k] = tk.StringVar(value=v)
        self.vars["infer_stuck"] = tk.BooleanVar(value=True)
        self.vars["single_flush"] = tk.BooleanVar(value=False)
        self.vars["shared_logic"] = tk.BooleanVar(value=False)

        self.vars["view_chain"] = tk.StringVar(value="210")
        self.vars["view_cell"] = tk.StringVar(value="380")
        self.vars["view_mode"] = tk.StringVar(value="both")
        self.vars["view_depth"] = tk.StringVar(value="5")
        self.vars["view_max_gates"] = tk.StringVar(value="150")
        self.vars["view_scan_radius"] = tk.StringVar(value="6")
        self.vars["view_ambiguous_cells"] = tk.StringVar(value="")
        self.vars["view_ambiguity_choice"] = tk.StringVar(value="")

    def _ui(self):
        root = ttk.Frame(self, padding=10)
        root.pack(fill="both", expand=True)

        # Header
        hdr = ttk.Frame(root)
        hdr.pack(fill="x", pady=(0, 8))
        ttk.Label(
            hdr, text="Scan-Chain Diagnosis Research Simulator + Results Dashboard",
            style="Title.TLabel"
        ).pack(side="left")
        self.status = tk.StringVar(value="Idle")
        ttk.Label(hdr, textvariable=self.status).pack(side="right")

        # Presets -- always visible
        presets = ttk.LabelFrame(root, text="Experiment presets", padding=8)
        presets.pack(fill="x", pady=(0, 8))
        ttk.Button(presets, text="Quick Test", command=self._preset_quick).pack(side="left", padx=(0,6))
        ttk.Button(presets, text="Full Latest Pipeline", command=self._preset_full).pack(side="left", padx=6)
        ttk.Button(presets, text="Pool Scaling Only", command=self._preset_pool).pack(side="left", padx=6)
        ttk.Button(presets, text="Targeted SAT: Chain 210", command=self._preset_sat).pack(side="left", padx=6)
        ttk.Button(presets, text="Shared Quick Test", command=self._preset_shared_quick).pack(side="left", padx=6)
        ttk.Button(presets, text="Shared 1% Full", command=self._preset_shared_full).pack(side="left", padx=6)

        # Main notebook
        self.nb = ttk.Notebook(root)
        self.nb.pack(fill="both", expand=True)

        self.tab_exp = ttk.Frame(self.nb, padding=10)
        self.tab_adv = ttk.Frame(self.nb, padding=10)
        self.tab_log = ttk.Frame(self.nb, padding=8)
        self.tab_results = ttk.Frame(self.nb, padding=8)
        self.tab_circuit = ttk.Frame(self.nb, padding=8)

        self.nb.add(self.tab_exp, text="Experiment")
        self.nb.add(self.tab_adv, text="Advanced")
        self.nb.add(self.tab_log, text="Live Log")
        self.nb.add(self.tab_results, text="Results")
        self.nb.add(self.tab_circuit, text="Circuit Viewer")

        self._experiment_tab()
        self._advanced_tab()
        self._log_tab()
        self._results_tab()
        self._circuit_tab()

        # Fixed action bar -- ALWAYS visible.
        bar = ttk.Frame(root)
        bar.pack(fill="x", pady=(9, 0))

        self.run_btn = ttk.Button(
            bar, text="▶ RUN & GENERATE RESULTS",
            style="Run.TButton", command=self._run
        )
        self.run_btn.pack(side="left")

        self.stop_btn = ttk.Button(
            bar, text="■ STOP", style="Big.TButton",
            command=self._stop, state="disabled"
        )
        self.stop_btn.pack(side="left", padx=8)

        ttk.Button(
            bar, text="Open Results Folder",
            style="Big.TButton", command=self._open_results
        ).pack(side="left", padx=8)

        ttk.Button(
            bar, text="Refresh Results",
            style="Big.TButton", command=self._refresh_results
        ).pack(side="left", padx=8)

        ttk.Button(
            bar,
            text="Experiment Dashboard",
            style="Big.TButton",
            command=self._open_dashboard,
        ).pack(side="left", padx=8)

        self.progress = ttk.Progressbar(bar, mode="indeterminate", length=220)
        self.progress.pack(side="right", padx=(8,0))
        self.elapsed = tk.StringVar(value="Elapsed 00:00:00")
        ttk.Label(bar, textvariable=self.elapsed).pack(side="right")

    # ------------------------------------------------------------------ #
    # Experiment tab
    # ------------------------------------------------------------------ #
    def _experiment_tab(self):
        t = self.tab_exp

        left = ttk.LabelFrame(t, text="Circuit / fault setup", padding=10)
        right = ttk.LabelFrame(t, text="Diagnosis experiment", padding=10)
        left.pack(side="left", fill="both", expand=True, padx=(0,6))
        right.pack(side="left", fill="both", expand=True, padx=(6,0))

        self._combo(left, 0, "Experiment mode", "mode",
                    ["all", "pool-scale", "targeted-sat"])
        self._entry(left, 1, "Scan chains", "chains")
        self._entry(left, 2, "Flops per chain", "chain_length")
        self._entry(left, 3, "Total gates", "total_gates")
        self._entry(left, 4, "Random failed chains", "failed_chains")

        ttk.Separator(left).grid(row=5, column=0, columnspan=3, sticky="ew", pady=10)

        ttk.Checkbutton(
            left, text="Infer SA0/SA1 polarity from alternating flush",
            variable=self.vars["infer_stuck"]
        ).grid(row=6, column=0, columnspan=3, sticky="w", pady=5)
        ttk.Checkbutton(
            left, text="Use only one 1010... flush pattern",
            variable=self.vars["single_flush"]
        ).grid(row=7, column=0, columnspan=3, sticky="w", pady=5)

        shared = ttk.LabelFrame(left, text="Shared functional logic", padding=8)
        shared.grid(
            row=8,
            column=0,
            columnspan=3,
            sticky="ew",
            pady=(12, 0),
        )

        ttk.Checkbutton(
            shared,
            text="Enable cross-chain functional coupling",
            variable=self.vars["shared_logic"],
            command=self._shared_logic_toggled,
        ).grid(
            row=0,
            column=0,
            columnspan=3,
            sticky="w",
            pady=(0, 5),
        )

        self._entry(
            shared,
            1,
            "Shared logic fraction",
            "shared_logic_fraction",
            "0..1, e.g. 0.01",
        )
        self._entry(
            shared,
            2,
            "Shared partners",
            "shared_partners",
            "source chains/block",
        )

        ttk.Label(
            shared,
            text=(
                "Shared mode uses scan_chain_diagnosis_shared.py. "
                "Use a separate design cache for each shared fraction."
            ),
            foreground="#555555",
            justify="left",
            wraplength=430,
        ).grid(
            row=3,
            column=0,
            columnspan=3,
            sticky="w",
            pady=(6, 0),
        )

        shared.columnconfigure(1, weight=1)

        self._entry(right, 0, "Pattern pools", "pattern_counts",
                    "e.g. 64 128 256 512 1024")
        self._entry(right, 1, "Max selected patterns", "max_selected")
        self._entry(right, 2, "SAT conflict budget", "sat_conf_budget")
        self._entry(right, 3, "Only chain (optional)", "only_chain",
                    "blank = all detected")

        ttk.Separator(right).grid(row=4, column=0, columnspan=3, sticky="ew", pady=10)

        ttk.Label(right, text="Output folder", style="H.TLabel").grid(
            row=5, column=0, sticky="w", pady=5
        )
        ttk.Entry(right, textvariable=self.vars["outdir"]).grid(
            row=5, column=1, sticky="ew", padx=6, pady=5
        )
        ttk.Button(right, text="Browse...", command=self._browse_outdir).grid(
            row=5, column=2, pady=5
        )

        ttk.Label(right, text="Design cache", style="H.TLabel").grid(
            row=6, column=0, sticky="w", pady=5
        )
        ttk.Entry(right, textvariable=self.vars["design_cache"]).grid(
            row=6, column=1, sticky="ew", padx=6, pady=5
        )
        ttk.Button(right, text="Browse...", command=self._browse_cache).grid(
            row=6, column=2, pady=5
        )

        # What will be generated
        generated = ttk.LabelFrame(right, text="Results generated automatically", padding=8)
        generated.grid(row=7, column=0, columnspan=3, sticky="nsew", pady=(12,0))
        txt = (
            "• pool_scaling_summary.csv\n"
            "• targeted_pair_sat_summary.csv\n"
            "• shared_logic_topology.csv (shared mode)\n"
            "• chain_XXXX_pool_YYYY.json\n"
            "• chain_XXXX_ambiguity_classes.json\n"
            "• chain_XXXX_pair_sat.csv\n"
            "• run_metadata.json\n"
            "• SAT witness .npz files when witnesses exist"
        )
        ttk.Label(generated, text=txt, justify="left").pack(anchor="w")

        left.columnconfigure(1, weight=1)
        right.columnconfigure(1, weight=1)

    # ------------------------------------------------------------------ #
    # Advanced tab
    # ------------------------------------------------------------------ #
    def _advanced_tab(self):
        t = self.tab_adv
        left = ttk.LabelFrame(t, text="Execution", padding=10)
        right = ttk.LabelFrame(t, text="Reproducibility / progress", padding=10)
        left.pack(side="left", fill="both", expand=True, padx=(0,6))
        right.pack(side="left", fill="both", expand=True, padx=(6,0))

        self._entry(left, 0, "Numba threads", "threads", "0 = default")
        ttk.Label(
            left,
            text=(
                "For a laptop, 4–8 threads is often a good starting point.\n"
                "The 10M-gate design cache avoids rebuilding the circuit on later runs."
            ),
            justify="left",
            foreground="#555555"
        ).grid(row=1, column=0, columnspan=3, sticky="w", pady=10)

        self._entry(right, 0, "Fault seed", "seed")
        self._entry(right, 1, "Logic seed", "logic_seed")
        self._entry(right, 2, "Pattern seed", "pattern_seed")
        self._entry(right, 3, "Hash seed", "hash_seed")
        self._entry(right, 4, "Shared topology seed", "shared_seed")
        ttk.Separator(right).grid(
            row=5,
            column=0,
            columnspan=3,
            sticky="ew",
            pady=10,
        )
        self._entry(right, 6, "Design progress every N", "design_progress")
        self._entry(right, 7, "Response progress every N", "response_progress")
        self._entry(right, 8, "SAT progress every N", "pair_progress")

        left.columnconfigure(1, weight=1)
        right.columnconfigure(1, weight=1)

        # command preview at bottom of Advanced, not on main screen
        cmd = ttk.LabelFrame(t, text="Command preview", padding=6)
        cmd.pack(fill="x", side="bottom", pady=(10,0))
        self.command = tk.Text(cmd, height=5, wrap="word", font=("Consolas", 9))
        self.command.pack(fill="x")

        for v in self.vars.values():
            try:
                v.trace_add("write", lambda *_: self._update_command())
            except Exception:
                pass
        self.after(100, self._update_command)

    # ------------------------------------------------------------------ #
    # log tab
    # ------------------------------------------------------------------ #
    def _log_tab(self):
        top = ttk.Frame(self.tab_log)
        top.pack(fill="x", pady=(0,6))
        ttk.Button(top, text="Clear Log", command=self._clear_log).pack(side="left")
        ttk.Button(top, text="Save Log...", command=self._save_log).pack(side="left", padx=6)

        frame = ttk.Frame(self.tab_log)
        frame.pack(fill="both", expand=True)

        self.log = tk.Text(
            frame, wrap="none", font=("Consolas", 9),
            background="#111315", foreground="#e6e6e6", insertbackground="white"
        )
        ys = ttk.Scrollbar(frame, orient="vertical", command=self.log.yview)
        xs = ttk.Scrollbar(frame, orient="horizontal", command=self.log.xview)
        self.log.configure(yscrollcommand=ys.set, xscrollcommand=xs.set)

        self.log.grid(row=0, column=0, sticky="nsew")
        ys.grid(row=0, column=1, sticky="ns")
        xs.grid(row=1, column=0, sticky="ew")
        frame.rowconfigure(0, weight=1)
        frame.columnconfigure(0, weight=1)

    # ------------------------------------------------------------------ #
    # results tab
    # ------------------------------------------------------------------ #
    def _results_tab(self):
        top = ttk.Frame(self.tab_results)
        top.pack(fill="x", pady=(0,6))
        ttk.Button(top, text="Refresh", command=self._refresh_results).pack(side="left")
        ttk.Button(top, text="Open Selected", command=self._open_selected_result).pack(side="left", padx=6)
        ttk.Button(top, text="Open Results Folder", command=self._open_results).pack(side="left", padx=6)

        pan = ttk.Panedwindow(self.tab_results, orient="horizontal")
        pan.pack(fill="both", expand=True)

        lf = ttk.LabelFrame(pan, text="Generated files", padding=6)
        rf = ttk.LabelFrame(pan, text="Preview", padding=6)
        pan.add(lf, weight=1)
        pan.add(rf, weight=3)

        self.result_list = tk.Listbox(lf, font=("Consolas", 9))
        self.result_list.pack(fill="both", expand=True)
        self.result_list.bind("<<ListboxSelect>>", lambda e: self._preview_selected_result())

        self.preview = tk.Text(rf, wrap="none", font=("Consolas", 9))
        pys = ttk.Scrollbar(rf, orient="vertical", command=self.preview.yview)
        pxs = ttk.Scrollbar(rf, orient="horizontal", command=self.preview.xview)
        self.preview.configure(yscrollcommand=pys.set, xscrollcommand=pxs.set)
        self.preview.grid(row=0, column=0, sticky="nsew")
        pys.grid(row=0, column=1, sticky="ns")
        pxs.grid(row=1, column=0, sticky="ew")
        rf.rowconfigure(0, weight=1)
        rf.columnconfigure(0, weight=1)


    # ------------------------------------------------------------------ #
    # circuit viewer
    # ------------------------------------------------------------------ #
    def _circuit_tab(self):
        top = ttk.Frame(self.tab_circuit)
        top.pack(fill="x", pady=(0, 8))

        controls = ttk.LabelFrame(top, text="Local circuit view", padding=8)
        controls.pack(fill="x")

        ttk.Label(controls, text="Chain").grid(row=0, column=0, sticky="w", padx=(0,4), pady=4)
        ttk.Entry(controls, textvariable=self.vars["view_chain"], width=8).grid(row=0, column=1, sticky="w", pady=4)

        ttk.Label(controls, text="Focus cell Q").grid(row=0, column=2, sticky="w", padx=(12,4), pady=4)
        ttk.Entry(controls, textvariable=self.vars["view_cell"], width=8).grid(row=0, column=3, sticky="w", pady=4)

        ttk.Label(controls, text="View").grid(row=0, column=4, sticky="w", padx=(12,4), pady=4)
        ttk.Combobox(
            controls,
            textvariable=self.vars["view_mode"],
            values=["scan-neighborhood", "fanin", "fanout", "both", "ambiguity"],
            state="readonly",
            width=18,
        ).grid(row=0, column=5, sticky="w", pady=4)

        ttk.Label(controls, text="Logic depth").grid(row=1, column=0, sticky="w", pady=4)
        ttk.Entry(controls, textvariable=self.vars["view_depth"], width=8).grid(row=1, column=1, sticky="w", pady=4)

        ttk.Label(controls, text="Max gates").grid(row=1, column=2, sticky="w", padx=(12,4), pady=4)
        ttk.Entry(controls, textvariable=self.vars["view_max_gates"], width=8).grid(row=1, column=3, sticky="w", pady=4)

        ttk.Label(controls, text="Scan radius").grid(row=1, column=4, sticky="w", padx=(12,4), pady=4)
        ttk.Entry(controls, textvariable=self.vars["view_scan_radius"], width=8).grid(row=1, column=5, sticky="w", pady=4)

        ttk.Label(controls, text="Ambiguous cells").grid(row=2, column=0, sticky="w", pady=4)
        ttk.Entry(
            controls,
            textvariable=self.vars["view_ambiguous_cells"],
            width=35,
        ).grid(row=2, column=1, columnspan=3, sticky="ew", pady=4)
        ttk.Label(
            controls,
            text="e.g. 196 197 198",
            foreground="#666666",
        ).grid(row=2, column=4, sticky="w", padx=(8,0), pady=4)

        ttk.Button(
            controls,
            text="Render Circuit",
            command=self._render_circuit,
        ).grid(row=0, column=6, rowspan=2, padx=(14,4), pady=4, sticky="ns")

        ttk.Button(
            controls,
            text="Load Ambiguity Classes",
            command=self._load_ambiguity_classes,
        ).grid(row=2, column=5, padx=(8,4), pady=4)

        self.ambiguity_combo = ttk.Combobox(
            controls,
            textvariable=self.vars["view_ambiguity_choice"],
            state="readonly",
            width=38,
        )
        self.ambiguity_combo.grid(row=2, column=6, sticky="ew", padx=(4,0), pady=4)
        self.ambiguity_combo.bind("<<ComboboxSelected>>", self._ambiguity_selected)

        controls.columnconfigure(3, weight=1)
        controls.columnconfigure(6, weight=1)

        ttk.Label(
            self.tab_circuit,
            text=(
                "Focused diagnostic viewer: scan neighborhood, fan-in/fan-out cones, "
                "and ambiguity-class comparisons. Whole-chain rendering is disabled."
            ),
            foreground="#555555",
        ).pack(fill="x", pady=(0, 6))

        tools = ttk.Frame(self.tab_circuit)
        tools.pack(fill="x", pady=(0, 6))

        self.graphviz_status = tk.StringVar(value="")
        ttk.Label(tools, textvariable=self.graphviz_status).pack(side="left")

        ttk.Button(tools, text="Fit", command=self._viewer_fit).pack(side="right")
        ttk.Button(tools, text="Zoom +", command=lambda: self._viewer_zoom_by(1.25)).pack(side="right", padx=4)
        ttk.Button(tools, text="Zoom −", command=lambda: self._viewer_zoom_by(0.8)).pack(side="right", padx=4)
        ttk.Button(tools, text="Open PNG", command=self._open_viewer_png).pack(side="right", padx=4)
        ttk.Button(tools, text="Open Circuit Folder", command=self._open_circuit_folder).pack(side="right", padx=4)

        # Scrollable canvas for rendered PNG.
        frame = ttk.Frame(self.tab_circuit)
        frame.pack(fill="both", expand=True)

        self.circuit_canvas = tk.Canvas(frame, background="white")
        ys = ttk.Scrollbar(frame, orient="vertical", command=self.circuit_canvas.yview)
        xs = ttk.Scrollbar(frame, orient="horizontal", command=self.circuit_canvas.xview)
        self.circuit_canvas.configure(yscrollcommand=ys.set, xscrollcommand=xs.set)

        self.circuit_canvas.grid(row=0, column=0, sticky="nsew")
        ys.grid(row=0, column=1, sticky="ns")
        xs.grid(row=1, column=0, sticky="ew")
        frame.rowconfigure(0, weight=1)
        frame.columnconfigure(0, weight=1)

        self._update_graphviz_status()

    def _update_graphviz_status(self):
        try:
            import scan_circuit_viewer_v5 as cv
            dot = cv.locate_dot()
        except Exception:
            dot = shutil.which("dot")

        if dot:
            self.graphviz_status.set(f"Graphviz: ready ({dot})")
        else:
            self.graphviz_status.set(
                "Graphviz: dot.exe not found — DOT export works, PNG/SVG rendering needs Graphviz installed"
            )

    def _parse_ambiguous_cells(self):
        raw = self.vars["view_ambiguous_cells"].get().replace(",", " ")
        vals = []
        for x in raw.split():
            x = x.strip()
            if not x:
                continue
            if x.upper().startswith("Q"):
                x = x[1:]
            vals.append(int(x))
        return sorted(set(vals))

    def _load_ambiguity_classes(self):
        try:
            chain = int(self.vars["view_chain"].get().strip())
        except Exception:
            messagebox.showerror("Circuit Viewer", "Chain must be an integer.")
            return

        path = self._outdir_path() / f"chain_{chain:04d}_ambiguity_classes.json"
        if not path.exists():
            messagebox.showinfo(
                "Circuit Viewer",
                f"No ambiguity-class file found for chain {chain}:\n{path}\n\n"
                "Run the full pipeline/targeted SAT first, or type ambiguous cells manually."
            )
            return

        try:
            records = json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:
            messagebox.showerror("Circuit Viewer", f"Could not read ambiguity file:\n{e}")
            return

        self.ambiguity_records = records
        values = []
        for rec in records:
            cells = rec.get("cells")
            if cells is None:
                cells = [f.get("cell") for f in rec.get("faults", [])]
            celltxt = ",".join(f"Q{int(c)}" for c in cells)
            values.append(
                f"Class {rec.get('class_id', '?')} | size {rec.get('size', len(cells))} | {celltxt}"
            )

        self.ambiguity_combo["values"] = values
        if values:
            self.ambiguity_combo.current(0)
            self._ambiguity_selected()
        else:
            messagebox.showinfo("Circuit Viewer", "No ambiguous classes in this file.")

    def _ambiguity_selected(self, event=None):
        idx = self.ambiguity_combo.current()
        if idx < 0 or idx >= len(self.ambiguity_records):
            return

        rec = self.ambiguity_records[idx]
        cells = rec.get("cells")
        if cells is None:
            cells = [f.get("cell") for f in rec.get("faults", [])]

        cells = [int(c) for c in cells]
        self.vars["view_ambiguous_cells"].set(
            " ".join(str(c) for c in cells)
        )
        if cells:
            self.vars["view_cell"].set(str(cells[0]))
        self.vars["view_mode"].set("ambiguity")

    def _viewer_design_config_key(self):
        return (
            int(self.vars["chains"].get()),
            int(self.vars["chain_length"].get()),
            int(self.vars["total_gates"].get()),
            int(self.vars["logic_seed"].get()),
            self.vars["design_cache"].get().strip(),
        )

    def _render_circuit(self):
        if self.vars["shared_logic"].get():
            messagebox.showinfo(
                "Circuit Viewer",
                "The current focused circuit viewer models the original "
                "local-block design only.\n\n"
                "Disable shared functional logic before rendering. "
                "Shared-simulator results remain fully available in the "
                "Results tab.",
            )
            return

        if self.viewer_thread is not None and self.viewer_thread.is_alive():
            messagebox.showinfo("Circuit Viewer", "A circuit render is already running.")
            return

        try:
            chain = int(self.vars["view_chain"].get().strip())
            cell = int(self.vars["view_cell"].get().strip())
            depth = int(self.vars["view_depth"].get().strip())
            max_gates = int(self.vars["view_max_gates"].get().strip())
            radius = int(self.vars["view_scan_radius"].get().strip())
            amb = self._parse_ambiguous_cells()
            config_key = self._viewer_design_config_key()
        except Exception as e:
            messagebox.showerror("Circuit Viewer", f"Invalid viewer parameter:\n{e}")
            return

        if max_gates <= 0 or depth < 0 or radius < 0:
            messagebox.showerror(
                "Circuit Viewer",
                "Depth/radius must be non-negative and max gates must be > 0."
            )
            return

        cache = Path(self.vars["design_cache"].get().strip()).expanduser()
        if self.vars["design_cache"].get().strip() and not cache.exists():
            ok = messagebox.askyesno(
                "Design cache not found",
                "The design cache does not exist yet.\n\n"
                "Build the synthetic design now using the current circuit parameters?\n"
                "This may take a little while for the 10M-gate configuration."
            )
            if not ok:
                return

        self.status.set("Rendering circuit...")
        self.graphviz_status.set("Circuit Viewer: working...")

        args = dict(
            chain=chain,
            focus_cell=cell,
            mode=self.vars["view_mode"].get(),
            depth=depth,
            max_gates=max_gates,
            scan_radius=radius,
            ambiguous_cells=amb,
            config_key=config_key,
        )

        self.viewer_thread = threading.Thread(
            target=self._viewer_worker,
            kwargs=args,
            daemon=True,
        )
        self.viewer_thread.start()

    def _viewer_worker(
        self,
        chain,
        focus_cell,
        mode,
        depth,
        max_gates,
        scan_radius,
        ambiguous_cells,
        config_key,
    ):
        try:
            import scan_circuit_viewer_v5 as cv

            if self.viewer_design is None or self.viewer_design_key != config_key:
                design, from_cache = cv.load_or_build_design(
                    chains=config_key[0],
                    chain_length=config_key[1],
                    total_gates=config_key[2],
                    logic_seed=config_key[3],
                    cache_path=config_key[4] or None,
                    build_progress=int(self.vars["design_progress"].get()),
                )
                self.viewer_design = design
                self.viewer_design_key = config_key
            else:
                from_cache = True

            out = self._outdir_path() / "circuit_views"
            result = cv.render_view(
                self.viewer_design,
                chain=chain,
                focus_cell=focus_cell,
                mode=mode,
                depth=depth,
                max_gates=max_gates,
                scan_radius=scan_radius,
                ambiguous_cells=ambiguous_cells,
                output_dir=out,
            )

            self.viewer_queue.put(
                ("done", result, from_cache)
            )
        except Exception as e:
            self.viewer_queue.put(("error", str(e), False))

    def _poll_viewer(self):
        try:
            while True:
                item = self.viewer_queue.get_nowait()
                kind = item[0]
                if kind == "done":
                    result = item[1]
                    self._viewer_finished(result)
                else:
                    self.status.set("Circuit render failed")
                    self._update_graphviz_status()
                    messagebox.showerror("Circuit Viewer", item[1])
        except queue.Empty:
            pass

        self.after(120, self._poll_viewer)

    def _viewer_finished(self, result):
        self.status.set("Circuit rendered")
        self._update_graphviz_status()

        if result.png_path is None:
            messagebox.showwarning(
                "Circuit Viewer",
                f"DOT file created:\n{result.dot_path}\n\n"
                "PNG/SVG could not be rendered because Graphviz 'dot' was not found.\n"
                "Install Graphviz and ensure its bin folder is on PATH."
            )
            return

        self.viewer_png_path = Path(result.png_path)

        try:
            self.viewer_original_image = Image.open(self.viewer_png_path).convert("RGB")
            self.viewer_zoom = 1.0
            self._viewer_fit()
        except Exception as e:
            messagebox.showerror("Circuit Viewer", f"Could not display PNG:\n{e}")
            return

        message = (
            f"Rendered {result.node_count} nodes "
            f"({result.scan_count} scan flops, {result.gate_count} gates).\n\n"
            f"PNG: {result.png_path}\n"
            f"SVG: {result.svg_path}\n"
            f"DOT: {result.dot_path}"
        )
        self._append("\n[Circuit Viewer] " + message.replace("\n", " ") + "\n")

    def _viewer_draw(self):
        if self.viewer_original_image is None:
            return

        w = max(1, int(self.viewer_original_image.width * self.viewer_zoom))
        h = max(1, int(self.viewer_original_image.height * self.viewer_zoom))

        resized = self.viewer_original_image.resize(
            (w, h),
            Image.Resampling.LANCZOS,
        )
        self.viewer_photo = ImageTk.PhotoImage(resized)

        self.circuit_canvas.delete("all")
        self.circuit_canvas.create_image(
            0, 0,
            anchor="nw",
            image=self.viewer_photo,
        )
        self.circuit_canvas.configure(
            scrollregion=(0, 0, w, h)
        )

    def _viewer_zoom_by(self, factor):
        if self.viewer_original_image is None:
            return
        self.viewer_zoom = min(4.0, max(0.1, self.viewer_zoom * factor))
        self._viewer_draw()

    def _viewer_fit(self):
        if self.viewer_original_image is None:
            return

        self.update_idletasks()
        cw = max(100, self.circuit_canvas.winfo_width() - 20)
        ch = max(100, self.circuit_canvas.winfo_height() - 20)

        zx = cw / self.viewer_original_image.width
        zy = ch / self.viewer_original_image.height
        self.viewer_zoom = min(1.0, zx, zy)
        self._viewer_draw()

    def _open_viewer_png(self):
        if self.viewer_png_path is None or not self.viewer_png_path.exists():
            messagebox.showinfo("Circuit Viewer", "Render a circuit first.")
            return
        self._open_path(self.viewer_png_path)

    def _open_circuit_folder(self):
        folder = self._outdir_path() / "circuit_views"
        folder.mkdir(parents=True, exist_ok=True)
        self._open_path(folder)


    # ------------------------------------------------------------------ #
    # field helpers
    # ------------------------------------------------------------------ #
    def _entry(self, parent, row, label, key, hint=""):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=5)
        ttk.Entry(parent, textvariable=self.vars[key]).grid(
            row=row, column=1, sticky="ew", padx=6, pady=5
        )
        if hint:
            ttk.Label(parent, text=hint, foreground="#666666").grid(
                row=row, column=2, sticky="w", pady=5
            )

    def _combo(self, parent, row, label, key, values):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=5)
        ttk.Combobox(
            parent, textvariable=self.vars[key],
            values=values, state="readonly"
        ).grid(row=row, column=1, sticky="ew", padx=6, pady=5)

    # ------------------------------------------------------------------ #
    # presets
    # ------------------------------------------------------------------ #
    def _preset_quick(self):
        self.vars["shared_logic"].set(False)
        self.vars["mode"].set("pool-scale")
        self.vars["chains"].set("20")
        self.vars["chain_length"].set("100")
        self.vars["total_gates"].set("200000")
        self.vars["failed_chains"].set("2")
        self.vars["pattern_counts"].set("16 32 64")
        self.vars["only_chain"].set("")
        self.vars["infer_stuck"].set(True)
        self.vars["design_cache"].set("")
        self.vars["outdir"].set("quick_test")

    def _preset_full(self):
        self.vars["shared_logic"].set(False)
        self.vars["mode"].set("all")
        self.vars["chains"].set("1000")
        self.vars["chain_length"].set("1000")
        self.vars["total_gates"].set("10000000")
        self.vars["failed_chains"].set("5")
        self.vars["pattern_counts"].set("64 128 256 512 1024")
        self.vars["max_selected"].set("50")
        self.vars["sat_conf_budget"].set("500000")
        self.vars["only_chain"].set("")
        self.vars["infer_stuck"].set(True)
        self.vars["single_flush"].set(False)
        self.vars["design_cache"].set("scan_design_1m_10m_seed12345.npz")
        self.vars["outdir"].set("local_full_run")

    def _preset_pool(self):
        self._preset_full()
        self.vars["mode"].set("pool-scale")
        self.vars["outdir"].set("local_pool_scaling")

    def _preset_sat(self):
        self._preset_full()
        self.vars["mode"].set("targeted-sat")
        self.vars["pattern_counts"].set("1024")
        self.vars["only_chain"].set("210")
        self.vars["outdir"].set("local_sat_chain210")

    def _preset_shared_quick(self):
        self.vars["shared_logic"].set(True)
        self.vars["mode"].set("pool-scale")

        self.vars["chains"].set("16")
        self.vars["chain_length"].set("64")
        self.vars["total_gates"].set("6400")
        self.vars["failed_chains"].set("2")

        self.vars["pattern_counts"].set("16 32 64")
        self.vars["max_selected"].set("50")
        self.vars["only_chain"].set("")

        self.vars["infer_stuck"].set(True)
        self.vars["single_flush"].set(False)

        self.vars["shared_logic_fraction"].set("0.05")
        self.vars["shared_partners"].set("4")
        self.vars["shared_seed"].set("314159")

        self.vars["design_cache"].set(
            "shared_design_quick_f0p05.npz"
        )
        self.vars["outdir"].set(
            "shared_quick_test"
        )

    def _preset_shared_full(self):
        self.vars["shared_logic"].set(True)
        self.vars["mode"].set("pool-scale")

        self.vars["chains"].set("1000")
        self.vars["chain_length"].set("1000")
        self.vars["total_gates"].set("10000000")
        self.vars["failed_chains"].set("10")

        self.vars["pattern_counts"].set(
            "64 128 256 512 1024"
        )
        self.vars["max_selected"].set("50")
        self.vars["sat_conf_budget"].set("500000")
        self.vars["only_chain"].set("")

        self.vars["infer_stuck"].set(True)
        self.vars["single_flush"].set(False)

        self.vars["shared_logic_fraction"].set("0.01")
        self.vars["shared_partners"].set("4")
        self.vars["shared_seed"].set("314159")

        self.vars["design_cache"].set(
            "shared_design_f0p01.npz"
        )
        self.vars["outdir"].set(
            "shared_logic_full_f0p01"
        )

    def _shared_logic_toggled(self):
        if self.vars["shared_logic"].get():
            if (
                self.vars["design_cache"].get().strip()
                == "scan_design_1m_10m_seed12345.npz"
            ):
                self.vars["design_cache"].set(
                    "shared_design_f0p01.npz"
                )
        else:
            if (
                self.vars["design_cache"].get().strip()
                == "shared_design_f0p01.npz"
            ):
                self.vars["design_cache"].set(
                    "scan_design_1m_10m_seed12345.npz"
                )

        self._update_command()

    # ------------------------------------------------------------------ #
    # command
    # ------------------------------------------------------------------ #
    def _sim_name(self):
        if self.vars["shared_logic"].get():
            return SHARED_SIMULATOR_NAME
        return SIMULATOR_NAME

    def _sim_path(self):
        return (
            Path(__file__).resolve().parent
            / self._sim_name()
        )

    def _cmd(self):
        pats = [
            x for x in self.vars["pattern_counts"].get().replace(",", " ").split()
            if x.strip()
        ]
        c = [
            sys.executable, "-u", str(self._sim_path()),
            "--mode", self.vars["mode"].get(),
            "--chains", self.vars["chains"].get().strip(),
            "--chain-length", self.vars["chain_length"].get().strip(),
            "--total-gates", self.vars["total_gates"].get().strip(),
            "--failed-chains", self.vars["failed_chains"].get().strip(),
            "--pattern-counts", *pats,
            "--max-selected", self.vars["max_selected"].get().strip(),
            "--sat-conf-budget", self.vars["sat_conf_budget"].get().strip(),
            "--seed", self.vars["seed"].get().strip(),
            "--logic-seed", self.vars["logic_seed"].get().strip(),
            "--pattern-seed", self.vars["pattern_seed"].get().strip(),
            "--hash-seed", self.vars["hash_seed"].get().strip(),
            "--design-progress", self.vars["design_progress"].get().strip(),
            "--response-progress", self.vars["response_progress"].get().strip(),
            "--pair-progress", self.vars["pair_progress"].get().strip(),
            "--outdir", self.vars["outdir"].get().strip(),
        ]

        if self.vars["shared_logic"].get():
            c += [
                "--shared-logic-fraction",
                self.vars["shared_logic_fraction"].get().strip(),
                "--shared-partners",
                self.vars["shared_partners"].get().strip(),
                "--shared-seed",
                self.vars["shared_seed"].get().strip(),
            ]

        if self.vars["infer_stuck"].get():
            c.append("--infer-stuck-from-flush")
        if self.vars["single_flush"].get():
            c.append("--single-flush")
        if self.vars["only_chain"].get().strip():
            c += ["--only-chain", self.vars["only_chain"].get().strip()]
        if self.vars["threads"].get().strip() not in ("", "0"):
            c += ["--threads", self.vars["threads"].get().strip()]
        if self.vars["design_cache"].get().strip():
            c += ["--design-cache", self.vars["design_cache"].get().strip()]
        return c

    def _update_command(self):
        try:
            c = self._cmd()
            s = subprocess.list2cmdline(c) if os.name == "nt" else " ".join(shlex.quote(x) for x in c)
        except Exception as e:
            s = f"<invalid: {e}>"
        self.command.delete("1.0", "end")
        self.command.insert("1.0", s)

    def _validate(self):
        if not self._sim_path().exists():
            raise ValueError(
                f"{self._sim_name()} must be in the same folder "
                "as this GUI."
            )

        numeric = [
            "chains","chain_length","total_gates","failed_chains",
            "max_selected","sat_conf_budget","seed","logic_seed",
            "pattern_seed","hash_seed","design_progress",
            "response_progress","pair_progress"
        ]
        for k in numeric:
            int(self.vars[k].get().strip())

        if self.vars["threads"].get().strip():
            int(self.vars["threads"].get().strip())
        if self.vars["only_chain"].get().strip():
            int(self.vars["only_chain"].get().strip())

        if self.vars["shared_logic"].get():
            frac = float(
                self.vars["shared_logic_fraction"].get().strip()
            )
            partners = int(
                self.vars["shared_partners"].get().strip()
            )
            int(
                self.vars["shared_seed"].get().strip()
            )

            chains = int(
                self.vars["chains"].get().strip()
            )

            if not 0.0 <= frac <= 1.0:
                raise ValueError(
                    "Shared logic fraction must be between 0 and 1."
                )

            if (
                partners < 0
                or partners > max(0, chains - 1)
            ):
                raise ValueError(
                    "Shared partners must be between "
                    "0 and chains - 1."
                )

            if (
                frac > 0.0
                and partners == 0
            ):
                raise ValueError(
                    "A positive shared logic fraction requires "
                    "at least one shared partner."
                )

            cache = (
                self.vars["design_cache"]
                .get()
                .strip()
            )

            if (
                cache
                and Path(cache).name
                == "scan_design_1m_10m_seed12345.npz"
            ):
                raise ValueError(
                    "Shared mode must use a separate shared-design "
                    "cache. For example: "
                    "shared_design_f0p01.npz"
                )

        ps = self.vars["pattern_counts"].get().replace(",", " ").split()
        if not ps:
            raise ValueError("Pattern pools cannot be empty.")
        [int(x) for x in ps]

        if not self.vars["outdir"].get().strip():
            raise ValueError("Choose an output folder.")

    # ------------------------------------------------------------------ #
    # run/stop
    # ------------------------------------------------------------------ #
    def _run(self):
        if self.proc is not None:
            messagebox.showwarning("Already running", "An experiment is already running.")
            return
        try:
            self._validate()

            # Check native PySAT availability before launching a long run.
            # On managed Windows machines the pysolvers DLL may be blocked
            # by Application Control even when python-sat is installed.
            if self.vars["mode"].get() in (
                "all",
                "targeted-sat",
            ):
                try:
                    from pysat.solvers import Glucose4  # noqa: F401
                except Exception as sat_exc:
                    use_pool = messagebox.askyesno(
                        "Exact SAT unavailable",
                        "The selected mode requires "
                        "python-sat/Glucose4, but it could not "
                        "be loaded.\n\n"
                        f"{sat_exc}\n\n"
                        "Run the experiment in pool-scale mode instead?",
                    )

                    if not use_pool:
                        return

                    self.vars["mode"].set(
                        "pool-scale"
                    )

            cmd = self._cmd()

        except Exception as e:
            messagebox.showerror(
                "Configuration error",
                str(e),
            )
            return

        out = Path(self.vars["outdir"].get().strip()).expanduser()
        out.mkdir(parents=True, exist_ok=True)

        self.nb.select(self.tab_log)
        self._append("\n" + "="*90 + "\n")
        self._append("RUN STARTED\n")
        self._append("="*90 + "\n")
        self._append((subprocess.list2cmdline(cmd) if os.name=="nt" else " ".join(shlex.quote(x) for x in cmd)) + "\n\n")

        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
        try:
            self.proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                cwd=str(Path(__file__).resolve().parent),
                creationflags=flags,
            )
        except Exception as e:
            self.proc = None
            messagebox.showerror("Launch failed", str(e))
            return

        self.start_time = time.time()
        self.status.set(f"Running — PID {self.proc.pid}")
        self.run_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")
        self.progress.start(10)
        self._tick_elapsed()

        self.reader = threading.Thread(target=self._read_worker, daemon=True)
        self.reader.start()

    def _read_worker(self):
        p = self.proc
        try:
            for line in p.stdout:
                self.q.put(("line", line))
            rc = p.wait()
            self.q.put(("done", rc))
        except Exception as e:
            self.q.put(("error", str(e)))

    def _poll(self):
        try:
            while True:
                kind, value = self.q.get_nowait()
                if kind == "line":
                    self._append(value)
                elif kind == "done":
                    self._finish(value)
                else:
                    self._append(f"\n[GUI error] {value}\n")
                    self._finish(-1)
        except queue.Empty:
            pass
        self.after(100, self._poll)

    def _finish(self, rc):
        if self.proc is None:
            return
        if rc == 0:
            self.status.set("Completed successfully")
            self._append("\n[GUI] Run completed. Result files are ready.\n")
            self.after(600, self._open_dashboard)
        else:
            self.status.set(f"Ended with exit code {rc}")
            self._append(f"\n[GUI] Process ended with exit code {rc}.\n")
        self.proc = None
        self.progress.stop()
        self.run_btn.configure(state="normal")
        self.stop_btn.configure(state="disabled")
        self._refresh_results()
        self.nb.select(self.tab_results)

    def _stop(self):
        if self.proc is None:
            return
        if not messagebox.askyesno("Stop experiment", "Terminate the current run?"):
            return
        try:
            self.proc.terminate()
        except Exception as e:
            messagebox.showerror("Stop failed", str(e))

    def _tick_elapsed(self):
        if self.proc is None or self.start_time is None:
            return
        n = int(time.time() - self.start_time)
        h, r = divmod(n, 3600)
        m, s = divmod(r, 60)
        self.elapsed.set(f"Elapsed {h:02d}:{m:02d}:{s:02d}")
        self.after(1000, self._tick_elapsed)

    def _open_dashboard(self):
        out = self._outdir_path()
        meta = out / "run_metadata.json"

        if not meta.exists():
            messagebox.showinfo(
                "Experiment Dashboard",
                "No completed experiment metadata was found in the current output folder.\n\n"
                "Run an experiment first, or point Output folder to an existing result directory."
            )
            return

        try:
            from scan_results_dashboard import open_dashboard
            open_dashboard(self, out)
        except Exception as e:
            messagebox.showerror(
                "Experiment Dashboard",
                f"Could not open the dashboard:\n\n{e}"
            )

    # ------------------------------------------------------------------ #
    # log
    # ------------------------------------------------------------------ #
    def _append(self, text):
        self.log.insert("end", text)
        self.log.see("end")

    def _clear_log(self):
        self.log.delete("1.0", "end")

    def _save_log(self):
        fn = filedialog.asksaveasfilename(
            defaultextension=".txt",
            initialfile="scan_diagnosis_run.log.txt",
            filetypes=[("Text", "*.txt"), ("All files", "*.*")]
        )
        if fn:
            Path(fn).write_text(self.log.get("1.0", "end-1c"), encoding="utf-8")

    # ------------------------------------------------------------------ #
    # results
    # ------------------------------------------------------------------ #
    def _outdir_path(self):
        return Path(self.vars["outdir"].get().strip()).expanduser().resolve()

    def _refresh_results_silent(self):
        try:
            self._refresh_results(show_message=False)
        except Exception:
            pass

    def _refresh_results(self, show_message=True):
        out = self._outdir_path()
        self.result_list.delete(0, "end")
        self.preview.delete("1.0", "end")

        if not out.exists():
            if show_message:
                messagebox.showinfo("Results", "The output folder does not exist yet.")
            return

        files = sorted(
            [p for p in out.iterdir() if p.is_file()],
            key=lambda p: p.name.lower()
        )
        for p in files:
            self.result_list.insert("end", p.name)

        if not files and show_message:
            messagebox.showinfo(
                "Results",
                "No result files yet. Click RUN & GENERATE RESULTS first."
            )

    def _selected_result(self):
        sel = self.result_list.curselection()
        if not sel:
            return None
        name = self.result_list.get(sel[0])
        return self._outdir_path() / name

    def _preview_selected_result(self):
        p = self._selected_result()
        if p is None:
            return
        self.preview.delete("1.0", "end")
        try:
            if p.suffix.lower() == ".json":
                obj = json.loads(p.read_text(encoding="utf-8"))
                text = json.dumps(obj, indent=2)
            elif p.suffix.lower() == ".csv":
                # Preview up to first 200 lines.
                lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
                text = "\n".join(lines[:200])
                if len(lines) > 200:
                    text += f"\n\n... ({len(lines)-200} more lines)"
            elif p.suffix.lower() in (".txt", ".log"):
                text = p.read_text(encoding="utf-8", errors="replace")[:100000]
            else:
                text = (
                    f"{p.name}\n\nBinary/non-text result file.\n"
                    f"Size: {p.stat().st_size:,} bytes"
                )
            self.preview.insert("1.0", text)
        except Exception as e:
            self.preview.insert("1.0", f"Preview failed:\n{e}")

    def _open_selected_result(self):
        p = self._selected_result()
        if p is None:
            messagebox.showinfo("Results", "Select a result file first.")
            return
        self._open_path(p)

    def _open_results(self):
        out = self._outdir_path()
        out.mkdir(parents=True, exist_ok=True)
        self._open_path(out)

    def _open_path(self, p):
        try:
            if os.name == "nt":
                os.startfile(str(p))
            elif sys.platform == "darwin":
                subprocess.Popen(["open", str(p)])
            else:
                subprocess.Popen(["xdg-open", str(p)])
        except Exception as e:
            messagebox.showerror("Open failed", str(e))

    # ------------------------------------------------------------------ #
    # browse
    # ------------------------------------------------------------------ #
    def _browse_outdir(self):
        p = filedialog.askdirectory(title="Choose output folder")
        if p:
            self.vars["outdir"].set(p)
            self._refresh_results_silent()

    def _browse_cache(self):
        current = self.vars["design_cache"].get().strip()
        fn = filedialog.asksaveasfilename(
            title="Choose design cache file",
            defaultextension=".npz",
            initialfile=Path(current or "scan_design_1m_10m_seed12345.npz").name,
            filetypes=[("NumPy cache", "*.npz"), ("All files", "*.*")]
        )
        if fn:
            self.vars["design_cache"].set(fn)


if __name__ == "__main__":
    App().mainloop()
