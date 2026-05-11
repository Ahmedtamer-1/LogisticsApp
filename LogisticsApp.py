#!/usr/bin/env python3
"""
Smart Logistics Optimization Platform
======================================
Implements: Dijkstra's Algorithm, 0/1 Knapsack (DP),
            Prim's MST, Kruskal's MST, Greedy Bin Packing (FFD)

Run:  python logistics_platform.py
Deps: pip install matplotlib networkx numpy
"""

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import networkx as nx
import numpy as np
import random
import time
import heapq
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional

# ─────────────────────────────────────────────────────────────
#  DATA MODELS
# ─────────────────────────────────────────────────────────────

@dataclass
class Order:
    id: int
    name: str
    weight: float
    value: float
    priority: str
    destination: str


# ─────────────────────────────────────────────────────────────
#  ALGORITHMS (pure functions, no UI dependency)
# ─────────────────────────────────────────────────────────────

class Algorithms:

    # ── 1. DIJKSTRA ──────────────────────────────────────────
    @staticmethod
    def dijkstra(adj: dict, source: str, nodes: list):
        """
        Single-source shortest path using a min-heap.
        Complexity: O((V + E) log V)

        Returns
        -------
        dist  : {node: float}   shortest distances from source
        prev  : {node: node}    predecessor map for path reconstruction
        steps : list[dict]      trace for step-by-step UI
        """
        dist = {n: float("inf") for n in nodes}
        prev = {n: None for n in nodes}
        dist[source] = 0
        pq   = [(0, source)]
        vis  = set()
        steps = []

        while pq:
            d, u = heapq.heappop(pq)
            if u in vis:
                continue
            vis.add(u)
            steps.append({
                "action":    f"Visit {u}  distance={d:.1f}",
                "node":      u,
                "distances": dict(dist),
                "visited":   set(vis),
            })
            for v, w in adj.get(u, {}).items():
                if dist[u] + w < dist[v]:
                    dist[v] = dist[u] + w
                    prev[v] = u
                    heapq.heappush(pq, (dist[v], v))
                    steps[-1]["action"] += f"  |  relax {u}→{v} new={dist[v]:.1f}"

        return dist, prev, steps

    @staticmethod
    def reconstruct_path(prev: dict, target: str) -> list:
        path, node = [], target
        while node is not None:
            path.append(node)
            node = prev[node]
        return list(reversed(path))

    # ── 2. 0/1 KNAPSACK (DP) ────────────────────────────────
    @staticmethod
    def knapsack_01(weights: list, values: list, capacity: float):
        """
        Bottom-up DP knapsack.
        Complexity: O(n * W)

        Returns
        -------
        max_val  : float        best total value
        selected : list[int]    indices of chosen items
        dp       : 2-D list     full DP table (for visualisation)
        steps    : list[dict]   trace
        """
        n = len(weights)
        W = int(capacity)
        dp = [[0] * (W + 1) for _ in range(n + 1)]
        steps = []

        for i in range(1, n + 1):
            wi, vi = int(weights[i - 1]), values[i - 1]
            for w in range(W + 1):
                if wi <= w:
                    dp[i][w] = max(dp[i - 1][w], dp[i - 1][w - wi] + vi)
                else:
                    dp[i][w] = dp[i - 1][w]
            steps.append({
                "action":  f"Item {i}  weight={wi}  value={vi:.1f}  "
                           f"best so far={dp[i][W]:.1f}",
                "item":    i - 1,
                "dp_row":  list(dp[i]),
            })

        # back-track
        selected, w = [], W
        for i in range(n, 0, -1):
            if dp[i][w] != dp[i - 1][w]:
                selected.append(i - 1)
                w -= int(weights[i - 1])

        return dp[n][W], selected, dp, steps

    # ── 3. PRIM'S MST ────────────────────────────────────────
    @staticmethod
    def prims_mst(nodes: list, edges: list):
        """
        Prim's minimum spanning tree (greedy, priority-queue).
        Complexity: O(E log V)

        Returns
        -------
        mst_edges    : list[(u,v,w)]
        total_weight : float
        steps        : list[dict]
        """
        adj = defaultdict(list)
        for u, v, w in edges:
            adj[u].append((w, v))
            adj[v].append((w, u))

        if not nodes:
            return [], 0, []

        start   = nodes[0]
        visited = {start}
        pq      = [(w, start, v) for w, v in adj[start]]
        heapq.heapify(pq)
        mst_edges, total, steps = [], 0, []

        steps.append({"action": f"Start at {start}", "mst_edges": [], "visited": {start}})

        while pq and len(visited) < len(nodes):
            w, u, v = heapq.heappop(pq)
            if v in visited:
                continue
            visited.add(v)
            mst_edges.append((u, v, w))
            total += w
            for w2, nb in adj[v]:
                if nb not in visited:
                    heapq.heappush(pq, (w2, v, nb))
            steps.append({
                "action":     f"Add {u}↔{v}  w={w:.1f}  MST cost={total:.1f}",
                "mst_edges":  list(mst_edges),
                "visited":    set(visited),
            })

        return mst_edges, total, steps

    # ── 4. KRUSKAL'S MST ────────────────────────────────────
    @staticmethod
    def kruskals_mst(nodes: list, edges: list):
        """
        Kruskal's MST using Union-Find with path compression & rank.
        Complexity: O(E log E)

        Returns
        -------
        mst_edges    : list[(u,v,w)]
        total_weight : float
        steps        : list[dict]
        """
        parent = {n: n for n in nodes}
        rank   = {n: 0 for n in nodes}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x, y):
            rx, ry = find(x), find(y)
            if rx == ry:
                return False
            if rank[rx] < rank[ry]:
                rx, ry = ry, rx
            parent[ry] = rx
            if rank[rx] == rank[ry]:
                rank[rx] += 1
            return True

        sorted_edges = sorted(edges, key=lambda e: e[2])
        mst_edges, total, steps = [], 0, []

        for u, v, w in sorted_edges:
            if union(u, v):
                mst_edges.append((u, v, w))
                total += w
                steps.append({
                    "action":   f"ACCEPT {u}↔{v}  w={w:.1f}  MST cost={total:.1f}",
                    "mst_edges": list(mst_edges),
                    "accepted": True,
                    "edge":     (u, v, w),
                    "sets":     {n: find(n) for n in nodes},
                })
            else:
                steps.append({
                    "action":   f"SKIP {u}↔{v}  w={w:.1f}  (would form cycle)",
                    "mst_edges": list(mst_edges),
                    "accepted": False,
                    "edge":     (u, v, w),
                    "sets":     {n: find(n) for n in nodes},
                })

        return mst_edges, total, steps

    # ── 5. FIRST-FIT DECREASING BIN PACKING ─────────────────
    @staticmethod
    def first_fit_decreasing(items: list, bin_capacity: float):
        """
        Greedy FFD bin packing approximation.
        Approximation ratio: ≤ 11/9 * OPT + 6/9
        Complexity: O(n log n)

        items  : list[(item_id, size)]
        Returns: (bins, steps)
                 bins = list[(item_list, used_capacity)]
        """
        sorted_items = sorted(items, key=lambda x: x[1], reverse=True)
        bins, steps  = [], []

        for item_id, size in sorted_items:
            placed = False
            for i, (bin_items, used) in enumerate(bins):
                if used + size <= bin_capacity:
                    bins[i] = (bin_items + [(item_id, size)], used + size)
                    steps.append({
                        "action": f"Place #{item_id}  size={size:.1f}  → Truck {i+1}",
                        "bins":   [(list(bi), bu) for bi, bu in bins],
                    })
                    placed = True
                    break
            if not placed:
                bins.append(([(item_id, size)], size))
                steps.append({
                    "action": f"Open Truck {len(bins)} for #{item_id}  size={size:.1f}",
                    "bins":   [(list(bi), bu) for bi, bu in bins],
                })

        return bins, steps


# ─────────────────────────────────────────────────────────────
#  COLOUR PALETTE
# ─────────────────────────────────────────────────────────────
BG        = "#1a1d2e"
PANEL     = "#252842"
ACCENT    = "#4a8fe8"
GREEN     = "#56cfaa"
ORANGE    = "#f5a623"
RED       = "#f56060"
PURPLE    = "#a78bfa"
TEXT      = "#e0e6f0"
MUTED     = "#7a869a"
EDGE_DEF  = "#3d4266"

NODE_COLORS = {
    "supplier":  "#f5a623",
    "warehouse": "#4a8fe8",
    "customer":  "#56cfaa",
}


# ─────────────────────────────────────────────────────────────
#  MAIN APPLICATION
# ─────────────────────────────────────────────────────────────

class LogisticsApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("🚚  Smart Logistics Optimization Platform")
        self.root.geometry("1420x920")
        self.root.configure(bg=BG)

        # graph & scenario state
        self.G          = nx.DiGraph()
        self.node_types = {}
        self.pos        = {}
        self.orders: List[Order] = []

        # algo results
        self.res_dijkstra   = {}
        self.res_knapsack   = {}
        self.res_prim       = {}
        self.res_kruskal    = {}
        self.res_binpack    = {}
        self._perf_times    = {}
        self._approx_ratios = {}

        # step-by-step
        self.step_steps: list = []
        self.step_idx:   int  = 0

        self._apply_theme()
        self._build_ui()
        self._randomize()

    # ─────────────── THEME ───────────────────────────────────
    def _apply_theme(self):
        s = ttk.Style()
        s.theme_use("clam")

        s.configure("TNotebook",     background=BG,    borderwidth=0)
        s.configure("TNotebook.Tab", background=PANEL, foreground=MUTED,
                    padding=[14, 6], font=("Segoe UI", 10))
        s.map("TNotebook.Tab",
              background=[("selected", ACCENT)],
              foreground=[("selected", "white")])

        s.configure("TFrame",          background=BG)
        s.configure("TPanedwindow",    background=BG)
        s.configure("TSeparator",      background=EDGE_DEF)

        s.configure("Treeview",
                    background=PANEL, foreground=TEXT,
                    fieldbackground=PANEL, font=("Consolas", 9),
                    rowheight=22)
        s.configure("Treeview.Heading",
                    background="#1e3a5f", foreground="white",
                    font=("Segoe UI", 9, "bold"))
        s.map("Treeview", background=[("selected", ACCENT)])

        s.configure("Vertical.TScrollbar", background=PANEL, troughcolor=BG)

    # ─────────────── TOP-LEVEL LAYOUT ────────────────────────
    def _build_ui(self):
        # ── header bar ──
        hdr = tk.Frame(self.root, bg="#151728", height=52)
        hdr.pack(fill=tk.X)
        hdr.pack_propagate(False)

        tk.Label(hdr, text="🚚  Smart Logistics Optimization Platform",
                 font=("Segoe UI", 15, "bold"),
                 bg="#151728", fg=ACCENT).pack(side=tk.LEFT, padx=18, pady=12)
        tk.Label(hdr,
                 text="Dijkstra  ·  0/1 Knapsack DP  ·  Prim's MST  "
                      "·  Kruskal's MST  ·  Bin Packing (FFD)",
                 font=("Segoe UI", 9), bg="#151728", fg=MUTED).pack(side=tk.LEFT)

        # ── toolbar ──
        self._build_toolbar()

        # ── notebook ──
        self.nb = ttk.Notebook(self.root)
        self.nb.pack(fill=tk.BOTH, expand=True, padx=6, pady=(4, 0))

        self._tab_graph    = ttk.Frame(self.nb)
        self._tab_knapsack = ttk.Frame(self.nb)
        self._tab_mst      = ttk.Frame(self.nb)
        self._tab_bin      = ttk.Frame(self.nb)
        self._tab_compare  = ttk.Frame(self.nb)

        self.nb.add(self._tab_graph,    text="📍  Graph & Dijkstra")
        self.nb.add(self._tab_knapsack, text="📦  Knapsack DP")
        self.nb.add(self._tab_mst,      text="🌲  MST Comparison")
        self.nb.add(self._tab_bin,      text="🚛  Bin Packing")
        self.nb.add(self._tab_compare,  text="📊  Algorithm Comparison")

        self._build_graph_tab()
        self._build_knapsack_tab()
        self._build_mst_tab()
        self._build_bin_tab()
        self._build_compare_tab()

        # ── trace panel ──
        self._build_trace_panel()

    # ─────────────── TOOLBAR ────────────────────────────────
    def _build_toolbar(self):
        bar = tk.Frame(self.root, bg=PANEL, height=46)
        bar.pack(fill=tk.X, padx=6, pady=(4, 0))
        bar.pack_propagate(False)

        def btn(parent, text, cmd, color=ACCENT):
            b = tk.Button(parent, text=text, command=cmd,
                          bg=color, fg="white",
                          font=("Segoe UI", 9, "bold"),
                          relief=tk.FLAT, bd=0,
                          padx=12, pady=4, cursor="hand2",
                          activebackground=color, activeforeground="white")
            b.pack(side=tk.LEFT, padx=4, pady=8)
            return b

        btn(bar, "🎲  Randomize",       self._randomize,  "#2d5fa0")
        btn(bar, "▶  Run All",          self._run_all,    "#1e7e4f")

        tk.Label(bar, text="  Nodes:", bg=PANEL, fg=MUTED,
                 font=("Segoe UI", 9)).pack(side=tk.LEFT, pady=8)
        self._v_nodes = tk.IntVar(value=9)
        tk.Spinbox(bar, from_=5, to=20, textvariable=self._v_nodes,
                   width=4, bg="#1e2138", fg=TEXT,
                   font=("Segoe UI", 9), relief=tk.FLAT).pack(side=tk.LEFT, pady=8)

        tk.Label(bar, text="  Edges:", bg=PANEL, fg=MUTED,
                 font=("Segoe UI", 9)).pack(side=tk.LEFT, pady=8)
        self._v_edges = tk.IntVar(value=13)
        tk.Spinbox(bar, from_=5, to=40, textvariable=self._v_edges,
                   width=4, bg="#1e2138", fg=TEXT,
                   font=("Segoe UI", 9), relief=tk.FLAT).pack(side=tk.LEFT, pady=8)

        tk.Label(bar, text="  Vehicle cap (kg):", bg=PANEL, fg=MUTED,
                 font=("Segoe UI", 9)).pack(side=tk.LEFT, pady=8)
        self._v_cap = tk.DoubleVar(value=100.0)
        tk.Entry(bar, textvariable=self._v_cap, width=6,
                 bg="#1e2138", fg=TEXT, font=("Segoe UI", 9),
                 relief=tk.FLAT).pack(side=tk.LEFT, pady=8)

        # separator
        tk.Label(bar, text="  │  Step:", bg=PANEL, fg=MUTED,
                 font=("Segoe UI", 9)).pack(side=tk.LEFT, pady=8)

        self._v_step_algo = tk.StringVar(value="Dijkstra")
        ttk.Combobox(bar, textvariable=self._v_step_algo,
                     values=["Dijkstra", "Knapsack", "Prim's", "Kruskal's", "Bin Packing"],
                     width=13, state="readonly").pack(side=tk.LEFT, padx=4, pady=8)

        btn(bar, "⏮ Reset",   self._step_reset,   "#7c3aed")
        btn(bar, "⏭ Step →",  self._step_forward, "#7c3aed")

        self._lbl_step = tk.Label(bar, text="0 / 0", bg=PANEL, fg=ORANGE,
                                  font=("Consolas", 9, "bold"))
        self._lbl_step.pack(side=tk.LEFT, padx=8)

    # ─────────────── TAB: GRAPH & DIJKSTRA ───────────────────
    def _build_graph_tab(self):
        pw = tk.PanedWindow(self._tab_graph, orient=tk.HORIZONTAL,
                            bg=BG, sashwidth=5, sashrelief=tk.FLAT)
        pw.pack(fill=tk.BOTH, expand=True)

        # graph canvas
        left = tk.Frame(pw, bg=BG)
        pw.add(left, width=820)

        self._fig_g, self._ax_g = plt.subplots(figsize=(8, 6), facecolor=BG)
        self._ax_g.set_facecolor(BG)
        self._cv_g = FigureCanvasTkAgg(self._fig_g, master=left)
        self._cv_g.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # right panel
        right = tk.Frame(pw, bg=PANEL)
        pw.add(right, width=400)

        tk.Label(right, text="Dijkstra — Shortest Paths",
                 font=("Segoe UI", 11, "bold"), bg=PANEL, fg=ACCENT).pack(
                     pady=(12, 4), padx=12, anchor="w")

        sf = tk.Frame(right, bg=PANEL)
        sf.pack(fill=tk.X, padx=12)
        tk.Label(sf, text="Source:", bg=PANEL, fg=MUTED,
                 font=("Segoe UI", 9)).pack(side=tk.LEFT)
        self._v_src = tk.StringVar()
        self._cb_src = ttk.Combobox(sf, textvariable=self._v_src,
                                    width=12, state="readonly")
        self._cb_src.pack(side=tk.LEFT, padx=6)

        tk.Button(right, text="Run Dijkstra",
                  command=self._run_dijkstra,
                  bg=ACCENT, fg="white", font=("Segoe UI", 9, "bold"),
                  relief=tk.FLAT, padx=10, pady=4,
                  cursor="hand2").pack(anchor="w", padx=12, pady=6)

        cols = ("Target", "Distance", "Path")
        self._tv_dijk = ttk.Treeview(right, columns=cols,
                                     show="headings", height=20)
        widths = [80, 80, 210]
        for c, w in zip(cols, widths):
            self._tv_dijk.heading(c, text=c)
            self._tv_dijk.column(c, width=w)
        sb = ttk.Scrollbar(right, orient=tk.VERTICAL,
                           command=self._tv_dijk.yview)
        self._tv_dijk.configure(yscrollcommand=sb.set)
        self._tv_dijk.pack(fill=tk.BOTH, expand=True, padx=12, pady=4)
        sb.pack(side=tk.RIGHT, fill=tk.Y)

    # ─────────────── TAB: KNAPSACK ───────────────────────────
    def _build_knapsack_tab(self):
        pw = tk.PanedWindow(self._tab_knapsack, orient=tk.HORIZONTAL,
                            bg=BG, sashwidth=5)
        pw.pack(fill=tk.BOTH, expand=True)

        left = tk.Frame(pw, bg=PANEL)
        pw.add(left, width=400)

        tk.Label(left, text="Pending Orders",
                 font=("Segoe UI", 11, "bold"), bg=PANEL, fg=ACCENT).pack(
                     pady=(12, 4), padx=12, anchor="w")

        cols = ("ID", "Name", "Weight", "Value", "Priority")
        self._tv_orders = ttk.Treeview(left, columns=cols,
                                       show="headings", height=18)
        for c, w in zip(cols, [40, 100, 75, 75, 70]):
            self._tv_orders.heading(c, text=c)
            self._tv_orders.column(c, width=w)
        self._tv_orders.pack(fill=tk.BOTH, expand=True, padx=12, pady=4)

        tk.Button(left, text="Run 0/1 Knapsack DP",
                  command=self._run_knapsack,
                  bg=GREEN, fg=BG, font=("Segoe UI", 9, "bold"),
                  relief=tk.FLAT, padx=10, pady=4,
                  cursor="hand2").pack(pady=8)

        self._lbl_ks = tk.Label(left, text="", bg=PANEL, fg=ORANGE,
                                font=("Segoe UI", 9, "bold"), wraplength=360,
                                justify=tk.LEFT)
        self._lbl_ks.pack(padx=12, anchor="w")

        right = tk.Frame(pw, bg=BG)
        pw.add(right, width=820)

        tk.Label(right, text="Knapsack DP — Item Selection Visualisation",
                 font=("Segoe UI", 11, "bold"), bg=BG, fg=ACCENT).pack(
                     pady=(12, 0), padx=12, anchor="w")

        self._fig_ks, self._ax_ks = plt.subplots(figsize=(8, 5.5), facecolor=BG)
        self._ax_ks.set_facecolor(BG)
        self._cv_ks = FigureCanvasTkAgg(self._fig_ks, master=right)
        self._cv_ks.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    # ─────────────── TAB: MST ────────────────────────────────
    def _build_mst_tab(self):
        pw = tk.PanedWindow(self._tab_mst, orient=tk.HORIZONTAL,
                            bg=BG, sashwidth=5)
        pw.pack(fill=tk.BOTH, expand=True)

        for side, title, color, attr_fig, attr_ax, attr_cv, attr_lbl in [
            ("left",  "Prim's MST  (greedy, priority-queue)",
             GREEN,  "_fig_prim", "_ax_prim", "_cv_prim", "_lbl_prim"),
            ("right", "Kruskal's MST  (sort + Union-Find)",
             RED,    "_fig_krus", "_ax_krus", "_cv_krus", "_lbl_krus"),
        ]:
            f = tk.Frame(pw, bg=BG)
            pw.add(f, width=600)

            tk.Label(f, text=title, font=("Segoe UI", 11, "bold"),
                     bg=BG, fg=color).pack(pady=(8, 0))

            fig, ax = plt.subplots(figsize=(5.5, 5), facecolor=BG)
            ax.set_facecolor(BG)
            cv = FigureCanvasTkAgg(fig, master=f)
            cv.get_tk_widget().pack(fill=tk.BOTH, expand=True)

            lbl = tk.Label(f, text="", bg=BG, fg=color,
                           font=("Segoe UI", 10, "bold"))
            lbl.pack(pady=4)

            setattr(self, attr_fig, fig)
            setattr(self, attr_ax,  ax)
            setattr(self, attr_cv,  cv)
            setattr(self, attr_lbl, lbl)

    # ─────────────── TAB: BIN PACKING ───────────────────────
    def _build_bin_tab(self):
        pw = tk.PanedWindow(self._tab_bin, orient=tk.HORIZONTAL,
                            bg=BG, sashwidth=5)
        pw.pack(fill=tk.BOTH, expand=True)

        left = tk.Frame(pw, bg=BG)
        pw.add(left, width=780)

        tk.Label(left, text="First-Fit Decreasing — Vehicle Loading Plan",
                 font=("Segoe UI", 11, "bold"), bg=BG, fg=ACCENT).pack(
                     pady=(8, 0))
        self._fig_bp, self._ax_bp = plt.subplots(figsize=(7.5, 5.5), facecolor=BG)
        self._ax_bp.set_facecolor(BG)
        self._cv_bp = FigureCanvasTkAgg(self._fig_bp, master=left)
        self._cv_bp.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        right = tk.Frame(pw, bg=PANEL)
        pw.add(right, width=420)

        tk.Label(right, text="Loading Summary",
                 font=("Segoe UI", 11, "bold"), bg=PANEL, fg=ACCENT).pack(
                     pady=(12, 4))

        cols = ("Truck", "Orders", "Weight", "Util %")
        self._tv_bp = ttk.Treeview(right, columns=cols,
                                   show="headings", height=18)
        for c, w in zip(cols, [60, 180, 90, 70]):
            self._tv_bp.heading(c, text=c)
            self._tv_bp.column(c, width=w)
        self._tv_bp.pack(fill=tk.BOTH, expand=True, padx=12, pady=4)

        self._lbl_bp = tk.Label(right, text="", bg=PANEL, fg=ORANGE,
                                font=("Segoe UI", 9, "bold"))
        self._lbl_bp.pack(pady=4)

    # ─────────────── TAB: COMPARISON ─────────────────────────
    def _build_compare_tab(self):
        f = tk.Frame(self._tab_compare, bg=BG)
        f.pack(fill=tk.BOTH, expand=True)

        tk.Label(f, text="Algorithm Performance Comparison",
                 font=("Segoe UI", 13, "bold"), bg=BG, fg=ACCENT).pack(pady=10)

        cols = ("Algorithm", "Category", "Input Size",
                "Time (ms)", "Optimal / Exact", "Greedy / Approx",
                "Approx Ratio %", "Complexity")
        self._tv_cmp = ttk.Treeview(f, columns=cols, show="headings", height=8)
        widths = [170, 140, 110, 90, 140, 140, 110, 150]
        for c, w in zip(cols, widths):
            self._tv_cmp.heading(c, text=c)
            self._tv_cmp.column(c, width=w)
        self._tv_cmp.pack(fill=tk.X, padx=20, pady=(0, 6))

        self._fig_cmp, self._axes_cmp = plt.subplots(
            1, 2, figsize=(11, 4), facecolor=BG)
        for ax in self._axes_cmp:
            ax.set_facecolor(PANEL)
        self._cv_cmp = FigureCanvasTkAgg(self._fig_cmp, master=f)
        self._cv_cmp.get_tk_widget().pack(fill=tk.BOTH, expand=True,
                                          padx=20, pady=4)

    # ─────────────── TRACE PANEL ─────────────────────────────
    def _build_trace_panel(self):
        outer = tk.Frame(self.root, bg=PANEL, height=115)
        outer.pack(fill=tk.X, padx=6, pady=4)
        outer.pack_propagate(False)

        tk.Label(outer, text="  Step-by-Step Trace  ▸",
                 bg=PANEL, fg=ACCENT,
                 font=("Segoe UI", 9, "bold")).pack(anchor="w", padx=8, pady=(4, 0))

        self._trace = scrolledtext.ScrolledText(
            outer, height=4,
            bg="#0f1120", fg="#78ffd6",
            font=("Consolas", 9),
            insertbackground="white",
            relief=tk.FLAT)
        self._trace.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 6))

    # ─────────────── RANDOM SCENARIO ─────────────────────────
    def _randomize(self):
        n_nodes = self._v_nodes.get()
        n_edges = self._v_edges.get()

        self.G          = nx.DiGraph()
        self.node_types = {}

        n_sup = max(1, n_nodes // 5)
        n_wh  = max(1, n_nodes // 4)
        n_cus = n_nodes - n_sup - n_wh

        nodes = (
            [f"S{i+1}" for i in range(n_sup)] +
            [f"W{i+1}" for i in range(n_wh)] +
            [f"C{i+1}" for i in range(n_cus)]
        )
        types = (
            ["supplier"]  * n_sup +
            ["warehouse"] * n_wh  +
            ["customer"]  * n_cus
        )

        for name, ntype in zip(nodes, types):
            self.G.add_node(name)
            self.node_types[name] = ntype

        # guarantee connectivity via a random spanning path
        shuffled = nodes[:]
        random.shuffle(shuffled)
        for i in range(len(shuffled) - 1):
            w = random.randint(5, 50)
            self.G.add_edge(shuffled[i], shuffled[i + 1], weight=w)

        extra = n_edges - (len(shuffled) - 1)
        for _ in range(max(0, extra)):
            u, v = random.sample(nodes, 2)
            if not self.G.has_edge(u, v):
                self.G.add_edge(u, v, weight=random.randint(5, 50))

        self.pos = nx.spring_layout(self.G, seed=42, k=2.0)

        # orders
        customers = [n for n, t in self.node_types.items() if t == "customer"]
        n_orders  = max(6, n_cus + 2)
        self.orders = []
        for i in range(n_orders):
            self.orders.append(Order(
                id=i + 1,
                name=f"ORD-{100 + i}",
                weight=round(random.uniform(5, 28), 1),
                value=round(random.uniform(60, 480), 1),
                priority=random.choice(["High", "Medium", "Low"]),
                destination=random.choice(customers) if customers else "C1",
            ))

        all_nodes = list(self.G.nodes())
        self._cb_src["values"] = all_nodes
        if all_nodes:
            self._v_src.set(all_nodes[0])

        self._refresh_orders_table()
        self._draw_graph()
        self._run_all()

    # ─────────────── RUN ALL ─────────────────────────────────
    def _run_all(self):
        self._perf_times    = {}
        self._approx_ratios = {}
        self._run_dijkstra()
        self._run_knapsack()
        self._run_mst()
        self._run_binpack()
        self._refresh_compare()

    # ─────────────── DIJKSTRA ────────────────────────────────
    def _run_dijkstra(self):
        if not self.G.nodes():
            return

        source = self._v_src.get()
        if source not in self.G.nodes():
            source = list(self.G.nodes())[0]
            self._v_src.set(source)

        adj   = defaultdict(dict)
        nodes = list(self.G.nodes())
        for u, v, d in self.G.edges(data=True):
            adj[u][v] = d.get("weight", 1)

        t0 = time.perf_counter()
        dist, prev, steps = Algorithms.dijkstra(dict(adj), source, nodes)
        self._perf_times["Dijkstra"] = (time.perf_counter() - t0) * 1000

        self.res_dijkstra = {"dist": dist, "prev": prev,
                             "steps": steps, "source": source}

        # populate table
        for row in self._tv_dijk.get_children():
            self._tv_dijk.delete(row)
        for n in sorted(nodes):
            if n == source:
                continue
            d    = dist[n]
            path = Algorithms.reconstruct_path(prev, n)
            ps   = " → ".join(path) if d != float("inf") else "No path"
            ds   = f"{d:.1f}" if d != float("inf") else "∞"
            self._tv_dijk.insert("", "end", values=(n, ds, ps))

        # highlight longest reachable path
        reachable = [(d, n) for n, d in dist.items()
                     if d != float("inf") and n != source]
        if reachable:
            reachable.sort(reverse=True)
            path  = Algorithms.reconstruct_path(prev, reachable[0][1])
            h_edges = {(path[i], path[i + 1]) for i in range(len(path) - 1)}
            self._draw_graph(highlight_path=h_edges)
        else:
            self._draw_graph()

    # ─────────────── KNAPSACK ────────────────────────────────
    def _run_knapsack(self):
        if not self.orders:
            return

        ws  = [o.weight for o in self.orders]
        vs  = [o.value  for o in self.orders]
        cap = self._v_cap.get()

        t0 = time.perf_counter()
        max_val, sel, dp, steps = Algorithms.knapsack_01(ws, vs, cap)
        self._perf_times["Knapsack DP"] = (time.perf_counter() - t0) * 1000

        self.res_knapsack = {"max_val": max_val, "selected": sel,
                             "dp": dp, "steps": steps}

        tw = sum(self.orders[i].weight for i in sel)
        self._lbl_ks.config(
            text=f"Max Value: ${max_val:.1f}  |  "
                 f"Selected: {len(sel)}/{len(self.orders)} orders  |  "
                 f"Weight used: {tw:.1f}/{cap:.0f} kg")
        self._draw_knapsack(sel, max_val)

    # ─────────────── MST ─────────────────────────────────────
    def _run_mst(self):
        nodes = list(self.G.nodes())
        und   = self.G.to_undirected()
        edges = [(u, v, d["weight"]) for u, v, d in und.edges(data=True)]

        t0 = time.perf_counter()
        pe, pc, ps = Algorithms.prims_mst(nodes, edges)
        self._perf_times["Prim's"] = (time.perf_counter() - t0) * 1000
        self.res_prim = {"edges": pe, "cost": pc, "steps": ps}

        t0 = time.perf_counter()
        ke, kc, ks = Algorithms.kruskals_mst(nodes, edges)
        self._perf_times["Kruskal's"] = (time.perf_counter() - t0) * 1000
        self.res_kruskal = {"edges": ke, "cost": kc, "steps": ks}

        self._draw_mst(self._ax_prim, self._cv_prim, pe,
                       "Prim's MST", GREEN, self._lbl_prim)
        self._draw_mst(self._ax_krus, self._cv_krus, ke,
                       "Kruskal's MST", RED, self._lbl_krus)

    # ─────────────── BIN PACKING ─────────────────────────────
    def _run_binpack(self):
        if not self.orders:
            return

        cap   = self._v_cap.get()
        items = [(o.id, o.weight) for o in self.orders]

        t0 = time.perf_counter()
        bins, steps = Algorithms.first_fit_decreasing(items, cap)
        self._perf_times["Bin Packing"] = (time.perf_counter() - t0) * 1000
        self.res_binpack = {"bins": bins, "steps": steps, "cap": cap}

        total_w    = sum(o.weight for o in self.orders)
        lower_bnd  = int(np.ceil(total_w / cap))
        ratio      = (lower_bnd / max(len(bins), 1)) * 100
        self._approx_ratios["bin"] = ratio

        for row in self._tv_bp.get_children():
            self._tv_bp.delete(row)
        for i, (bi, used) in enumerate(bins):
            ids  = ", ".join([f"#{iid}" for iid, _ in bi])
            util = (used / cap) * 100
            self._tv_bp.insert("", "end",
                               values=(f"Truck {i+1}", ids,
                                       f"{used:.1f} kg", f"{util:.1f}%"))
        self._lbl_bp.config(
            text=f"Trucks used: {len(bins)}  |  Lower bound: {lower_bnd}  |  "
                 f"Efficiency: {ratio:.1f}%\nTotal cargo: {total_w:.1f} kg")
        self._draw_binpack(bins, cap)

    # ─────────────── DRAWINGS ────────────────────────────────
    def _mpl_style(self, ax):
        ax.set_facecolor(BG)
        ax.tick_params(colors=MUTED)
        for sp in ax.spines.values():
            sp.set_color(EDGE_DEF)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    def _draw_graph(self, highlight_path=None, highlight_edges=None):
        ax = self._ax_g
        ax.clear()
        ax.set_facecolor(BG)

        if not self.G.nodes():
            self._cv_g.draw()
            return

        nc = [NODE_COLORS.get(self.node_types.get(n, "customer"), MUTED)
              for n in self.G.nodes()]

        ec, ew = [], []
        for u, v in self.G.edges():
            if highlight_path and (u, v) in highlight_path:
                ec.append(RED);    ew.append(3.0)
            elif highlight_edges and (u, v) in highlight_edges:
                ec.append(ORANGE); ew.append(2.5)
            else:
                ec.append(EDGE_DEF); ew.append(1.0)

        nx.draw_networkx_nodes(self.G, self.pos, ax=ax,
                               node_color=nc, node_size=650, alpha=0.95)
        nx.draw_networkx_labels(self.G, self.pos, ax=ax,
                                font_color="white", font_size=8,
                                font_weight="bold")
        nx.draw_networkx_edges(self.G, self.pos, ax=ax,
                               edge_color=ec, width=ew,
                               arrows=True, arrowsize=14, alpha=0.85,
                               connectionstyle="arc3,rad=0.1")
        elabels = nx.get_edge_attributes(self.G, "weight")
        nx.draw_networkx_edge_labels(self.G, self.pos, ax=ax,
                                     edge_labels=elabels,
                                     font_color=MUTED, font_size=7)
        patches = [mpatches.Patch(color=c, label=t.capitalize())
                   for t, c in NODE_COLORS.items()]
        ax.legend(handles=patches, loc="upper left",
                  facecolor=PANEL, edgecolor="none",
                  labelcolor=TEXT, fontsize=8)
        ax.axis("off")
        self._fig_g.tight_layout(pad=0.5)
        self._cv_g.draw()

    def _draw_mst(self, ax, cv, mst_edges, title, color, lbl_widget):
        ax.clear()
        ax.set_facecolor(BG)
        und = self.G.to_undirected()
        pos = {n: self.pos[n] for n in und.nodes() if n in self.pos}

        mst_set = ({(u, v) for u, v, _ in mst_edges} |
                   {(v, u) for u, v, _ in mst_edges})
        ec  = [color  if (u, v) in mst_set else "#2d3055"
               for u, v in und.edges()]
        ew  = [2.5    if (u, v) in mst_set else 0.7
               for u, v in und.edges()]
        nc  = [NODE_COLORS.get(self.node_types.get(n, "customer"), MUTED)
               for n in und.nodes()]

        nx.draw_networkx_nodes(und, pos, ax=ax, node_color=nc, node_size=550)
        nx.draw_networkx_labels(und, pos, ax=ax,
                                font_color="white", font_size=8, font_weight="bold")
        nx.draw_networkx_edges(und, pos, ax=ax,
                               edge_color=ec, width=ew, alpha=0.85)
        mst_labels = {(u, v): f"{w:.0f}" for u, v, w in mst_edges}
        nx.draw_networkx_edge_labels(und, pos, ax=ax,
                                     edge_labels=mst_labels,
                                     font_color=color, font_size=7)
        ax.axis("off")
        ax.set_title(title, color=color, fontsize=9, fontweight="bold", pad=4)
        cv.draw()

        total = sum(w for _, _, w in mst_edges)
        lbl_widget.config(text=f"Total MST Cost: {total:.1f}")

    def _draw_knapsack(self, selected, max_val):
        ax = self._ax_ks
        ax.clear()
        ax.set_facecolor(BG)
        orders = self.orders
        if not orders:
            self._cv_ks.draw()
            return

        x      = np.arange(len(orders))
        vals   = [o.value for o in orders]
        wts    = [o.weight for o in orders]
        colors = [GREEN if i in selected else RED for i in range(len(orders))]

        bars = ax.bar(x, vals, color=colors, alpha=0.85,
                      edgecolor="white", linewidth=0.4)
        for bar, w in zip(bars, wts):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 3,
                    f"W:{w:.0f}",
                    ha="center", va="bottom", color=MUTED, fontsize=7)

        ax.set_xticks(x)
        ax.set_xticklabels([f"#{o.id}" for o in orders],
                           rotation=40, ha="right", color=MUTED, fontsize=8)
        ax.set_ylabel("Value ($)", color=MUTED)
        ax.set_title(
            f"Knapsack Selection  —  Total Value: ${max_val:.1f}  "
            f"(green=selected, red=excluded)",
            color=TEXT, fontsize=9)
        self._mpl_style(ax)
        ax.legend(
            handles=[mpatches.Patch(color=GREEN, label="Selected"),
                     mpatches.Patch(color=RED,   label="Excluded")],
            facecolor=PANEL, edgecolor="none", labelcolor=TEXT, fontsize=8)
        self._fig_ks.tight_layout(pad=0.5)
        self._cv_ks.draw()

    def _draw_binpack(self, bins, cap):
        ax = self._ax_bp
        ax.clear()
        ax.set_facecolor(BG)
        if not bins:
            self._cv_bp.draw()
            return

        palette = plt.cm.tab20(np.linspace(0, 1, 20))
        for ti, (bi, used) in enumerate(bins):
            y = 0
            for iid, sz in bi:
                ax.bar(ti, sz, bottom=y,
                       color=palette[iid % 20], alpha=0.85,
                       edgecolor=BG, linewidth=0.8)
                if sz > 2:
                    ax.text(ti, y + sz / 2,
                            f"#{iid}\n{sz:.0f}",
                            ha="center", va="center",
                            color="white", fontsize=7, fontweight="bold")
                y += sz
        ax.axhline(cap, color=RED, linestyle="--",
                   linewidth=1.2, alpha=0.75, label=f"Capacity {cap:.0f} kg")
        ax.set_xticks(range(len(bins)))
        ax.set_xticklabels([f"Truck {i+1}" for i in range(len(bins))],
                           color=MUTED, fontsize=8)
        ax.set_ylabel("Weight (kg)", color=MUTED)
        ax.set_ylim(0, cap * 1.15)
        ax.set_title("Bin Packing — First Fit Decreasing",
                     color=TEXT, fontsize=9)
        self._mpl_style(ax)
        ax.legend(facecolor=PANEL, labelcolor=TEXT, fontsize=8,
                  edgecolor="none")
        self._fig_bp.tight_layout(pad=0.5)
        self._cv_bp.draw()

    def _refresh_compare(self):
        for row in self._tv_cmp.get_children():
            self._tv_cmp.delete(row)

        nv = self.G.number_of_nodes()
        ne = self.G.number_of_edges()
        no = len(self.orders)
        cap = self._v_cap.get()

        ks = self.res_knapsack
        pr = self.res_prim
        kr = self.res_kruskal
        bp = self.res_binpack

        cap_int = int(cap)
        total_w = sum(o.weight for o in self.orders)
        lb      = int(np.ceil(total_w / cap)) if cap > 0 else "—"
        bp_t    = len(bp["bins"]) if bp else "—"
        bp_ratio = self._approx_ratios.get("bin", 100)

        rows = [
            ("Dijkstra's",    "Shortest Path",   f"V={nv}, E={ne}",
             f"{self._perf_times.get('Dijkstra', 0):.4f}",
             "Optimal",  "N/A", "100%", "O((V+E) log V)"),
            ("0/1 Knapsack",  "Dynamic Program.", f"n={no}, W={cap_int}",
             f"{self._perf_times.get('Knapsack DP', 0):.4f}",
             f"${ks['max_val']:.1f}" if ks else "—", "Exact", "100%",
             "O(n · W)"),
            ("Prim's MST",    "Greedy + PQ",     f"V={nv}, E={ne}",
             f"{self._perf_times.get('Prim\'s', 0):.4f}",
             f"{pr['cost']:.1f}" if pr else "—", "Optimal", "100%",
             "O(E log V)"),
            ("Kruskal's MST", "Greedy + UF",     f"V={nv}, E={ne}",
             f"{self._perf_times.get('Kruskal\'s', 0):.4f}",
             f"{kr['cost']:.1f}" if kr else "—", "Optimal", "100%",
             "O(E log E)"),
            ("Bin Packing FFD", "Greedy Approx.", f"n={no}",
             f"{self._perf_times.get('Bin Packing', 0):.4f}",
             f"{lb} trucks (bound)", f"{bp_t} trucks",
             f"{bp_ratio:.1f}%", "O(n log n)"),
        ]
        for row in rows:
            self._tv_cmp.insert("", "end", values=row)

        # charts
        for ax in self._axes_cmp:
            ax.clear()
            ax.set_facecolor(PANEL)

        names  = ["Dijkstra", "Knapsack", "Prim's", "Kruskal's", "Bin Pack"]
        keys   = ["Dijkstra", "Knapsack DP", "Prim's", "Kruskal's", "Bin Packing"]
        times  = [self._perf_times.get(k, 0) for k in keys]
        colors = [ACCENT, GREEN, ORANGE, RED, PURPLE]

        b = self._axes_cmp[0].bar(names, times, color=colors, alpha=0.85)
        for bar, t in zip(b, times):
            self._axes_cmp[0].text(
                bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{t:.4f}", ha="center", va="bottom",
                color=TEXT, fontsize=7)
        self._axes_cmp[0].set_title("Execution Time (ms)",
                                    color=TEXT, fontsize=9)
        self._axes_cmp[0].tick_params(colors=MUTED)
        for sp in self._axes_cmp[0].spines.values():
            sp.set_color(EDGE_DEF)

        # Approximation ratio (only bin packing has a meaningful ratio here)
        self._axes_cmp[1].bar(["Bin Packing\n(FFD)"], [bp_ratio],
                              color=PURPLE, alpha=0.85)
        self._axes_cmp[1].axhline(100, color=GREEN, linestyle="--",
                                  linewidth=1.5, label="Optimal = 100%")
        self._axes_cmp[1].set_ylim(0, 150)
        self._axes_cmp[1].set_title("Approximation Ratio (%)",
                                    color=TEXT, fontsize=9)
        self._axes_cmp[1].tick_params(colors=MUTED)
        self._axes_cmp[1].legend(facecolor=PANEL, labelcolor=TEXT,
                                 edgecolor="none", fontsize=8)
        for sp in self._axes_cmp[1].spines.values():
            sp.set_color(EDGE_DEF)

        self._fig_cmp.tight_layout(pad=0.6)
        self._cv_cmp.draw()

    def _refresh_orders_table(self):
        for row in self._tv_orders.get_children():
            self._tv_orders.delete(row)
        for o in self.orders:
            self._tv_orders.insert("", "end",
                                   values=(o.id, o.name,
                                           f"{o.weight:.1f} kg",
                                           f"${o.value:.1f}",
                                           o.priority))

    # ─────────────── STEP-BY-STEP ────────────────────────────
    def _step_reset(self):
        algo = self._v_step_algo.get()
        self.step_idx = 0

        mapping = {
            "Dijkstra":    self.res_dijkstra.get("steps", []),
            "Knapsack":    self.res_knapsack.get("steps", []),
            "Prim's":      self.res_prim.get("steps",    []),
            "Kruskal's":   self.res_kruskal.get("steps", []),
            "Bin Packing": self.res_binpack.get("steps", []),
        }
        self.step_steps = mapping.get(algo, [])
        self._lbl_step.config(text=f"0 / {len(self.step_steps)}")
        self._trace.delete("1.0", tk.END)
        self._trace.insert(tk.END,
                           f"[{algo}] Reset — {len(self.step_steps)} steps total. "
                           "Press 'Step →' to begin.\n")

    def _step_forward(self):
        if not self.step_steps:
            self._step_reset()
        if self.step_idx >= len(self.step_steps):
            self._trace.insert(tk.END, "✅  Algorithm complete.\n")
            self._trace.see(tk.END)
            return

        step = self.step_steps[self.step_idx]
        self.step_idx += 1
        n_total = len(self.step_steps)
        self._lbl_step.config(text=f"{self.step_idx} / {n_total}")

        self._trace.insert(
            tk.END,
            f"[Step {self.step_idx:>3}]  {step.get('action', '')}\n")
        self._trace.see(tk.END)

        algo = self._v_step_algo.get()

        if algo == "Dijkstra":
            visited = step.get("visited", set())
            prev    = self.res_dijkstra.get("prev", {})
            src     = self.res_dijkstra.get("source", "")
            h = {(prev[v], v) for v in visited if prev.get(v)}
            self._draw_graph(highlight_edges=h)
            ds = step.get("distances", {})
            line = "         dist: " + "  ".join(
                f"{k}={('∞' if v == float('inf') else str(int(v)))}"
                for k, v in sorted(ds.items())) + "\n"
            self._trace.insert(tk.END, line)

        elif algo in ("Prim's", "Kruskal's"):
            mst_so_far = step.get("mst_edges", [])
            if algo == "Prim's":
                self._draw_mst(self._ax_prim, self._cv_prim, mst_so_far,
                               f"Prim's — step {self.step_idx}", GREEN, self._lbl_prim)
                self.nb.select(self._tab_mst)
            else:
                self._draw_mst(self._ax_krus, self._cv_krus, mst_so_far,
                               f"Kruskal's — step {self.step_idx}", RED, self._lbl_krus)
                self.nb.select(self._tab_mst)
                sets   = step.get("sets", {})
                unique = len(set(sets.values()))
                self._trace.insert(tk.END,
                                   f"         Components: {unique}\n")

        elif algo == "Bin Packing":
            bins_so_far = step.get("bins", [])
            self._draw_binpack(bins_so_far, self._v_cap.get())
            self.nb.select(self._tab_bin)

        self._trace.see(tk.END)


# ─────────────────────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────────────────────

def main():
    root = tk.Tk()
    app  = LogisticsApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()