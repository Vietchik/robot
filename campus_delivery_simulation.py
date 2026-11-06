#!/usr/bin/env python3
"""
Campus Delivery Robot Pathfinding Simulation & Benchmark Suite
Course: Introduction to Artificial Intelligence (TOKT11121)
Faculty of Data Science and Artificial Intelligence (FDA)
College of Technology (NCT) - National Economics University (NEU)
Instructor: Dr. Nguyen Trong Nghia (Business AI Lab)

Group 5 - Topic 3: Campus Delivery Robot (Canteen -> Classrooms)
Algorithms compared:
  1. Breadth-First Search (BFS) - Uninformed level-order search
  2. Greedy Best-First Search (GBFS) - Informed search with Manhattan heuristic h(n)
  3. A* Search - Optimal informed search with f(n) = g(n) + h(n)
"""

from collections import deque
import heapq
import json
import os
import random
import time
import tracemalloc
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

Coord = Tuple[int, int]
MOVES: List[Tuple[int, int, str]] = [
    (-1, 0, "Up"),
    (1, 0, "Down"),
    (0, -1, "Left"),
    (0, 1, "Right"),
]


def manhattan(a: Coord, b: Coord) -> int:
    """Compute Manhattan (L1) distance between two matrix cells."""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def in_bounds(r: int, c: int, rows: int, cols: int) -> bool:
    return 0 <= r < rows and 0 <= c < cols


def get_neighbors(grid: np.ndarray, pos: Coord) -> List[Coord]:
    rows, cols = grid.shape
    r, c = pos
    result = []
    for dr, dc, _ in MOVES:
        nr, nc = r + dr, c + dc
        if in_bounds(nr, nc, rows, cols) and grid[nr, nc] == 0:
            result.append((nr, nc))
    return result


def reconstruct_path(parent: Dict[Coord, Optional[Coord]], goal: Coord) -> List[Coord]:
    path = []
    curr: Optional[Coord] = goal
    while curr is not None:
        path.append(curr)
        curr = parent[curr]
    path.reverse()
    return path


# ==============================================================================
# 1. CORE SEARCH ALGORITHMS (BFS, GBFS-Manhattan, A*)
# ==============================================================================

def bfs_search(grid: np.ndarray, start: Coord, goal: Coord) -> Dict:
    """Breadth-First Search (FIFO Queue) guaranteeing shortest path on unweighted grids."""
    q = deque([start])
    visited = {start}
    parent: Dict[Coord, Optional[Coord]] = {start: None}
    expanded_order: List[Coord] = []
    trace_log: List[Dict] = []

    while q:
        curr = q.popleft()
        expanded_order.append(curr)
        if curr == goal:
            path = reconstruct_path(parent, goal)
            trace_log.append({
                "step": len(expanded_order),
                "u": curr,
                "neighbors": [],
                "open": list(q)[:6],
                "note": "GOAL REACHED",
            })
            return {
                "algorithm": "BFS",
                "path": path,
                "cost": len(path) - 1,
                "expanded_nodes": len(expanded_order),
                "expanded_set": expanded_order,
                "frontier_set": list(q),
                "trace": trace_log,
            }

        added_nbs = []
        for nb in get_neighbors(grid, curr):
            if nb not in visited:
                visited.add(nb)
                parent[nb] = curr
                q.append(nb)
                added_nbs.append(nb)

        if len(trace_log) < 12:
            trace_log.append({
                "step": len(expanded_order),
                "u": curr,
                "neighbors": added_nbs,
                "open": list(q)[:6],
                "note": "",
            })

    return {
        "algorithm": "BFS",
        "path": [],
        "cost": -1,
        "expanded_nodes": len(expanded_order),
        "expanded_set": expanded_order,
        "frontier_set": [],
        "trace": trace_log,
    }


def gbfs_search(grid: np.ndarray, start: Coord, goal: Coord) -> Dict:
    """Greedy Best-First Search (Min-Heap ordered purely by Manhattan h(n))."""
    counter = 0
    open_heap = [(manhattan(start, goal), counter, start)]
    visited = {start}
    parent: Dict[Coord, Optional[Coord]] = {start: None}
    expanded_order: List[Coord] = []
    trace_log: List[Dict] = []

    while open_heap:
        h_val, _, curr = heapq.heappop(open_heap)
        expanded_order.append(curr)

        if curr == goal:
            path = reconstruct_path(parent, goal)
            trace_log.append({
                "step": len(expanded_order),
                "u": curr,
                "h": h_val,
                "neighbors": [],
                "open": [( item[2], item[0] ) for item in sorted(open_heap)[:5]],
                "note": "GOAL REACHED",
            })
            return {
                "algorithm": "GBFS",
                "path": path,
                "cost": len(path) - 1,
                "expanded_nodes": len(expanded_order),
                "expanded_set": expanded_order,
                "frontier_set": [item[2] for item in open_heap],
                "trace": trace_log,
            }

        added_nbs = []
        for nb in get_neighbors(grid, curr):
            if nb not in visited:
                visited.add(nb)
                parent[nb] = curr
                counter += 1
                h_nb = manhattan(nb, goal)
                heapq.heappush(open_heap, (h_nb, counter, nb))
                added_nbs.append((nb, h_nb))

        if len(trace_log) < 12:
            trace_log.append({
                "step": len(expanded_order),
                "u": curr,
                "h": h_val,
                "neighbors": added_nbs,
                "open": [(item[2], item[0]) for item in sorted(open_heap)[:5]],
                "note": "",
            })

    return {
        "algorithm": "GBFS",
        "path": [],
        "cost": -1,
        "expanded_nodes": len(expanded_order),
        "expanded_set": expanded_order,
        "frontier_set": [],
        "trace": trace_log,
    }


def astar_search(grid: np.ndarray, start: Coord, goal: Coord) -> Dict:
    """A* Search (Min-Heap ordered by f(n) = g(n) + h(n) with Manhattan tie-breaking)."""
    counter = 0
    h0 = manhattan(start, goal)
    open_heap = [(h0, h0, counter, start)]
    g_score: Dict[Coord, int] = {start: 0}
    parent: Dict[Coord, Optional[Coord]] = {start: None}
    closed = set()
    expanded_order: List[Coord] = []
    trace_log: List[Dict] = []

    while open_heap:
        f_val, h_val, _, curr = heapq.heappop(open_heap)
        if curr in closed:
            continue
        closed.add(curr)
        expanded_order.append(curr)
        g_val = g_score[curr]

        if curr == goal:
            path = reconstruct_path(parent, goal)
            trace_log.append({
                "step": len(expanded_order),
                "u": curr,
                "g": g_val,
                "h": h_val,
                "f": f_val,
                "neighbors": [],
                "open": [(item[3], item[0]) for item in sorted(open_heap) if item[3] not in closed][:5],
                "note": "GOAL REACHED",
            })
            return {
                "algorithm": "A*",
                "path": path,
                "cost": len(path) - 1,
                "expanded_nodes": len(expanded_order),
                "expanded_set": expanded_order,
                "frontier_set": [item[3] for item in open_heap if item[3] not in closed],
                "trace": trace_log,
            }

        added_nbs = []
        for nb in get_neighbors(grid, curr):
            if nb in closed:
                continue
            tent_g = g_val + 1
            if tent_g < g_score.get(nb, 10**9):
                g_score[nb] = tent_g
                parent[nb] = curr
                h_nb = manhattan(nb, goal)
                f_nb = tent_g + h_nb
                counter += 1
                heapq.heappush(open_heap, (f_nb, h_nb, counter, nb))
                added_nbs.append((nb, tent_g, h_nb, f_nb))

        if len(trace_log) < 12:
            trace_log.append({
                "step": len(expanded_order),
                "u": curr,
                "g": g_val,
                "h": h_val,
                "f": f_val,
                "neighbors": added_nbs,
                "open": [(item[3], item[0]) for item in sorted(open_heap) if item[3] not in closed][:5],
                "note": "",
            })

    return {
        "algorithm": "A*",
        "path": [],
        "cost": -1,
        "expanded_nodes": len(expanded_order),
        "expanded_set": expanded_order,
        "frontier_set": [],
        "trace": trace_log,
    }


# ==============================================================================
# 2. TIER 1: 5x5 MINI-MATRIX FOR STEP-BY-STEP TRACE VERIFICATION
# ==============================================================================

def build_5x5_demo_grid() -> Tuple[np.ndarray, Coord, Coord]:
    grid = np.zeros((5, 5), dtype=int)
    # Obstacles creating a classic heuristic local trap vs optimal corridor
    for r, c in [(0, 3), (1, 1), (1, 3), (2, 3), (3, 1)]:
        grid[r, c] = 1
    start = (0, 0)  # Canteen
    goal = (0, 4)   # Classroom L1 (behind vertical wall at col 3, row 0..2)
    return grid, start, goal


# ==============================================================================
# 3. TIER 2: 15x15 NEU CAMPUS BUILDING & COURTYARD MATRIX (CANTEEN -> 4 CLASSROOMS)
# ==============================================================================

def build_campus_15x15_grid() -> Tuple[np.ndarray, Coord, Dict[str, Coord]]:
    grid = np.zeros((15, 15), dtype=int)
    # Building blocks & courtyard walls (1 = obstacle)
    obstacles = [
        # Upper-left pillar block
        (1, 4), (2, 4), (3, 4), (4, 4),
        # Upper-mid lecture block with U-trap facing Canteen
        (2, 7), (2, 8), (2, 9), (2, 10),
        (3, 10), (4, 10), (5, 10), (5, 8), (5, 9),
        # Central courtyard garden & fountain
        (6, 3), (6, 4), (6, 5), (7, 5), (8, 5),
        (7, 7), (7, 8), (8, 7), (8, 8),
        # South wing U-shaped corridor trap (tricks Greedy Best-First Search)
        (9, 2), (10, 2), (10, 3), (10, 4), (11, 4), (12, 4),
        # East wing lab partitions
        (7, 11), (8, 11), (9, 11), (10, 11),
        (11, 8), (11, 9), (12, 9), (13, 9),
        (4, 13), (5, 13),
    ]
    for r, c in obstacles:
        grid[r, c] = 1

    canteen = (1, 1)
    classrooms = {
        "C101 (North Hall)": (1, 12),
        "C202 (South Wing)": (12, 2),
        "Lab-AI (East Tech)": (8, 13),
        "C305 (Auditorium)": (13, 13),
    }
    return grid, canteen, classrooms


# ==============================================================================
# 4. TIER 3: MULTI-SCALE 100-RUN BENCHMARK SUITE (10x10, 25x25, 50x50, 100x100)
# ==============================================================================

def generate_random_campus_grid(dim: int, obstacle_prob: float, seed: int) -> Tuple[np.ndarray, Coord, Coord]:
    rng = random.Random(seed)
    while True:
        grid = np.zeros((dim, dim), dtype=int)
        for r in range(dim):
            for c in range(dim):
                if rng.random() < obstacle_prob:
                    grid[r, c] = 1
        start = (0, 0)
        goal = (dim - 1, dim - 1)
        grid[start] = 0
        grid[goal] = 0
        # Ensure valid path exists via quick BFS check
        res = bfs_search(grid, start, goal)
        if res["cost"] > 0:
            return grid, start, goal


def run_scalability_benchmarks(runs_per_scale: int = 50) -> Dict:
    scales = [
        ("10x10", 10, 0.22),
        ("25x25", 25, 0.25),
        ("50x50", 50, 0.26),
        ("100x100", 100, 0.26),
    ]
    summary = {}
    for label, dim, obs_p in scales:
        stats = {
            alg: {"cost": [], "expanded": [], "time_ms": [], "mem_kb": [], "optimal_hits": 0}
            for alg in ("BFS", "GBFS", "A*")
        }
        for i in range(runs_per_scale):
            grid, start, goal = generate_random_campus_grid(dim, obs_p, seed=1000 * dim + i)
            opt_cost = None
            for alg, fn in (("BFS", bfs_search), ("GBFS", gbfs_search), ("A*", astar_search)):
                tracemalloc.start()
                t0 = time.perf_counter()
                res = fn(grid, start, goal)
                dt_ms = (time.perf_counter() - t0) * 1000.0
                _, peak_bytes = tracemalloc.get_traced_memory()
                tracemalloc.stop()

                if alg == "BFS":
                    opt_cost = res["cost"]
                stats[alg]["cost"].append(res["cost"])
                stats[alg]["expanded"].append(res["expanded_nodes"])
                stats[alg]["time_ms"].append(dt_ms)
                stats[alg]["mem_kb"].append(peak_bytes / 1024.0)
                if res["cost"] == opt_cost:
                    stats[alg]["optimal_hits"] += 1

        summary[label] = {}
        for alg in ("BFS", "GBFS", "A*"):
            s = stats[alg]
            summary[label][alg] = {
                "avg_cost": round(float(np.mean(s["cost"])), 2),
                "avg_expanded": round(float(np.mean(s["expanded"])), 1),
                "avg_time_ms": round(float(np.mean(s["time_ms"])), 3),
                "avg_mem_kb": round(float(np.mean(s["mem_kb"])), 2),
                "optimality_pct": round(100.0 * s["optimal_hits"] / runs_per_scale, 1),
            }
    return summary


# ==============================================================================
# 5. HIGH-RESOLUTION VISUALIZATION GENERATORS FOR SLIDES
# ==============================================================================

def plot_5x5_trace_cards(out_dir: str):
    """Render 3 equal-sized square inset visuals (1200x1200 px) for the 5x5 trace slide."""
    grid, start, goal = build_5x5_demo_grid()
    alg_List = [
        ("BFS", bfs_search(grid, start, goal), "#2563EB"),
        ("GBFS", gbfs_search(grid, start, goal), "#D97706"),
        ("A*", astar_search(grid, start, goal), "#059669"),
    ]
    saved = []
    for name, res, color in alg_List:
        fig, ax = plt.subplots(figsize=(4.5, 4.5), dpi=300)
        fig.patch.set_facecolor("#F8FAFC")
        ax.set_facecolor("#FFFFFF")

        rows, cols = grid.shape
        # Draw cells
        for r in range(rows):
            for c in range(cols):
                y = rows - 1 - r
                if grid[r, c] == 1:
                    face = "#1E293B"
                elif (r, c) in res["expanded_set"]:
                    face = "#DBEAFE" if name == "BFS" else ("#FEF3C7" if name == "GBFS" else "#D1FAE5")
                else:
                    face = "#FFFFFF"
                rect = plt.Rectangle((c, y), 1, 1, facecolor=face, edgecolor="#CBD5E1", linewidth=1.5)
                ax.add_patch(rect)
                if grid[r, c] == 0:
                    h_v = manhattan((r, c), goal)
                    ax.text(c + 0.12, y + 0.15, f"h={h_v}", fontsize=8, color="#64748B", fontweight="bold")
                    if (r, c) in res["expanded_set"]:
                        order_idx = res["expanded_set"].index((r, c)) + 1
                        ax.text(c + 0.82, y + 0.82, f"#{order_idx}", fontsize=8, color="#334155",
                                ha="right", va="top", fontweight="bold")

        # Draw path
        path = res["path"]
        if path:
            px = [c + 0.5 for r, c in path]
            py = [rows - 1 - r + 0.5 for r, c in path]
            ax.plot(px, py, color=color, linewidth=4.0, marker="o", markersize=8, zorder=5)

        # Mark Start and Goal
        sr, sc = start
        gr, gc = goal
        ax.scatter([sc + 0.5], [rows - 1 - sr + 0.5], s=260, c="#DC2626", zorder=6, edgecolors="white", linewidth=2)
        ax.text(sc + 0.5, rows - 1 - sr + 0.5, "C", color="white", ha="center", va="center", fontsize=10, fontweight="bold", zorder=7)
        ax.scatter([gc + 0.5], [rows - 1 - gr + 0.5], s=260, c="#16A34A", zorder=6, edgecolors="white", linewidth=2)
        ax.text(gc + 0.5, rows - 1 - gr + 0.5, "L1", color="white", ha="center", va="center", fontsize=9, fontweight="bold", zorder=7)

        ax.set_xlim(0, cols)
        ax.set_ylim(0, rows)
        ax.set_xticks([i + 0.5 for i in range(cols)])
        ax.set_xticklabels([f"Col {i}" for i in range(cols)], fontsize=8.5, fontweight="bold", color="#334155")
        ax.set_yticks([i + 0.5 for i in range(rows)])
        ax.set_yticklabels([f"Row {rows - 1 - i}" for i in range(rows)], fontsize=8.5, fontweight="bold", color="#334155")
        ax.set_title(f"{name}: Cost = {res['cost']} steps | Expanded = {res['expanded_nodes']} nodes",
                     fontsize=10.5, fontweight="bold", color="#0F172A", pad=10)
        plt.tight_layout()
        fname = os.path.join(out_dir, f"trace_5x5_{name.lower().replace('*', 'star')}.png")
        plt.savefig(fname, dpi=300)
        plt.close(fig)
        saved.append(fname)
    return saved


def plot_campus_overview_bleed(out_dir: str) -> str:
    """Render a right-half split-bleed image (6.666 x 7.5 in) of the 15x15 Campus Matrix Map."""
    grid, canteen, classrooms = build_campus_15x15_grid()
    fig, ax = plt.subplots(figsize=(6.666, 7.5), dpi=300)
    fig.patch.set_facecolor("#0F172A")
    ax.set_facecolor("#1E293B")

    rows, cols = grid.shape
    for r in range(rows):
        for c in range(cols):
            y = rows - 1 - r
            if grid[r, c] == 1:
                face = "#334155"
                edge = "#475569"
            else:
                face = "#F8FAFC"
                edge = "#E2E8F0"
            rect = plt.Rectangle((c, y), 1, 1, facecolor=face, edgecolor=edge, linewidth=1.0)
            ax.add_patch(rect)
            if grid[r, c] == 1:
                ax.text(c + 0.5, y + 0.5, "1", color="#94A3B8", ha="center", va="center", fontsize=7.5, fontweight="bold")
            else:
                ax.text(c + 0.5, y + 0.5, "0", color="#CBD5E1", ha="center", va="center", fontsize=6.5)

    # Plot A* routes from Canteen to all 4 classrooms
    route_colors = ["#2563EB", "#D97706", "#059669", "#7C3AED"]
    for (label, goal), col in zip(classrooms.items(), route_colors):
        res = astar_search(grid, canteen, goal)
        path = res["path"]
        px = [c + 0.5 for r, c in path]
        py = [rows - 1 - r + 0.5 for r, c in path]
        ax.plot(px, py, color=col, linewidth=3.0, alpha=0.88, label=f"{label.split()[0]} ({res['cost']} steps)", zorder=4)
        gr, gc = goal
        ax.scatter([gc + 0.5], [rows - 1 - gr + 0.5], s=260, c=col, edgecolors="white", linewidth=2, zorder=6)
        ax.text(gc + 0.5, rows - 1 - gr + 0.5, label.split()[0][:4], color="white",
                ha="center", va="center", fontsize=6.5, fontweight="bold", zorder=7)

    sr, sc = canteen
    ax.scatter([sc + 0.5], [rows - 1 - sr + 0.5], s=340, c="#DC2626", edgecolors="white", linewidth=2.5, zorder=8)
    ax.text(sc + 0.5, rows - 1 - sr + 0.5, "CAN", color="white", ha="center", va="center", fontsize=7.5, fontweight="bold", zorder=9)

    ax.set_xlim(0, cols)
    ax.set_ylim(0, rows)
    ax.set_aspect("equal")
    ax.set_xticks([i + 0.5 for i in range(cols)])
    ax.set_xticklabels([str(i) for i in range(cols)], color="#CBD5E1", fontsize=8)
    ax.set_yticks([i + 0.5 for i in range(rows)])
    ax.set_yticklabels([str(rows - 1 - i) for i in range(rows)], color="#CBD5E1", fontsize=8)
    ax.set_title("NEU Campus 15×15 Matrix Map\n(Canteen [1,1] → 4 Target Classrooms)",
                 color="#F8FAFC", fontsize=13, fontweight="bold", pad=14)
    leg = ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.05), ncol=2, frameon=True,
                    facecolor="#1E293B", edgecolor="#475569", fontsize=8.5, labelcolor="#F8FAFC")
    plt.tight_layout()
    out_path = os.path.join(out_dir, "bleed_campus_matrix_map.png")
    plt.savefig(out_path, dpi=300, facecolor=fig.get_facecolor())
    plt.close(fig)
    return out_path


def plot_algorithm_comparison_bleed(out_dir: str, classroom_key: str, filename: str) -> str:
    """Render a right-half split-bleed image (6.666 x 7.5 in) comparing BFS, GBFS, and A* on the 15x15 matrix."""
    grid, canteen, classrooms = build_campus_15x15_grid()
    goal = classrooms[classroom_key]
    res_bfs = bfs_search(grid, canteen, goal)
    res_gbfs = gbfs_search(grid, canteen, goal)
    res_astar = astar_search(grid, canteen, goal)

    fig, axes = plt.subplots(3, 1, figsize=(6.666, 7.5), dpi=300)
    fig.patch.set_facecolor("#0F172A")

    configs = [
        (axes[0], "BFS (Uninformed)", res_bfs, "#3B82F6", "#DBEAFE"),
        (axes[1], "GBFS (Manhattan Heuristic)", res_gbfs, "#F59E0B", "#FEF3C7"),
        (axes[2], "A* Search (f = g + h)", res_astar, "#10B981", "#D1FAE5"),
    ]
    rows, cols = grid.shape
    for ax, title, res, line_col, shade_col in configs:
        ax.set_facecolor("#1E293B")
        for r in range(rows):
            for c in range(cols):
                y = rows - 1 - r
                if grid[r, c] == 1:
                    face = "#334155"
                elif (r, c) in res["expanded_set"]:
                    face = shade_col
                else:
                    face = "#F8FAFC"
                rect = plt.Rectangle((c, y), 1, 1, facecolor=face, edgecolor="#CBD5E1", linewidth=0.5)
                ax.add_patch(rect)
        path = res["path"]
        px = [c + 0.5 for r, c in path]
        py = [rows - 1 - r + 0.5 for r, c in path]
        ax.plot(px, py, color=line_col, linewidth=2.8, marker="o", markersize=3.5, zorder=5)
        sr, sc = canteen
        gr, gc = goal
        ax.scatter([sc + 0.5], [rows - 1 - sr + 0.5], s=90, c="#DC2626", zorder=6)
        ax.scatter([gc + 0.5], [rows - 1 - gr + 0.5], s=90, c="#16A34A", zorder=6)
        ax.set_xlim(0, cols)
        ax.set_ylim(0, rows)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"{title} — Path Cost: {res['cost']} steps | Expanded: {res['expanded_nodes']} cells",
                     color="#F8FAFC", fontsize=9.5, fontweight="bold", pad=5)

    fig.suptitle(f"Route Comparison: Canteen → {classroom_key}", color="#93C5FD", fontsize=12, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out_path = os.path.join(out_dir, filename)
    plt.savefig(out_path, dpi=300, facecolor=fig.get_facecolor())
    plt.close(fig)
    return out_path


def plot_all_classrooms_grid_bleed(out_dir: str) -> str:
    """Render a 2x2 grid on a split-bleed canvas showing A* vs GBFS paths across all 4 classrooms."""
    grid, canteen, classrooms = build_campus_15x15_grid()
    fig, axes = plt.subplots(2, 2, figsize=(6.666, 7.5), dpi=300)
    fig.patch.set_facecolor("#0F172A")
    rows, cols = grid.shape

    for ax, (label, goal) in zip(axes.flat, classrooms.items()):
        res_gbfs = gbfs_search(grid, canteen, goal)
        res_astar = astar_search(grid, canteen, goal)
        ax.set_facecolor("#1E293B")
        for r in range(rows):
            for c in range(cols):
                y = rows - 1 - r
                if grid[r, c] == 1:
                    face = "#334155"
                elif (r, c) in res_astar["expanded_set"]:
                    face = "#D1FAE5"
                else:
                    face = "#F8FAFC"
                ax.add_patch(plt.Rectangle((c, y), 1, 1, facecolor=face, edgecolor="#E2E8F0", linewidth=0.4))

        # Plot GBFS dashed orange vs A* solid green
        gx = [c + 0.5 for r, c in res_gbfs["path"]]
        gy = [rows - 1 - r + 0.5 for r, c in res_gbfs["path"]]
        ax.plot(gx, gy, color="#F59E0B", linestyle="--", linewidth=2.2, label=f"GBFS ({res_gbfs['cost']})", zorder=4)

        ax_x = [c + 0.5 for r, c in res_astar["path"]]
        ax_y = [rows - 1 - r + 0.5 for r, c in res_astar["path"]]
        ax.plot(ax_x, ax_y, color="#059669", linewidth=2.6, label=f"A* ({res_astar['cost']})", zorder=5)

        sr, sc = canteen
        gr, gc = goal
        ax.scatter([sc + 0.5], [rows - 1 - sr + 0.5], s=80, c="#DC2626", zorder=6)
        ax.scatter([gc + 0.5], [rows - 1 - gr + 0.5], s=80, c="#2563EB", zorder=6)
        ax.set_xlim(0, cols)
        ax.set_ylim(0, rows)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        short_lbl = label.split()[0]
        ax.set_title(f"{short_lbl}: A*={res_astar['cost']} vs GBFS={res_gbfs['cost']} steps\n"
                     f"Exp: A*={res_astar['expanded_nodes']}, GBFS={res_gbfs['expanded_nodes']}",
                     color="#F8FAFC", fontsize=8.2, fontweight="bold", pad=5)
        leg_loc = "lower right" if short_lbl in ("C101", "C202") else "lower left"
        ax.legend(loc=leg_loc, fontsize=6.5, framealpha=0.9, facecolor="#0F172A", edgecolor="#475569", labelcolor="#F8FAFC")

    fig.suptitle("Multi-Classroom Delivery: A* (Solid) vs. GBFS (Dashed)",
                 color="#93C5FD", fontsize=11.5, fontweight="bold", y=0.97)
    plt.subplots_adjust(left=0.04, right=0.96, bottom=0.03, top=0.90, wspace=0.14, hspace=0.26)
    out_path = os.path.join(out_dir, "bleed_four_classrooms_comparison.png")
    plt.savefig(out_path, dpi=300, facecolor=fig.get_facecolor())
    plt.close(fig)
    return out_path


def render_ascii_route(grid: np.ndarray, start: Coord, goal: Coord, path: List[Coord]) -> str:
    rows, cols = grid.shape
    path_set = set(tuple(p) for p in path)
    lines = ["   " + "".join(f"{c:2d}" for c in range(cols))]
    lines.append("   +" + "--" * cols + "+")
    for r in range(rows):
        row_str = []
        for c in range(cols):
            if (r, c) == tuple(start):
                row_str.append(" C")
            elif (r, c) == tuple(goal):
                row_str.append(" G")
            elif (r, c) in path_set:
                row_str.append(" *")
            elif grid[r, c] == 1:
                row_str.append(" #")
            else:
                row_str.append(" .")
        lines.append(f"{r:2d} |" + "".join(row_str) + "|")
    lines.append("   +" + "--" * cols + "+")
    return "\n".join(lines)


def main():
    # Save outputs to ./outputs/ for clean GitHub repo usage (and sync to sandbox intermediate if present)
    out_dir = os.path.abspath("outputs")
    os.makedirs(out_dir, exist_ok=True)
    sandbox_mid = os.path.abspath("artifacts/file_generation/ttl=63d/intermediate")
    target_dirs = [out_dir]
    if os.path.exists(os.path.dirname(sandbox_mid)):
        os.makedirs(sandbox_mid, exist_ok=True)
        if sandbox_mid != out_dir:
            target_dirs.append(sandbox_mid)

    # 1. Tier 1: 5x5 Mini-Matrix Trace Verification (Slides 12-14)
    grid5, s5, g5 = build_5x5_demo_grid()
    trace_results = {
        "BFS": bfs_search(grid5, s5, g5),
        "GBFS": gbfs_search(grid5, s5, g5),
        "A*": astar_search(grid5, s5, g5),
    }

    print("=" * 78)
    print("GITHUB REPO: https://github.com/Vietchik/robot")
    print("GROUP 5 (TOPIC 3): CAMPUS DELIVERY ROBOT PATHFINDING (BFS, GBFS, A*)")
    print("=" * 78)
    print("\n[TIER 1] 5x5 MANUAL TRACE VERIFICATION (Start (0,0) -> Goal (0,4)) — Slides 12-14")
    print("-" * 78)
    print(f"{'Algorithm':<12} | {'Path Cost':<12} | {'Expanded Nodes':<16} | {'Optimal Path Sequence'}")
    print("-" * 78)
    for alg_name, res in trace_results.items():
        path_str = " -> ".join(f"({r},{c})" for r, c in res["path"])
        print(f"{alg_name:<12} | {str(res['cost']) + ' steps':<12} | {str(res['expanded_nodes']) + ' / 21 cells':<16} | {path_str}")

    print("\n--- A* Step-by-Step Priority Queue Trace on 5x5 Grid (Slide 14) ---")
    for row in trace_results["A*"]["trace"]:
        print(f"  Step {row['step']:2d}: Pop u={tuple(row['u'])} (g={row['g']}, h={row['h']}, f={row['f']}) "
              f"| Open Top={[(tuple(x[0]), x[1]) for x in row['open'][:4]]} {row['note']}")

    # 2. Tier 2: 15x15 NEU Campus Classroom Delivery (Slides 15-16)
    grid15, canteen, classrooms = build_campus_15x15_grid()
    campus_results = {}
    print("\n" + "=" * 78)
    print("[TIER 2] 15x15 NEU CAMPUS DELIVERY ROUTES (Canteen (1,1) -> 4 Classrooms) — Slides 15-16")
    print("-" * 78)
    print(f"{'Target Classroom':<20} | {'h0':<4} | {'BFS Cost (Nodes)':<18} | {'GBFS Cost (Nodes)':<18} | {'A* Cost (Nodes)':<18}")
    print("-" * 78)

    tot_bfs_c = tot_bfs_n = tot_gbfs_c = tot_gbfs_n = tot_ast_c = tot_ast_n = 0
    for cname, gpos in classrooms.items():
        h0 = manhattan(canteen, gpos)
        r_bfs = bfs_search(grid15, canteen, gpos)
        r_gbfs = gbfs_search(grid15, canteen, gpos)
        r_ast = astar_search(grid15, canteen, gpos)
        campus_results[cname] = {
            "goal": gpos,
            "manhattan_h0": h0,
            "BFS": r_bfs,
            "GBFS": r_gbfs,
            "A*": r_ast,
        }
        tot_bfs_c += r_bfs["cost"]; tot_bfs_n += r_bfs["expanded_nodes"]
        tot_gbfs_c += r_gbfs["cost"]; tot_gbfs_n += r_gbfs["expanded_nodes"]
        tot_ast_c += r_ast["cost"]; tot_ast_n += r_ast["expanded_nodes"]
        s_bfs = f"{r_bfs['cost']} steps ({r_bfs['expanded_nodes']})"
        s_gbfs = f"{r_gbfs['cost']} steps ({r_gbfs['expanded_nodes']})"
        s_ast = f"{r_ast['cost']} steps ({r_ast['expanded_nodes']})"
        print(f"{cname:<20} | {h0:<4d} | {s_bfs:<18} | {s_gbfs:<18} | {s_ast:<18}")

    print("-" * 78)
    print(f"{'Total (All 4)':<20} | {'66':<4} | {f'{tot_bfs_c} steps ({tot_bfs_n})':<18} | "
          f"{f'{tot_gbfs_c} steps ({tot_gbfs_n})':<18} | {f'{tot_ast_c} steps ({tot_ast_n})':<18}")
    reduction_pct = (1.0 - tot_ast_n / tot_bfs_n) * 100.0
    print(f"--> A* achieves 100% path optimality ({tot_ast_c} steps vs {tot_gbfs_c} in GBFS) "
          f"while cutting node expansions by {reduction_pct:.1f}% over BFS ({tot_ast_n} vs {tot_bfs_n} cells)!")

    print("\n--- ASCII Map Preview: Canteen (1,1) -> Lab-AI (8,13) [A* Optimal 19-Step Route] ---")
    print(render_ascii_route(grid15, canteen, classrooms["Lab-AI (East Tech)"], campus_results["Lab-AI (East Tech)"]["A*"]["path"]))

    # 3. Tier 3: 200-Maze Multi-Scale Benchmark (Slides 18-20)
    bench_summary = run_scalability_benchmarks(runs_per_scale=50)
    print("\n" + "=" * 78)
    print("[TIER 3] MULTI-SCALE SCALABILITY BENCHMARK (50 Mazes/Scale = 200 Mazes) — Slide 18")
    print("-" * 78)
    print(f"{'Grid Scale':<12} | {'Algorithm':<8} | {'Avg Path':<10} | {'Optimal %':<10} | {'Expanded':<10} | {'Time (ms)':<10} | {'RAM (KB)':<10}")
    print("-" * 78)
    for sc, algs in bench_summary.items():
        for alg_name in ("BFS", "GBFS", "A*"):
            m = algs[alg_name]
            print(f"{sc:<12} | {alg_name:<8} | {m['avg_cost']:<10.2f} | {m['optimality_pct']:<9.1f}% | "
                  f"{m['avg_expanded']:<10.1f} | {m['avg_time_ms']:<10.3f} | {m['avg_mem_kb']:<10.2f}")
        print("-" * 78)

    # 4. Generate 300-DPI Matplotlib Visual Plots
    for d in target_dirs:
        plot_5x5_trace_cards(d)
        plot_campus_overview_bleed(d)
        plot_algorithm_comparison_bleed(d, "Lab-AI (East Tech)", "bleed_c202_trap_comparison.png")
        plot_algorithm_comparison_bleed(d, "C305 (Auditorium)", "bleed_c305_long_comparison.png")
        plot_all_classrooms_grid_bleed(d)

        report_data = {
            "trace_5x5": {
                k: {
                    "cost": v["cost"],
                    "expanded_nodes": v["expanded_nodes"],
                    "path": v["path"],
                    "trace": v["trace"],
                }
                for k, v in trace_results.items()
            },
            "campus_15x15": {
                cname: {
                    "goal": info["goal"],
                    "h0": info["manhattan_h0"],
                    "BFS": {"cost": info["BFS"]["cost"], "expanded": info["BFS"]["expanded_nodes"], "path": info["BFS"]["path"]},
                    "GBFS": {"cost": info["GBFS"]["cost"], "expanded": info["GBFS"]["expanded_nodes"], "path": info["GBFS"]["path"]},
                    "A*": {"cost": info["A*"]["cost"], "expanded": info["A*"]["expanded_nodes"], "path": info["A*"]["path"]},
                }
                for cname, info in campus_results.items()
            },
            "benchmarks": bench_summary,
        }
        with open(os.path.join(d, "simulation_results.json"), "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=2)

    print(f"\n[SUCCESS] All simulation figures and JSON results saved to: {out_dir}")


if __name__ == "__main__":
    main()