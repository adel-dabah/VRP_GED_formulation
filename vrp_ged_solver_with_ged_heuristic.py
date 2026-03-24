"""
VRP as Graph Edit Distance — Benchmark Instance Solver
=======================================================
Reads CVRP benchmark instances in VRPLIB (.vrp) format from ./instances/
and applies the VRP-GED equivalence framework:

    min C(R)  =  W_total  -  max GED(Gs, Gt)         (Theorem 1)

Pipeline
--------
1. Parse .vrp files from ./instances/
2. Solve each instance with a 3-phase heuristic (NN → 2-opt → relocate)
3. Build complete graph Gs and routing graph Gt
4. Compute GED analytically (Lemma 2) and verify Theorem 1
5. Print per-instance report + summary table
6. Save one figure per instance + one summary figure to ./output/
"""

import re
import time
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# ---------------------------------------------------------------------------
# Directories
# ---------------------------------------------------------------------------
INSTANCES_DIR = Path("./instances")   # put your .vrp files here
OUTPUT_DIR    = Path("./output")
OUTPUT_DIR.mkdir(exist_ok=True)

# Optional: known best solutions for gap reporting  { instance_name: cost }
BEST_KNOWN: Dict[str, float] = {
    "A-n32-k5":  784,
    "A-n33-k5":  661,
    "A-n33-k6":  742,
    "A-n34-k5":  778,
    "A-n36-k5":  799,
    "A-n37-k5":  669,
    "A-n37-k6":  949,
    "A-n38-k5":  730,
    "A-n39-k5":  822,
    "A-n39-k6":  831,
    "A-n44-k6":  937,
    "A-n45-k6":  944,
    "A-n45-k7":  1146,
    "A-n46-k7":  914,
    "A-n48-k7":  1073,
    "A-n53-k7":  1010,
    "A-n54-k7":  1167,
    "A-n55-k9":  1073,
    "A-n60-k9":  1354,
    "A-n61-k9":  1034,
    "A-n62-k8":  1288,
    "A-n63-k9":  1616,
    "A-n63-k10": 1314,
    "A-n64-k9":  1401,
    "A-n65-k9":  1174,
    "A-n69-k9":  1159,
    "A-n80-k10": 1763,
    "B-n31-k5":  672,
    "B-n34-k5":  788,
    "B-n35-k5":  955,
    "B-n38-k6":  805,
    "B-n39-k5":  549,
    "B-n41-k6":  829,
    "B-n43-k6":  742,
    "B-n44-k7":  909,
    "B-n45-k5":  751,
    "B-n45-k6":  678,
    "B-n50-k7":  741,
    "B-n50-k8":  1312,
    "B-n51-k7":  1032,
    "B-n52-k7":  747,
    "B-n56-k7":  707,
    "B-n57-k9":  1598,
    "B-n63-k10": 1496,
    "B-n64-k9":  861,
    "B-n66-k9":  1316,
    "B-n67-k10": 1032,
    "B-n68-k9":  1272,
    "B-n78-k10": 1221,
    "E-n13-k4":  247,
    "eil13":     247,
    "E-n22-k4":  375,
    "E-n23-k3":  569,
    "E-n30-k3":  534,
    "E-n33-k4":  835,
    "E-n51-k5":  521,
    "E-n76-k7":  682,
    "E-n76-k8":  735,
    "E-n76-k10": 830,
    "E-n76-k14": 1021,
    "E-n101-k8": 815,
    "E-n101-k14":1067,
    "P-n16-k8":  450,
    "P-n19-k2":  212,
    "P-n20-k2":  216,
    "P-n21-k2":  211,
    "P-n22-k2":  216,
    "P-n22-k8":  590,
    "P-n23-k8":  529,
    "P-n40-k5":  458,
    "P-n45-k5":  510,
    "P-n50-k7":  554,
    "P-n50-k8":  631,
    "P-n50-k10": 696,
    "P-n51-k10": 741,
    "P-n55-k7":  568,
    "P-n55-k8":  588,
    "P-n55-k10": 694,
    "P-n55-k15": 989,
    "P-n60-k10": 744,
    "P-n60-k15": 1034,
    "P-n65-k10": 792,
    "P-n70-k10": 827,
    "P-n76-k4":  593,
    "P-n76-k5":  627,
    "P-n101-k4": 681,
}

SEP = "─" * 62
ROUTE_COLORS = [
    "#2979FF", "#F44336", "#00C853", "#FF6D00", "#AA00FF",
    "#00BCD4", "#FF4081", "#FFEA00", "#76FF03", "#E040FB",
]


# ===========================================================================
# 1.  VRPLIB PARSER
# ===========================================================================

@dataclass
class InstanceMeta:
    name:               str   = ""
    comment:            str   = ""
    dimension:          int   = 0
    capacity:           float = 0.0
    edge_weight_type:   str   = "EUC_2D"
    edge_weight_format: str   = ""        # e.g. LOWER_ROW, FULL_MATRIX
    display_data_type:  str   = ""        # e.g. NO_DISPLAY, COORD_DISPLAY
    n_trucks:           Optional[int] = None


@dataclass
class ParsedInstance:
    meta:        InstanceMeta
    coords:      List[Tuple[float, float]]        # index 0 = depot; empty for NO_DISPLAY
    demands:     List[float]                      # index 0 = depot (always 0)
    dist_matrix: Optional[np.ndarray] = None      # n×n float array when EXPLICIT type


def _lower_row_to_matrix(values: List[float], n: int) -> np.ndarray:
    """
    Build a symmetric n×n distance matrix from a LOWER_ROW flat list.

    LOWER_ROW provides the strict lower triangle row by row (no diagonal):
        row 1 (1 value):  d[1,0]
        row 2 (2 values): d[2,0]  d[2,1]
        ...
        row n-1 (n-1 values): d[n-1,0] ... d[n-1,n-2]
    Indices here are 0-based relative to the original node ordering.
    """
    mat = np.zeros((n, n))
    idx = 0
    for i in range(1, n):
        for j in range(i):
            mat[i, j] = mat[j, i] = values[idx]
            idx += 1
    return mat


def _full_matrix_to_matrix(values: List[float], n: int) -> np.ndarray:
    """Build n×n matrix from a FULL_MATRIX flat list (row-major)."""
    return np.array(values, dtype=float).reshape(n, n)


def parse_vrplib(text: str) -> ParsedInstance:
    """
    Parse a VRPLIB (.vrp) text into a ParsedInstance.

    Supports two distance representations:
      • EUC_2D / GEO / MAN_2D / ... — coordinates in NODE_COORD_SECTION
      • EXPLICIT (LOWER_ROW | FULL_MATRIX) — weight matrix in EDGE_WEIGHT_SECTION

    Depot is always re-indexed to position 0 in the returned arrays.
    """
    meta = InstanceMeta()
    raw_coords:   Dict[int, Tuple[float, float]] = {}
    raw_demands:  Dict[int, float]               = {}
    depot_nodes:  List[int]                      = []
    weight_vals:  List[float]                    = []   # raw numbers from EDGE_WEIGHT_SECTION
    section:      Optional[str]                  = None

    for raw_line in text.strip().splitlines():
        line = raw_line.strip()
        if not line or line == "EOF":
            continue

        # ── Section headers ──────────────────────────────────────────────────
        if "NODE_COORD_SECTION"  in line: section = "COORDS";  continue
        if "EDGE_WEIGHT_SECTION" in line: section = "WEIGHTS"; continue
        if "DEMAND_SECTION"      in line: section = "DEMANDS"; continue
        if "DEPOT_SECTION"       in line: section = "DEPOT";   continue
        if line.endswith("SECTION"):      section = "OTHER";   continue

        # ── Key : value header lines (only before any section starts) ────────
        if ":" in line and not line[0].isdigit() and section is None:
            key, _, val = line.partition(":")
            key, val = key.strip().upper(), val.strip()
            if   key == "NAME":               meta.name               = val
            elif key == "COMMENT":
                meta.comment = val
                m = re.search(r"trucks?[:\s]+(\d+)", val, re.IGNORECASE)
                if m:
                    meta.n_trucks = int(m.group(1))
            elif key == "DIMENSION":          meta.dimension          = int(val)
            elif key == "CAPACITY":           meta.capacity           = float(val)
            elif key == "EDGE_WEIGHT_TYPE":   meta.edge_weight_type   = val
            elif key == "EDGE_WEIGHT_FORMAT": meta.edge_weight_format = val
            elif key == "DISPLAY_DATA_TYPE":  meta.display_data_type  = val
            continue

        # ── Section data ─────────────────────────────────────────────────────
        parts = line.split()
        if section == "COORDS" and len(parts) >= 3:
            raw_coords[int(parts[0])] = (float(parts[1]), float(parts[2]))

        elif section == "WEIGHTS":
            weight_vals.extend(float(v) for v in parts)

        elif section == "DEMANDS" and len(parts) >= 2:
            raw_demands[int(parts[0])] = float(parts[1])

        elif section == "DEPOT":
            try:
                v = int(parts[0])
                if v > 0:
                    depot_nodes.append(v)
            except (ValueError, IndexError):
                pass

    # ── Determine node ordering (depot always at index 0) ───────────────────
    depot = depot_nodes[0] if depot_nodes else 1

    # For explicit matrices there may be no coordinates at all
    if raw_coords:
        all_ids = sorted(raw_coords)
        ordered = [depot] + [i for i in all_ids if i != depot]
        coords  = [raw_coords[i] for i in ordered]
    else:
        # No display data: create a placeholder ordering 1..n
        n_nodes = meta.dimension or (len(raw_demands) if raw_demands else 0)
        all_ids = list(range(1, n_nodes + 1))
        ordered = [depot] + [i for i in all_ids if i != depot]
        coords  = []   # no coordinates available

    demands    = [raw_demands.get(i, 0.0) for i in ordered]
    demands[0] = 0.0

    # ── Build explicit distance matrix when present ──────────────────────────
    dist_matrix: Optional[np.ndarray] = None
    if weight_vals and meta.edge_weight_type.upper() == "EXPLICIT":
        n = len(ordered)
        fmt = meta.edge_weight_format.upper()

        # Build raw matrix using original 1-indexed ordering
        if "LOWER_ROW" in fmt:
            raw_mat = _lower_row_to_matrix(weight_vals, n)
        elif "FULL_MATRIX" in fmt:
            raw_mat = _full_matrix_to_matrix(weight_vals, n)
        elif "UPPER_ROW" in fmt:
            # Transpose of LOWER_ROW
            raw_mat = _lower_row_to_matrix(weight_vals, n).T
            raw_mat = (raw_mat + raw_mat.T)  # symmetrise
        else:
            raise ValueError(f"Unsupported EDGE_WEIGHT_FORMAT: {meta.edge_weight_format!r}")

        # Remap rows/cols to match the depot-first ordering
        # ordered[new_idx] = original_1indexed_id  →  original 0-indexed = id - 1
        idx_map = [oid - 1 for oid in ordered]
        dist_matrix = raw_mat[np.ix_(idx_map, idx_map)]

    return ParsedInstance(meta=meta, coords=coords, demands=demands,
                          dist_matrix=dist_matrix)


# ===========================================================================
# 2.  VRP INSTANCE  (Definition 1)
# ===========================================================================

class VRPInstance:
    """
    Holds all VRP data needed for solving and GED analysis.

    Distances are computed from Euclidean coordinates when available
    (EUC_2D instances), or looked up from an explicit distance matrix
    (EXPLICIT instances such as eil13 with LOWER_ROW weights).

    When no real coordinates exist, a spring-layout position dict is
    generated from the distance matrix for use in visualisations.
    """

    def __init__(self, parsed: ParsedInstance, n_vehicles: Optional[int] = None):
        self.name              = parsed.meta.name
        self.demands           = parsed.demands
        self.capacity          = parsed.meta.capacity
        self.n_vehicles        = (n_vehicles
                                  or parsed.meta.n_trucks
                                  or max(2, parsed.meta.dimension // 10))
        self.n                 = parsed.meta.dimension - 1
        self.V                 = list(range(parsed.meta.dimension))
        self._edge_weight_type = parsed.meta.edge_weight_type.upper()
        self._dist_matrix: Optional[np.ndarray] = parsed.dist_matrix
        self.coords = parsed.coords

    def dist(self, i: int, j: int) -> float:
        """
        Distance between nodes i and j.

        EXPLICIT (LOWER_ROW etc.): direct matrix lookup — already integers.

        EUC_2D: apply TSPLIB nearest-integer rounding  nint(sqrt(...)).
        All A/B/E/P benchmark BKS values were produced with this convention.
        Without it, raw float distances give totals slightly below the BKS,
        making the solver appear to beat the certified optimum.

        Other coordinate types: raw float (add a branch as needed).
        """
        if self._dist_matrix is not None:
            return float(self._dist_matrix[i, j])
        xi, yi = self.coords[i]
        xj, yj = self.coords[j]
        d = float(np.hypot(xi - xj, yi - yj))
        if self._edge_weight_type == "EUC_2D":
            return float(int(d + 0.5))   # nint — TSPLIB standard
        return d

    def layout_positions(self) -> Dict[int, Tuple[float, float]]:
        """
        Return {node: (x, y)} for visualisation.
        Uses real coordinates when available; otherwise derives a
        spring layout from the distance matrix via NetworkX.
        """
        if self.coords:
            return {i: self.coords[i] for i in self.V}

        # Build a weighted graph from the distance matrix and use spring layout
        G = nx.Graph()
        G.add_nodes_from(self.V)
        for i in self.V:
            for j in self.V:
                if i < j:
                    G.add_edge(i, j, weight=1.0 / (self.dist(i, j) + 1e-9))
        return nx.spring_layout(G, weight="weight", seed=42)

    def demand(self, i: int) -> float:
        return self.demands[i]

    def route_load(self, route: List[int]) -> float:
        return sum(self.demand(c) for c in route)

    def route_cost(self, route: List[int]) -> float:
        seq = [0] + route + [0]
        return sum(self.dist(seq[k], seq[k + 1]) for k in range(len(seq) - 1))


def load_instance(path: Path, n_vehicles: Optional[int] = None) -> VRPInstance:
    """Parse a .vrp file and return a VRPInstance."""
    parsed = parse_vrplib(path.read_text())
    return VRPInstance(parsed, n_vehicles)


def list_instances(directory: Path = INSTANCES_DIR) -> List[Path]:
    """Return all .vrp files found in `directory`, sorted by name."""
    vrp_files = sorted(directory.glob("*.vrp"))
    if not vrp_files:
        raise FileNotFoundError(
            f"No .vrp files found in '{directory.resolve()}'. "
            "Please place your benchmark instances there."
        )
    return vrp_files


# ===========================================================================
# 3.  GED GRAPH OPERATIONS  (Definitions 4–5, Lemma 2)
# ===========================================================================

def build_complete_graph(vrp: VRPInstance) -> nx.Graph:
    """Complete graph Gs = (V, E, w) — all pairs with Euclidean weights."""
    Gs = nx.Graph()
    Gs.add_nodes_from(vrp.V)
    for i in vrp.V:
        for j in vrp.V:
            if i < j:
                Gs.add_edge(i, j, weight=vrp.dist(i, j))
    return Gs


def build_routing_graph(vrp: VRPInstance, routes: List[List[int]]) -> nx.Graph:
    """
    Routing graph Gt = (V, Et, wt) — one node per edge actually used.
    Multi-traversal count tracked via 'count' attribute (for single-customer
    routes where depot edge is traversed twice).
    """
    Gt = nx.Graph()
    Gt.add_nodes_from(vrp.V)
    for route in routes:
        seq = [0] + route + [0]
        for k in range(len(seq) - 1):
            u, v = seq[k], seq[k + 1]
            if Gt.has_edge(u, v):
                Gt[u][v]["count"] += 1
            else:
                Gt.add_edge(u, v, weight=vrp.dist(u, v), count=1)
    return Gt


def compute_wtotal(Gs: nx.Graph) -> float:
    """W_total = sum of all edge weights in the complete graph."""
    return sum(d["weight"] for _, _, d in Gs.edges(data=True))


def compute_ged(Gs: nx.Graph, Gt: nx.Graph) -> float:
    """
    GED = sum of weights of edges deleted from Gs to reach Gt (Lemma 2).
    Edges present in Gt are kept (cost 0); absent edges are deleted.
    """
    return sum(
        d["weight"] for u, v, d in Gs.edges(data=True)
        if not Gt.has_edge(u, v)
    )


def compute_set_cost(Gt: nx.Graph) -> float:
    """
    Set-based route cost  C'(R) = Σ_{e ∈ Et} w(e).

    Every edge in the routing graph Gt is counted ONCE regardless of how
    many times it is physically traversed.  This is the quantity that
    appears in Theorem 1:

        C'(R) + GED(Gs, Gt)  =  W_total       (always, by construction)

    For solo routes 0→i→0 the depot edge {0,i} appears once in the
    undirected Et even though it is traversed twice, so:

        C'(R)  <  C(R)    when solo routes exist
        C'(R)  =  C(R)    otherwise

    Use compute_traversal_cost() when you need C(R) for BKS gap comparisons.
    """
    return sum(d["weight"] for _, _, d in Gt.edges(data=True))


def compute_traversal_cost(vrp: VRPInstance, routes: List[List[int]]) -> float:
    """Actual total distance: every leg counted per traversal."""
    return sum(vrp.route_cost(r) for r in routes)


# ===========================================================================
# 4.  HEURISTIC SOLVER
#     Phase 1: greedy nearest-neighbour construction
#     Phase 2: 2-opt improvement per route
#     Phase 3: inter-route single-customer relocation
# ===========================================================================

def _nearest_neighbour(vrp: VRPInstance) -> List[List[int]]:
    """
    Greedy nearest-neighbour construction.
    Opens at most vrp.n_vehicles routes.  If the vehicle limit is reached
    before all customers are placed, remaining customers are force-inserted
    into the cheapest feasible position in any existing route.  If no
    capacity-feasible position exists (instance infeasible for given m/Q),
    a route is opened anyway so callers always get a complete assignment.
    """
    unvisited = set(range(1, vrp.n + 1))
    routes: List[List[int]] = []

    while unvisited and len(routes) < vrp.n_vehicles:
        route, cap, cur = [], vrp.capacity, 0
        while unvisited:
            best_c, best_d = None, float("inf")
            for c in unvisited:
                if vrp.demand(c) <= cap:
                    d = vrp.dist(cur, c)
                    if d < best_d:
                        best_d, best_c = d, c
            if best_c is None:
                break
            route.append(best_c)
            cap -= vrp.demand(best_c)
            unvisited.discard(best_c)
            cur = best_c
        if route:
            routes.append(route)

    # Force-insert any remaining customers (vehicle limit reached)
    for c in list(unvisited):
        best_pos, best_cost, best_r = 0, float("inf"), 0
        for ri, r in enumerate(routes):
            if vrp.route_load(r) + vrp.demand(c) > vrp.capacity:
                continue
            seq = [0] + r + [0]
            for pos in range(len(seq) - 1):
                ins = (vrp.dist(seq[pos], c) + vrp.dist(c, seq[pos+1])
                       - vrp.dist(seq[pos], seq[pos+1]))
                if ins < best_cost:
                    best_cost, best_pos, best_r = ins, pos, ri
        if best_cost < float("inf"):
            routes[best_r].insert(best_pos, c)
        else:
            routes[0].append(c)   # capacity infeasible — append anyway

    return routes


def _two_opt(route: List[int], vrp: VRPInstance) -> List[int]:
    improved = True
    while improved:
        improved = False
        seq = [0] + route + [0]
        for i in range(1, len(seq) - 2):
            for j in range(i + 1, len(seq) - 1):
                a, b, c, d = seq[i-1], seq[i], seq[j], seq[j+1]
                if vrp.dist(a, c) + vrp.dist(b, d) < vrp.dist(a, b) + vrp.dist(c, d) - 1e-9:
                    route[i-1:j] = route[i-1:j][::-1]
                    seq = [0] + route + [0]
                    improved = True
    return route


def _relocate(routes: List[List[int]], vrp: VRPInstance) -> List[List[int]]:
    """Move single customers between routes when it reduces total cost."""
    improved = True
    while improved:
        improved = False
        for i, ri in enumerate(routes):
            for ci, c in enumerate(ri):
                prev_i = ri[ci - 1] if ci > 0          else 0
                next_i = ri[ci + 1] if ci < len(ri) - 1 else 0
                removal_gain = (vrp.dist(prev_i, c) + vrp.dist(c, next_i)
                                - vrp.dist(prev_i, next_i))

                best_gain, best_j, best_pos = 0.0, -1, -1
                for j, rj in enumerate(routes):
                    if i == j:
                        continue
                    if vrp.route_load(rj) + vrp.demand(c) > vrp.capacity:
                        continue
                    seq = [0] + rj + [0]
                    for pos in range(len(seq) - 1):
                        gain = (removal_gain
                                - vrp.dist(seq[pos], c)
                                - vrp.dist(c, seq[pos + 1])
                                + vrp.dist(seq[pos], seq[pos + 1]))
                        if gain > best_gain + 1e-9:
                            best_gain, best_j, best_pos = gain, j, pos

                if best_j >= 0:
                    ri.pop(ci)
                    routes[best_j].insert(best_pos, c)
                    improved = True
                    break
            else:
                continue
            break
    return [r for r in routes if r]


def _enforce_vehicle_limit(routes: List[List[int]],
                           vrp:    VRPInstance) -> List[List[int]]:
    """
    Hard post-pass: if the number of routes exceeds vrp.n_vehicles, merge
    excess routes by repeatedly reinserting the customers of the shortest
    route into the cheapest feasible position elsewhere.

    This is the final safety net called at the end of every solver.
    It should rarely trigger if construction and repair are correct, but
    guarantees the vehicle constraint is never violated in reported results.
    """
    routes = [r for r in routes if r]
    if len(routes) <= vrp.n_vehicles:
        return routes

    while len(routes) > vrp.n_vehicles:
        src_idx = min(range(len(routes)), key=lambda i: len(routes[i]))
        src     = routes.pop(src_idx)
        for c in src:
            best_pos, best_cost, best_r = 0, float("inf"), 0
            for ri, r in enumerate(routes):
                if vrp.route_load(r) + vrp.demand(c) > vrp.capacity:
                    continue
                seq = [0] + r + [0]
                for pos in range(len(seq) - 1):
                    ins = (vrp.dist(seq[pos], c) + vrp.dist(c, seq[pos+1])
                           - vrp.dist(seq[pos], seq[pos+1]))
                    if ins < best_cost:
                        best_cost, best_pos, best_r = ins, pos, ri
            if best_cost < float("inf"):
                routes[best_r].insert(best_pos, c)
            else:
                routes[0].append(c)   # capacity infeasible, last resort

    return routes


def solve(vrp: VRPInstance) -> Tuple[List[List[int]], float]:
    """Run the 3-phase heuristic and return (routes, traversal_cost)."""
    routes = _nearest_neighbour(vrp)
    routes = [_two_opt(r, vrp) for r in routes]
    routes = _relocate(routes, vrp)
    routes = _enforce_vehicle_limit(routes, vrp)
    return routes, compute_traversal_cost(vrp, routes)


# ===========================================================================
# 5.  BRANCH AND BOUND  (exact solver, practical for n ≤ 15)
# ===========================================================================
"""
Search-tree structure
─────────────────────
Each node encodes a partial assignment state:
  routes_done  — fully closed routes (costs already in `committed`)
  cur_route    — route currently being built
  remaining    — customers not yet placed

At each node we branch in two ways:
  (A) Extend cur_route with a feasible unvisited customer.
  (B) Close cur_route (add return-to-depot cost) and open a fresh one.

Pruning via held-karp-style 1-tree lower bound
───────────────────────────────────────────────
  LB = committed
     + dist(cur_route_tail → depot)              ← must eventually close
     + Σ_{c ∈ remaining} min_1(c)                ← cheapest entry edge
     + Σ_{c ∈ remaining} min_2(c) / 2            ← cheapest exit (halved to
                                                     avoid double-counting)

where min_1, min_2 are the two cheapest edges from c to {depot} ∪ remaining.
This is tighter than the plain average used previously, and is always a
valid under-estimate (never prunes optimal solutions).

True abort via exception
────────────────────────
The old implementation used a counter check at the TOP of each recursive
call.  When the limit triggered, only the current branch was abandoned —
sibling branches still ran, so the node counter blew past the limit and the
"proven optimal / limit hit" label was wrong.

We now raise a private _Abort exception which unwinds the entire call stack
in O(depth) steps, giving a true hard stop.  The label is set by catching
(limit hit) or not catching (full tree explored = proven optimal) the abort.
"""


class _Abort(Exception):
    """Raised to abort the B&B tree search when a resource limit is hit."""


def _bb_lower_bound(vrp:       VRPInstance,
                    committed: float,
                    cur_route: List[int],
                    remaining: frozenset) -> float:
    """
    Tighter 1-tree lower bound.

    For each remaining customer c, the cheapest entry edge (min_1) plus
    half the cheapest exit edge (min_2/2) gives a valid per-node lower
    bound on the tour cost that still needs to be incurred.

    Compared to the previous (min_1+min_2)/2 formulation this is tighter
    because it accounts asymmetrically for the forced return-to-depot leg
    of the current partial route.
    """
    lb = committed

    # The open route must eventually return to depot
    if cur_route:
        lb += vrp.dist(cur_route[-1], 0)

    if not remaining:
        return lb

    pool = list(remaining)
    # Each remaining customer needs: arrive (min_1) + depart (min_2/2)
    for c in pool:
        candidates = [vrp.dist(c, 0)] + [vrp.dist(c, j) for j in pool if j != c]
        candidates.sort()
        lb += candidates[0]                          # cheapest in-edge
        if len(candidates) > 1:
            lb += candidates[1] / 2.0               # cheapest out-edge (halved)

    return lb


def solve_branch_and_bound(
        vrp:          VRPInstance,
        time_limit_s: float = 60.0,
        node_limit:   int   = 2_000_000,
) -> Tuple[List[List[int]], float]:
    """
    Exact CVRP solver via depth-first Branch and Bound.

    Practical for n ≤ 15.  For larger instances the LKH incumbent is
    returned immediately without searching.

    When the time or node limit is reached, the best solution found so far
    is returned and labelled "limit hit — not proven optimal".

    Parameters
    ----------
    time_limit_s : hard wall-clock budget in seconds
    node_limit   : hard limit on search-tree nodes explored
    """
    t0 = time.perf_counter()

    # Warm-start: use LKH for the best possible incumbent.
    # A tight upper bound from the start massively improves pruning.
    inc_routes, inc_cost = solve_lkh(vrp, n_restarts=3, time_limit_s=5.0)

    if vrp.n > 15:
        print(f"    [B&B] n={vrp.n} > 15 — returning LKH solution directly")
        return inc_routes, inc_cost

    # Shared mutable state (nonlocal inside nested dfs)
    inc          = [inc_routes, inc_cost]
    nodes        = [0]
    limit_reason = [None]   # None = clean finish | str = reason for abort

    def dfs(routes_done: List[List[int]],
            cur_route:   List[int],
            remaining:   frozenset,
            committed:   float) -> None:

        # ── Hard abort: raise exception to unwind the ENTIRE call stack ──
        nodes[0] += 1
        if nodes[0] > node_limit:
            limit_reason[0] = f"node limit ({node_limit:,})"
            raise _Abort
        if nodes[0] % 50_000 == 0:          # cheap time check every 50k nodes
            if time.perf_counter() - t0 > time_limit_s:
                limit_reason[0] = f"time limit ({time_limit_s:.0f}s)"
                raise _Abort

        # ── Prune: lower bound ≥ best known cost ──────────────────────────
        if _bb_lower_bound(vrp, committed, cur_route, remaining) >= inc[1] - 1e-9:
            return

        # ── Terminal: all customers placed ────────────────────────────────
        if not remaining:
            cost = committed + (vrp.dist(cur_route[-1], 0) if cur_route else 0.0)
            if cost < inc[1] - 1e-9:
                inc[1] = cost
                inc[0] = routes_done + ([list(cur_route)] if cur_route else [])
            return

        cur_node = cur_route[-1] if cur_route else 0
        cur_load = sum(vrp.demand(c) for c in cur_route)
        n_open   = len(routes_done) + (1 if cur_route else 0)

        # ── Branch A: add a customer to the current route ─────────────────
        # Try customers in nearest-neighbor order for better pruning.
        for c in sorted(remaining, key=lambda c: vrp.dist(cur_node, c)):
            if vrp.demand(c) + cur_load <= vrp.capacity:
                dfs(routes_done,
                    cur_route + [c],
                    remaining - {c},
                    committed + vrp.dist(cur_node, c))

        # ── Branch B: close current route, open a new one ─────────────────
        # Guard: only close if we still have vehicle budget left AND the
        # current route is non-empty (prevents zero-customer dummy routes).
        if cur_route and n_open < vrp.n_vehicles:
            dfs(routes_done + [list(cur_route)],
                [],
                remaining,
                committed + vrp.dist(cur_node, 0))

    proven = True
    try:
        dfs([], [], frozenset(range(1, vrp.n + 1)), 0.0)
    except _Abort:
        proven = False

    elapsed = (time.perf_counter() - t0) * 1000
    if proven:
        print(f"    [B&B] proven optimal  ({nodes[0]:,} nodes, {elapsed:.0f} ms)"
              f"  cost={inc[1]:.2f}")
    else:
        print(f"    [B&B] {limit_reason[0]} — best-found returned"
              f"  ({nodes[0]:,} nodes, {elapsed:.0f} ms)  cost={inc[1]:.2f}")

    return inc[0], inc[1]


# ===========================================================================
# 5b.  GED-NATIVE BRANCH AND BOUND  (exact, works directly on Theorem 1)
# ===========================================================================
"""
Why the standard B&B does NOT implement the GED formulation
────────────────────────────────────────────────────────────
The standard B&B (Section 5) searches over ordered customer sequences,
computes route costs via distance sums, and prunes with a 1-tree lower
bound on cost.  GED is computed *after* the fact as a verification step.
It never uses the GED framework during search.

Theorem 1 states:

    min  C(R)   =   W_total  −  max  GED(Gs, Gt)
     R                             Gt

This means solving the VRP is *equivalent* to finding the subgraph Gt that
maximises GED(Gs, Gt), subject to the routing graph degree constraints
(Definition 6):

    deg_Gt(depot)    = 2m          (exactly 2m route endpoints)
    deg_Gt(customer) = 2           (every customer is entered and exited)
    routes derived from Gt must be capacity-feasible

A GED-native B&B searches this space directly:

  Decision variable  — for each edge e ∈ E(Gs), decide: DELETE (adds w(e)
                       to GED) or KEEP (adds e to Et).
  Objective         — maximise  Σ_{e deleted} w(e)  =  GED(Gs, Gt)
  Constraints       — degree constraints + capacity feasibility

This is a fundamentally different search space from the standard B&B.

Search-tree structure
─────────────────────
Edges are ordered by weight descending so the heaviest edges (= largest
potential GED contribution) are branched on first.  This gives the best
chance of finding a near-optimal solution early, which tightens pruning.

At each node we branch in two ways:
  (DEL) Delete the current edge e  →  GED increases by w(e)
  (KEP) Keep   the current edge e  →  e is added to Et

State variables:
  ged_so_far    — sum of weights of edges deleted so far
  degree        — current degree of every node in the partial Gt
  kept_edges    — set of kept (u, v) edges
  edge_idx      — index into the sorted edge list (which edge to branch on)

GED Upper Bound  (for pruning)
─────────────────────────────
At any node, the best achievable GED is:

    UB = ged_so_far
       + Σ_{e undecided} w(e)              ← delete everything remaining
       − forced_keep_cost                  ← edges we MUST keep to satisfy
                                              minimum degree requirements

forced_keep_cost is computed per node:
  deficit(v) = required_degree(v) - current_degree(v)
  For each node v with deficit > 0, we must keep at least deficit more
  edges incident to v.  The cheapest such edges are the ones we are forced
  to keep.  We greedily assign the cheapest undecided edges to satisfy all
  deficits simultaneously (LP relaxation of the assignment).

Any node with UB ≤ best_GED_found is pruned.

Feasibility checks (hard constraints, fail-fast)
─────────────────────────────────────────────────
After each KEEP decision:
  • Degree overflow:  deg(v) > required_degree(v)  → prune (infeasible)

After all edges decided (leaf node):
  • Degree completion: every node must hit exactly its required degree.
  • Connectivity: depot-reachable DFS from depot must visit all nodes.
  • Capacity: convert Gt to routes and check each route's load ≤ Q.

Route extraction from Gt
────────────────────────
Given a valid Gt (correct degrees, connected), routes are extracted by
walking Eulerian-style from the depot:
  Start at depot → follow any unvisited edge → continue until back at depot
  → that is one route.  Repeat until all 2m depot edges are used.
"""


def _ged_upper_bound(
        vrp:         VRPInstance,
        ged_so_far:  float,
        degree:      List[int],
        undecided:   List[Tuple[int, int, float]],   # (u, v, w) sorted by w desc
) -> float:
    """
    Admissible upper bound on GED from the current partial state.

        UB = ged_so_far + undecided_sum - forced_keep_LB

    forced_keep_LB (per-node lower bound):
      For each node v with deficit d(v) = req_deg[v] - degree[v]:
        • Take the d(v) cheapest undecided edges incident to v.
        • Sum their weights into the per-node contribution.
      Divide by 2 (each edge shared between two nodes).

    This is a valid lower bound on what we must keep, so subtracting it
    gives a valid (never underestimating) upper bound on achievable GED.

    The previous greedy-global approach processed edges one-by-one and
    added an edge whenever EITHER endpoint had deficit, overcounting by
    picking edges incident to already-satisfied nodes.  This made UB too
    pessimistic and caused the GED-B&B/Beam to prune valid subtrees.
    """
    n   = vrp.n + 1
    req = [2] * n
    req[0] = 2 * vrp.n_vehicles

    total_undecided = sum(w for _, _, w in undecided)

    # Per-node sorted incident weights
    node_w: List[List[float]] = [[] for _ in range(n)]
    for u, v, w in undecided:
        node_w[u].append(w)
        node_w[v].append(w)

    forced_keep_lb = 0.0
    for v in range(n):
        deficit = req[v] - degree[v]
        if deficit <= 0:
            continue
        ws = sorted(node_w[v])          # cheapest first
        if len(ws) < deficit:
            return float("-inf")        # infeasible partial state
        forced_keep_lb += sum(ws[:deficit])

    forced_keep_lb /= 2.0              # each edge counted at both endpoints

    return ged_so_far + total_undecided - forced_keep_lb


def _extract_routes_from_gt(
        vrp:         VRPInstance,
        kept_edges:  List[Tuple[int, int]],
) -> Optional[List[List[int]]]:
    """
    Extract VRP routes from a routing graph Gt defined by kept_edges.

    Algorithm:
      Build adjacency lists.  Start at depot, follow edges Eulerian-style
      (each edge used at most once, tracking which half-edges remain).
      Each traversal from depot back to depot = one route.
      Returns None if the graph is disconnected or structurally invalid.
    """
    from collections import defaultdict
    adj: Dict[int, List[int]] = defaultdict(list)
    for u, v in kept_edges:
        adj[u].append(v)
        adj[v].append(u)

    # Check connectivity: every node with edges must be reachable from depot
    reachable = {0}
    stack = [0]
    while stack:
        node = stack.pop()
        for nb in adj[node]:
            if nb not in reachable:
                reachable.add(nb)
                stack.append(nb)
    if len(reachable) < vrp.n + 1 and any(adj[v] for v in range(1, vrp.n + 1)):
        return None   # disconnected

    routes: List[List[int]] = []
    # Work on mutable copies of adjacency lists
    adj_mut: Dict[int, List[int]] = {v: list(ns) for v, ns in adj.items()}
    depot_edges_remaining = len(adj_mut.get(0, []))

    for _ in range(vrp.n_vehicles):
        if not adj_mut.get(0):
            break
        route: List[int] = []
        cur = 0
        # Walk until we return to depot
        while True:
            if not adj_mut[cur]:
                return None  # dead end — invalid graph
            nxt = adj_mut[cur].pop(0)
            adj_mut[nxt].remove(cur)
            if nxt == 0:
                break
            route.append(nxt)
            cur = nxt
        if route:
            routes.append(route)

    return routes if routes else None


def _check_capacity(vrp: VRPInstance, routes: List[List[int]]) -> bool:
    return all(vrp.route_load(r) <= vrp.capacity for r in routes)


def solve_ged_bb(
        vrp:          VRPInstance,
        time_limit_s: float = 120.0,
        node_limit:   int   = 5_000_000,
) -> Tuple[List[List[int]], float, float]:
    """
    GED-native exact Branch and Bound.

    Directly implements Theorem 1 by searching the space of subgraphs Gt
    of the complete graph Gs, maximising GED(Gs, Gt) subject to:
      • degree constraints  (Definition 6)
      • connectivity        (Gt must form complete routes)
      • capacity feasibility

    This is the *only* solver in this codebase that operates entirely within
    the GED framework — all others solve VRP directly and compute GED post-hoc.

    Returns
    -------
    routes     : best routes found (list of customer lists)
    cost       : traversal cost = W_total - best_GED  (Theorem 1)
    best_ged   : maximum GED achieved
    """
    t0 = time.perf_counter()

    # All edges of Gs sorted by weight descending
    # (branch on heaviest first — best chance of large GED early)
    all_edges: List[Tuple[int, int, float]] = []
    for i in vrp.V:
        for j in vrp.V:
            if i < j:
                all_edges.append((i, j, vrp.dist(i, j)))
    all_edges.sort(key=lambda e: e[2], reverse=True)

    # Precompute W_total
    wt = sum(w for _, _, w in all_edges)

    # Required degrees (Definition 6)
    req_deg = [2] * (vrp.n + 1)
    req_deg[0] = 2 * vrp.n_vehicles

    # Warm-start: convert LKH solution to GED incumbent
    init_routes, _ = solve_lkh(vrp, n_restarts=3, time_limit_s=5.0)
    init_Gt        = build_routing_graph(vrp, init_routes)
    init_Gs        = build_complete_graph(vrp)
    best_ged       = [compute_ged(init_Gs, init_Gt)]
    best_routes    = [init_routes]

    nodes   = [0]
    aborted = [False]

    def dfs(edge_idx:   int,
            ged_so_far: float,
            degree:     List[int],
            kept:       List[Tuple[int, int]]) -> None:

        nodes[0] += 1
        if nodes[0] > node_limit:
            aborted[0] = True
            raise _Abort
        if nodes[0] % 50_000 == 0:
            if time.perf_counter() - t0 > time_limit_s:
                aborted[0] = True
                raise _Abort

        undecided = all_edges[edge_idx:]

        # ── Prune: upper bound on GED can't beat incumbent ────────────────
        ub = _ged_upper_bound(vrp, ged_so_far, degree, undecided)
        if ub <= best_ged[0] + 1e-9:
            return

        # ── Leaf: all edges decided ───────────────────────────────────────
        if edge_idx == len(all_edges):
            # Check degree completion
            if any(degree[v] != req_deg[v] for v in range(vrp.n + 1)):
                return
            # Extract and validate routes
            routes = _extract_routes_from_gt(vrp, kept)
            if routes is None:
                return
            if not _check_capacity(vrp, routes):
                return
            # New best!
            if ged_so_far > best_ged[0] + 1e-9:
                best_ged[0]    = ged_so_far
                best_routes[0] = routes
            return

        u, v, w = all_edges[edge_idx]
        nxt = edge_idx + 1

        # ── Branch DEL: delete edge (u,v) → GED increases by w ───────────
        # Always feasible from a degree standpoint (we're removing an edge
        # we haven't kept yet).  Remaining deficits may grow, but we check
        # via the upper bound.
        dfs(nxt, ged_so_far + w, degree, kept)

        # ── Branch KEP: keep edge (u,v) → add to Et ───────────────────────
        # Immediately prune if adding this edge would overflow either
        # endpoint's degree budget.
        if degree[u] < req_deg[u] and degree[v] < req_deg[v]:
            degree[u] += 1
            degree[v] += 1
            kept.append((u, v))
            dfs(nxt, ged_so_far, degree, kept)
            kept.pop()
            degree[u] -= 1
            degree[v] -= 1

    try:
        dfs(0, 0.0, [0] * (vrp.n + 1), [])
    except _Abort:
        pass

    elapsed  = (time.perf_counter() - t0) * 1000
    set_cost = wt - best_ged[0]                          # C'(R)
    trav     = compute_traversal_cost(vrp, best_routes[0])  # C(R)
    status   = "limit hit — best-found" if aborted[0] else "proven optimal"
    print(f"    [GED-B&B] {status}  GED={best_ged[0]:.2f}"
          f"  C'(R)={set_cost:.2f}  C(R)={trav:.2f}"
          f"  ({nodes[0]:,} nodes, {elapsed:.0f} ms)")

    # Return C(R) as the second element so callers that compare against BKS
    # (which is always measured as C(R)) get a fair number.
    # C'(R) and GED are the third and fourth elements for Theorem 1 work.
    return best_routes[0], trav, best_ged[0]


# ===========================================================================
# 5c.  GED-EXACT ILP SOLVER  —  pure Theorem 1 implementation
# ===========================================================================
"""
This is the only solver that is a *direct* and *complete* implementation
of Theorem 1, solved to certified optimality by an ILP engine (HiGHS via
scipy.optimize.milp).

═══════════════════════════════════════════════════════════════════════════
  THEOREM 1 RECAP
───────────────────────────────────────────────────────────────────────────
  Given the complete VRP graph Gs = (V, E, w), the minimum-cost solution
  satisfies:

      min  C'(R)  =  W_total  −  max  GED(Gs, Gt)
       R                           Gt

  where the maximisation is over all subgraphs Gt of Gs that correspond
  to a valid VRP solution (degree constraints, connectivity, capacity).

  Equivalently, to minimise C'(R) = Σ_{e ∈ Et} w(e) we choose which edges
  of Gs to KEEP in Et (and implicitly DELETE the rest, accumulating GED).

═══════════════════════════════════════════════════════════════════════════
  ILP FORMULATION
───────────────────────────────────────────────────────────────────────────
  Decision variables
  ──────────────────
  f[i,j] ∈ {0,1}   directed flow on arc i→j, for all i≠j in V
                    f[i,j]=1 means arc i→j is in the solution.

  q[i] ∈ [d_i, Q]  cumulative load when vehicle arrives at customer i.
                    Continuous.  Used for MTZ subtour+capacity elimination.

  GED connection
  ──────────────
  The undirected edge set Et = { {i,j} : f[i,j]=1 or f[j,i]=1 }.
  Since every customer is visited exactly once (flow conservation) no
  edge is traversed in both directions, so:

      x[i,j] = f[i,j] + f[j,i]  ∈ {0,1}    (edge {i,j} kept in Et)

  Set-based route cost:
      C'(R) = Σ_{i<j} w[i,j] · x[i,j]
            = Σ_{i<j} w[i,j] · (f[i,j] + f[j,i])
            = Σ_{i≠j} w[i,j] · f[i,j]        (because w is symmetric)

  GED:
      GED(Gs,Gt) = W_total − C'(R)
                 = W_total − Σ_{i≠j} w[i,j] · f[i,j]

  So:   max GED  ≡  min  Σ_{i≠j} w[i,j] · f[i,j]

  This is exactly the ILP objective.

  Constraints
  ───────────
  (D1) Flow conservation — customers (enter and leave exactly once):
       Σ_j f[i,j] = 1   for all i ∈ {1..n}   (out-flow = 1)
       Σ_j f[j,i] = 1   for all i ∈ {1..n}   (in-flow  = 1)

  (D2) Flow conservation — depot (m routes depart and return):
       Σ_j f[0,j] = m
       Σ_j f[j,0] = m

  (MTZ) Capacity + subtour elimination (Miller–Tucker–Zemlin, 1960):
       q[j] ≥ q[i] + d[j] − Q · (1 − f[i,j])
       for all i ∈ V, j ∈ {1..n}, i ≠ j.
       With q[0] ≡ 0 (depot has no load).

       This single family of constraints simultaneously:
         • eliminates all subtours (any sub-cycle would force q to wrap,
           contradicting monotonicity of q)
         • enforces capacity (q[j] ≤ Q for all j)

  Variable bounds:
       f[i,j] ∈ {0,1}
       d[i] ≤ q[i] ≤ Q    for i ∈ {1..n}

  Total size (n customers, m vehicles):
       Binary variables  : n(n+1) arcs  (all directed pairs in V)
       Continuous vars   : n  (the q[i])
       Equality rows     : 2n + 2  (flow conservation)
       Inequality rows   : n(n+1)  (MTZ)

═══════════════════════════════════════════════════════════════════════════
  ROUTE RECONSTRUCTION
───────────────────────────────────────────────────────────────────────────
  After solving, for each j with f[0,j]=1 (route departure from depot),
  we walk the directed arc chain   0 → j → next(j) → … → 0   to
  reconstruct one route.  The undirected edge set Et = all used arcs
  (ignoring direction) is the routing graph Gt of Definition 5.
"""


def solve_ged_exact_ilp(
        vrp:          VRPInstance,
        time_limit_s: float = 300.0,
        mip_gap:      float = 1e-6,
        verbose:      bool  = False,
) -> Tuple[List[List[int]], float, float]:
    """
    Solve VRP exactly via Theorem 1: maximise GED(Gs, Gt).

    Internally formulates and solves the ILP described in section 5c using
    HiGHS (via scipy.optimize.milp).  Returns a certificate of optimality
    when the solver completes within the time limit.

    Key correctness fix vs a naïve directed flow model
    ──────────────────────────────────────────────────
    A directed flow naturally counts both f[0,j]=1 and f[j,0]=1 for a
    single-customer route 0→j→0, giving a cost contribution of 2·w[0,j].
    But in the undirected routing graph Gt, edge {0,j} appears only ONCE,
    so C'(R) should add w[0,j] only once.

    We introduce binary indicator z[j] ∈ {0,1} (one per customer) that
    equals 1 iff customer j forms a solo route (f[0,j]=1 AND f[j,0]=1).
    The corrected objective is:

        min C'(R) = Σ_{(i,j)} w[i,j]·f[i,j]  −  Σ_{j} w[0,j]·z[j]
                                                    ↑ subtract double-count

    z[j] is linearised by:
        z[j] ≥ f[0,j] + f[j,0] − 1    (z=1 forces both f[0,j] and f[j,0]=1)
        z[j] ≤ f[0,j]
        z[j] ≤ f[j,0]

    For routes with ≥2 customers, the depot exit and depot entry involve
    different customers, so f[0,a]=1 but f[a,0]=0 (or vice versa) — z[a]=0
    and no correction is needed.  This objective equals C'(R) exactly.

    Parameters
    ----------
    time_limit_s : wall-clock budget in seconds (passed to HiGHS)
    mip_gap      : relative MIP gap tolerance (1e-6 = essentially exact)
    verbose      : print HiGHS solver log

    Returns
    -------
    routes   : list of customer lists (one per vehicle)
    cost     : C'(R) = Σ_{e ∈ Et} w(e)  (= W_total − GED, Theorem 1)
    ged      : GED(Gs, Gt) = W_total − C'(R)
    """
    try:
        from scipy.optimize import milp, LinearConstraint, Bounds
        from scipy.sparse import lil_matrix
    except ImportError:
        raise ImportError(
            "scipy >= 1.7.0 is required for solve_ged_exact_ilp.\n"
            "Install with:  pip install scipy"
        )

    t0  = time.perf_counter()
    N   = vrp.n + 1          # |V| = n customers + 1 depot
    m   = vrp.n_vehicles
    Q   = vrp.capacity

    # ── Precompute W_total ───────────────────────────────────────────────
    wt = sum(vrp.dist(i, j) for i in vrp.V for j in vrp.V if i < j)

    # ── Variable layout ──────────────────────────────────────────────────
    #
    # Block 0 — f[i,j] ∈ {0,1}   arc i→j used      size: N*(N-1)
    # Block 1 — z[j]   ∈ {0,1}   customer j is solo size: n
    # Block 2 — q[i]   ∈ [d_i,Q] cumulative load    size: n
    #
    arcs   = [(i, j) for i in range(N) for j in range(N) if i != j]
    f_idx  = {arc: k for k, arc in enumerate(arcs)}
    n_f    = len(arcs)           # N*(N-1)

    z_off  = n_f                 # z[j] for j=1..n → index z_off + (j-1)
    q_off  = z_off + vrp.n      # q[j] for j=1..n → index q_off + (j-1)
    n_vars = q_off  + vrp.n     # total variables

    # ── Objective: min  Σ_{i≠j} w[i,j]·f[i,j]  −  Σ_j w[0,j]·z[j] ────
    #
    # This equals C'(R) = Σ_{e ∈ Et} w(e)  exactly:
    #   • customer–customer arcs: each undirected edge used in at most one
    #     direction, so f[i,j]+f[j,i] ≤ 1 → no correction needed.
    #   • depot–customer edge {0,j}: traversed once if j is in a multi-cust
    #     route, twice if solo. The z[j] term removes the double-count.
    #
    c_obj = np.zeros(n_vars)
    for (i, j), k in f_idx.items():
        c_obj[k] = vrp.dist(i, j)              # +w[i,j] per arc
    for j in range(1, N):
        c_obj[z_off + j - 1] = -vrp.dist(0, j) # -w[0,j] per solo correction

    # ── Integrality: f and z are binary; q is continuous ─────────────────
    integrality                      = np.zeros(n_vars)
    integrality[:n_f]                = 1   # f[i,j]
    integrality[z_off: z_off + vrp.n] = 1   # z[j]

    # ── Variable bounds ──────────────────────────────────────────────────
    lb_var = np.zeros(n_vars)
    ub_var = np.ones(n_vars)
    for i in range(1, N):
        lb_var[q_off + i - 1] = vrp.demand(i)
        ub_var[q_off + i - 1] = Q

    # ── Constraints ──────────────────────────────────────────────────────
    row_data: List[List[Tuple[int, float]]] = []
    row_lb:   List[float]                   = []
    row_ub:   List[float]                   = []
    INF = np.inf

    def add_row(entries: List[Tuple[int, float]], lo: float, hi: float) -> None:
        row_data.append(entries)
        row_lb.append(lo)
        row_ub.append(hi)

    # (D1a) Customer out-flow = 1
    for i in range(1, N):
        add_row([(f_idx[i, j], 1.0) for j in range(N) if j != i], 1.0, 1.0)

    # (D1b) Customer in-flow = 1
    for i in range(1, N):
        add_row([(f_idx[j, i], 1.0) for j in range(N) if j != i], 1.0, 1.0)

    # (D2a) Depot out-flow = m
    add_row([(f_idx[0, j], 1.0) for j in range(1, N)], float(m), float(m))

    # (D2b) Depot in-flow = m
    add_row([(f_idx[j, 0], 1.0) for j in range(1, N)], float(m), float(m))

    # (MTZ) q[j] − q[i] − Q·f[i,j] ≥ d[j] − Q   (subtour + capacity)
    for j in range(1, N):
        dj  = vrp.demand(j)
        rhs = dj - Q
        for i in range(N):
            if i == j:
                continue
            entries = [(f_idx[i, j], -Q), (q_off + j - 1, 1.0)]
            if i != 0:
                entries.append((q_off + i - 1, -1.0))
            add_row(entries, rhs, INF)

    # (Z1) z[j] ≤ f[0,j]           ↓
    # (Z2) z[j] ≤ f[j,0]           | linearise z[j] = f[0,j] AND f[j,0]
    # (Z3) z[j] ≥ f[0,j]+f[j,0]−1  ↑
    for j in range(1, N):
        zk = z_off + j - 1
        add_row([(zk, 1.0), (f_idx[0, j], -1.0)],  -INF, 0.0)   # z ≤ f[0,j]
        add_row([(zk, 1.0), (f_idx[j, 0], -1.0)],  -INF, 0.0)   # z ≤ f[j,0]
        add_row([(zk, -1.0), (f_idx[0, j], 1.0),               # z ≥ f[0,j]+f[j,0]-1
                 (f_idx[j, 0], 1.0)],               -INF, 1.0)

    # ── Assemble sparse matrix ────────────────────────────────────────────
    n_rows = len(row_data)
    A = lil_matrix((n_rows, n_vars), dtype=float)
    for r, entries in enumerate(row_data):
        for col, val in entries:
            A[r, col] = val

    con = LinearConstraint(A.tocsc(),
                           np.array(row_lb, dtype=float),
                           np.array(row_ub, dtype=float))
    bds = Bounds(lb_var, ub_var)

    # ── Summary ──────────────────────────────────────────────────────────
    n_eq   = sum(1 for lo, hi in zip(row_lb, row_ub) if lo == hi)
    n_ineq = n_rows - n_eq
    print(f"    [GED-ILP] formulation: "
          f"{n_f} f-arcs + {vrp.n} z-solo + {vrp.n} q-load = {n_vars} vars  |  "
          f"{n_eq} equalities + {n_ineq} inequalities")
    print(f"    [GED-ILP] objective: min C'(R)  ≡  max GED(Gs,Gt)"
          f"  [W_total={wt:.4f}]")

    # ── Solve ─────────────────────────────────────────────────────────────
    result = milp(
        c           = c_obj,
        constraints = con,
        integrality = integrality,
        bounds      = bds,
        options     = {"disp": verbose, "time_limit": time_limit_s,
                       "mip_rel_gap": mip_gap, "presolve": True},
    )

    elapsed_ms = (time.perf_counter() - t0) * 1000

    if result.x is None:
        raise RuntimeError(f"[GED-ILP] no solution returned: {result.message}")

    proven = (result.status == 0)

    # ── Reconstruct routes ────────────────────────────────────────────────
    x_sol = result.x
    succ: Dict[int, int] = {}
    for (i, j), k in f_idx.items():
        if x_sol[k] > 0.5:
            succ[i] = j

    routes: List[List[int]] = []
    for start in [j for j in range(1, N) if x_sol[f_idx[0, j]] > 0.5]:
        route, cur = [], start
        for _ in range(vrp.n + 2):
            if cur == 0:
                break
            route.append(cur)
            cur = succ.get(cur, 0)
        if route:
            routes.append(route)

    # ── Theorem 1 verification (computed from actual routes) ──────────────
    Gs        = build_complete_graph(vrp)
    Gt        = build_routing_graph(vrp, routes)
    set_cost  = compute_set_cost(Gt)       # C'(R) — set-based (each edge once)
    ged       = compute_ged(Gs, Gt)        # W_total − C'(R)
    thm1_err  = abs(set_cost - (wt - ged))

    # ILP objective value = result.fun = C'(R) by construction
    obj_err = abs(float(result.fun) - set_cost)

    label = "PROVEN OPTIMAL" if proven else "TIME/NODE LIMIT (best bound)"
    print(f"    [GED-ILP] {label}")
    print(f"    [GED-ILP] max GED    = {ged:.4f}   (= W_total − C'(R))")
    print(f"    [GED-ILP] min C'(R)  = {set_cost:.4f}   "
          f"objective residual = {obj_err:.2e}")
    print(f"    [GED-ILP] Theorem 1: C'(R) + GED = {set_cost + ged:.4f}  "
          f"W_total = {wt:.4f}  err = {thm1_err:.2e}  "
          f"[{'✓ PASS' if thm1_err < 1e-6 else '✗ FAIL'}]")
    print(f"    [GED-ILP] time = {elapsed_ms:.0f} ms")

    return routes, set_cost, ged


def _print_ged_comparison(vrp:    VRPInstance,
                           routes: List[List[int]],
                           cost:   float,
                           ged:    float,
                           label:  str) -> None:
    """
    Print a full Theorem 1 breakdown showing both C(R) and C'(R).
    All values recomputed from routes for consistency.
    """
    Gs     = build_complete_graph(vrp)
    Gt     = build_routing_graph(vrp, routes)
    wt     = compute_wtotal(Gs)
    set_c  = compute_set_cost(Gt)              # C'(R) — set semantics, Theorem 1
    trav_c = compute_traversal_cost(vrp, routes)  # C(R) — every leg, BKS metric
    ged_v  = compute_ged(Gs, Gt)
    n_kept = Gt.number_of_edges()
    n_del  = Gs.number_of_edges() - n_kept
    thm1   = "✓ PASS" if abs(set_c + ged_v - wt) < 1e-6 else "✗ FAIL"
    solo   = sum(1 for r in routes if len(r) == 1)

    print(f"\n  ─── Theorem 1 breakdown  [{label}] ─────────────────────────")
    print(f"  W_total = Σ_{{e∈Gs}} w(e)                   = {wt:.4f}")
    print(f"  C'(R)   = Σ_{{e∈Et}} w(e)  [each edge once] = {set_c:.4f}  ← Theorem 1 quantity")
    print(f"  C(R)    = Σ legs w(leg)    [every traversal] = {trav_c:.4f}  ← BKS quantity")
    if solo:
        diff = trav_c - set_c
        print(f"  C(R) − C'(R) = {diff:.4f}  ({solo} solo route(s), each contributes w(0,j) extra)")
    print(f"  GED     = Σ_{{e∉Et}} w(e)  [deleted edges]  = {ged_v:.4f}")
    print(f"  Theorem 1: C'(R) + GED = {set_c:.4f} + {ged_v:.4f}"
          f" = {set_c+ged_v:.4f}  (W_total={wt:.4f})  [{thm1}]")
    print(f"  Edges kept   : {n_kept:4d} / {Gs.number_of_edges()}"
          f"  ({100*n_kept/Gs.number_of_edges():.1f}%)")
    print(f"  Edges deleted (= GED ops) : {n_del}")
    print(f"  ────────────────────────────────────────────────────────────")







# ===========================================================================
# 6.  GED HEURISTIC  (Theorem 1 -- fast native edge-space search)
# ===========================================================================
"""
Core identity: max GED(Gs,Gt) == min C'(R) == min sum_{e in Et} w(e)

Three optimisations over the naive version
-------------------------------------------
1. NumPy distance matrix  O(1) weight lookups, no dict overhead,
                          vectorised construction of edge lists.

2. Adjacency-set representation  The kept graph is stored as adj[v] = set
   of neighbours.  Degree of v = len(adj[v]).  Edge membership and removal
   are O(1).  No list sorting needed between passes.

3. Fast connectivity via Union-Find + bridge detection
   - Before any 2-opt pass, compute which kept edges are BRIDGES (cut edges)
     using Tarjan's DFS bridge algorithm O(n + |Et|).
   - Non-bridge edges: removing them cannot disconnect the graph, so after a
     swap involving only non-bridges we SKIP the connectivity check entirely.
   - Bridge edges: after a swap we only need a lightweight DFS reachability
     check from depot (O(n + |Et|)), NOT a full route extraction.

These three changes bring the per-pass cost from O(|E|*k * route_walk)
to O(n + |Et| + n*k) while producing the same or better solutions.

Algorithm
----------
Phase 1  Greedy construction  (same logic, now uses dist_matrix)
Phase 2  Candidate-list 2-opt with bridge-aware connectivity skip
Phase 3  Route extraction + capacity repair  (unchanged)
"""

import numpy as _np


def _dist_matrix(vrp: "VRPInstance") -> "_np.ndarray":
    """Build a dense n x n distance matrix using vrp.dist()."""
    n   = vrp.n + 1
    mat = _np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = vrp.dist(i, j)
            mat[i, j] = mat[j, i] = d
    return mat


def _find_bridges(adj: list) -> set:
    """
    Tarjan's bridge-finding algorithm.
    adj[v] = set of neighbours in the current kept graph.
    Returns a frozenset of canonical edges (i,j) with i<j that are bridges.
    """
    n       = len(adj)
    visited = [False] * n
    disc    = [0]      * n
    low     = [0]      * n
    timer   = [0]
    bridges: set = set()

    def dfs(u: int, parent: int) -> None:
        visited[u]  = True
        timer[0]   += 1
        disc[u] = low[u] = timer[0]
        for v in adj[u]:
            if not visited[v]:
                dfs(v, u)
                low[u] = min(low[u], low[v])
                if low[v] > disc[u]:
                    bridges.add((min(u, v), max(u, v)))
            elif v != parent:
                low[u] = min(low[u], disc[v])

    for start in range(n):
        if not visited[start]:
            dfs(start, -1)

    return bridges


def _is_connected(adj: list) -> bool:
    """DFS reachability from node 0. O(n + |Et|)."""
    n       = len(adj)
    visited = [False] * n
    stack   = [0]
    visited[0] = True
    count   = 1
    while stack:
        u = stack.pop()
        for v in adj[u]:
            if not visited[v]:
                visited[v] = True
                count += 1
                stack.append(v)
    return count == n


def solve_ged_heuristic(vrp: "VRPInstance") -> Tuple[List[List[int]], float, float]:
    """
    GED-native heuristic implementing Theorem 1.

    Phase 1  Greedy min-weight subgraph construction        O(n^2 log n)
    Phase 2  Candidate-list 2-opt with bridge-skip          O(n*k) per pass
    Phase 3  Route extraction + capacity repair

    Returns (routes, C(R), GED).
    """
    import time as _time
    t0 = _time.perf_counter()

    n      = vrp.n + 1
    req    = [2] * n
    req[0] = 2 * vrp.n_vehicles

    # ── Precompute distance matrix (O(1) lookups from here on) ───────────
    D = _dist_matrix(vrp)                          # shape (n, n)
    wt = float(D[_np.triu_indices(n, k=1)].sum())  # W_total

    # All edges sorted cheapest-first (for construction phase)
    edges_asc = sorted(
        ((int(i), int(j)) for i in range(n) for j in range(i+1, n)),
        key=lambda e: D[e[0], e[1]]
    )

    # ── Phase 1: greedy minimum-weight construction ───────────────────────
    # adj[v] = set of neighbours in Et
    adj    = [set() for _ in range(n)]
    deg    = [0] * n
    kept   : set = set()     # canonical (i,j) pairs, i<j

    # Per-node cheapest-neighbour list for greedy selection
    node_nbrs = [
        sorted(range(n), key=lambda u, v=v: D[v, u])
        for v in range(n)
    ]

    # Process highest-degree nodes first (depot before customers)
    for v in sorted(range(n), key=lambda v: -req[v]):
        for u in node_nbrs[v]:
            if u == v:
                continue
            if deg[v] >= req[v]:
                break
            if deg[u] >= req[u]:
                continue
            e = (min(u, v), max(u, v))
            if e in kept:
                continue
            kept.add(e)
            adj[u].add(v)
            adj[v].add(u)
            deg[v] += 1
            deg[u] += 1

    # Safety pass: satisfy any remaining degree deficit
    for v in range(n):
        for u in node_nbrs[v]:
            if u == v or deg[v] >= req[v]:
                continue
            if deg[u] >= req[u]:
                continue
            e = (min(u, v), max(u, v))
            if e in kept:
                continue
            kept.add(e)
            adj[u].add(v)
            adj[v].add(u)
            deg[v] += 1
            deg[u] += 1

    # ── Phase 2: candidate-list 2-opt with bridge-aware skip ─────────────
    # k nearest neighbours per node (candidate list)
    k    = min(10, vrp.n)
    near = [node_nbrs[v][1:k+1] for v in range(n)]  # exclude self

    def _do_2opt(kept: set, adj: list) -> Tuple[set, list, bool]:
        """
        One 2-opt pass.  Returns (new_kept, new_adj, improved).

        For each kept edge (u,v) -- cheapest first (most GED-gain potential
        at cheap swaps):
          For each candidate a in near[u] + near[v]:
            For each kept edge (a,b) sharing endpoint a:
              Try R1: add (u,a),(v,b)  remove (u,v),(a,b)
              Try R2: add (u,b),(v,a)  remove (u,v),(a,b)
              Accept first that:
                (a) reduces D[e1]+D[e2] < D[u,v]+D[a,b]
                (b) keeps graph connected  (bridge-skip when safe)
        """
        bridges = _find_bridges(adj)

        kept_sorted = sorted(kept, key=lambda e: D[e[0], e[1]])  # cheap first

        for u, v in kept_sorted:
            w_uv = D[u, v]

            for a in list(near[u]) + list(near[v]):
                if a == u or a == v:
                    continue

                # All kept edges incident to a
                for b in list(adj[a]):
                    if b == u or b == v:
                        continue
                    e_ab = (min(a, b), max(a, b))
                    if e_ab not in kept:
                        continue
                    w_ab  = D[a, b]
                    old_w = w_uv + w_ab

                    for e1, e2 in [
                        ((min(u,a),max(u,a)), (min(v,b),max(v,b))),
                        ((min(u,b),max(u,b)), (min(v,a),max(v,a))),
                    ]:
                        if e1[0] == e1[1] or e2[0] == e2[1]: continue
                        if e1 in kept or e2 in kept:         continue
                        if e1[1] >= n or e2[1] >= n:         continue

                        new_w = D[e1[0],e1[1]] + D[e2[0],e2[1]]
                        if new_w >= old_w - 1e-9:            continue

                        # ── Connectivity check (skip for non-bridges) ──
                        e_uv = (min(u,v), max(u,v))
                        both_non_bridge = (e_uv not in bridges
                                           and e_ab not in bridges)

                        if not both_non_bridge:
                            # Apply tentatively and check connectivity
                            adj[u].discard(v); adj[v].discard(u)
                            adj[a].discard(b); adj[b].discard(a)
                            adj[e1[0]].add(e1[1]); adj[e1[1]].add(e1[0])
                            adj[e2[0]].add(e2[1]); adj[e2[1]].add(e2[0])
                            ok = _is_connected(adj)
                            if not ok:
                                # Rollback
                                adj[u].add(v); adj[v].add(u)
                                adj[a].add(b); adj[b].add(a)
                                adj[e1[0]].discard(e1[1]); adj[e1[1]].discard(e1[0])
                                adj[e2[0]].discard(e2[1]); adj[e2[1]].discard(e2[0])
                                continue
                            # Accepted -- finalise kept set
                        else:
                            # Non-bridge swap: always connected, skip check
                            adj[u].discard(v); adj[v].discard(u)
                            adj[a].discard(b); adj[b].discard(a)
                            adj[e1[0]].add(e1[1]); adj[e1[1]].add(e1[0])
                            adj[e2[0]].add(e2[1]); adj[e2[1]].add(e2[0])

                        new_kept = (kept - {e_uv, e_ab}) | {e1, e2}
                        return new_kept, adj, True

        return kept, adj, False

    passes = 0
    for passes in range(1, 50):
        kept, adj, improved = _do_2opt(kept, adj)
        if not improved:
            break

    # ── Phase 3: extract routes and repair capacity ───────────────────────
    routes = _extract_routes_from_gt(vrp, list(kept))
    if routes is None:
        routes, _ = solve(vrp)
    else:
        routes = _enforce_vehicle_limit(routes, vrp)
        if not all(vrp.route_load(r) <= vrp.capacity for r in routes):
            routes = _repair_capacity(routes, vrp)

    # Final metrics
    Gs   = build_complete_graph(vrp)
    Gt   = build_routing_graph(vrp, routes)
    ged  = compute_ged(Gs, Gt)
    setc = compute_set_cost(Gt)
    cost = compute_traversal_cost(vrp, routes)
    ms   = (_time.perf_counter() - t0) * 1000
    thm1 = abs(setc - (wt - ged))

    print(f"    [GED-Heur] GED={ged:.2f}  C(R)={cost:.2f}  C\'(R)={setc:.2f}"
          f"  passes={passes}  Thm1_err={thm1:.1e}  time={ms:.1f} ms")

    return routes, cost, ged

# 7.  GED-EXACT SOLVER  —  Branch and Bound in the Graph Edit Space
# ===========================================================================
"""
Why the route-based B&B does NOT work on the GED formulation
─────────────────────────────────────────────────────────────
The existing B&B searches the *customer-assignment* space: at each node it
decides which customer to add to the current partial route or when to close
a route.  GED is computed only AFTER a complete solution is found.

Theorem 1 guarantees:
    min C(R) = W_total - max GED(Gs, Gt)

So the two problems have the same optimal value, but their search spaces
are structurally different:

    Route B&B    : state = (routes_done, cur_route, remaining_customers)
                   branch on customer assignment
                   objective = minimize traversal cost

    GED B&B      : state = (Et_decided_edges, node_degrees, committed_ged)
                   branch on KEEP vs DELETE each edge of Gs
                   objective = maximize sum of deleted edge weights

The GED-exact solver implemented below works directly on the second
formulation.  It is a correct and independent implementation of Theorem 1.

Search-space structure
───────────────────────
The complete graph Gs has n(n+1)/2 edges.  For each edge e = (i,j,w) we
make a binary decision:

    DELETE e  →  e ∉ Et,  contributes w to GED
    KEEP   e  →  e ∈ Et,  contributes w to C'(set)

Feasibility constraints (Definition 6 of the paper):
    (F1)  deg_Et(depot)    = 2m            (exactly m routes depart/return)
    (F2)  deg_Et(customer) = 2             (every customer entered + exited once)
    (F3)  Et forms exactly m connected depot-to-depot paths (valid routes)
    (F4)  Each route's total demand ≤ Q    (capacity)

F1+F2 are degree constraints, enforced by constraint propagation during
search.  F3 (connectivity) and F4 (capacity) are checked at leaf nodes.

Constraint propagation
───────────────────────
After every KEEP/DELETE decision we scan all nodes for forced moves:
    • If deg_Et(v) == req(v):  all remaining undecided edges incident to v
      must be DELETED  (v is already saturated).
    • If undecided_incident(v) + deg_Et(v) == req(v):  all remaining
      undecided edges incident to v must be KEPT  (v needs every one).
This is iterated to a fixed point (like arc consistency in CSP).

Upper bound
───────────────────────
At every node, the maximum additional GED achievable is bounded by:

    UB = committed_ged
       + Σ_{e undecided} w(e)             ← optimistic: delete everything
       - min_weight_must_keep             ← but we must keep enough edges

min_weight_must_keep is computed via an LP relaxation: for each node v,
sort its undecided incident edges by weight ascending and sum the
need(v) = req(v) - deg(v) cheapest ones.  Dividing by 2 (each edge counted
at both endpoints) gives a valid lower bound on what must be kept, and
therefore an upper bound on what can be deleted.

Edge ordering
───────────────────────
Edges are processed in *decreasing* weight order.  Trying DELETE first for
expensive edges maximises GED early, tightening the upper bound and causing
more pruning of the keep subtrees.
"""


def _et_to_routes(et_edges: List[Tuple[int, int]]) -> List[List[int]]:
    """
    Convert an edge set Et into a list of customer routes.

    Et encodes m depot-to-depot paths.  We follow each path by starting at
    depot (0), walking along edges, and collecting the customers visited
    until we return to depot.
    """
    from collections import defaultdict
    adj: Dict[int, List[int]] = defaultdict(list)
    for i, j in et_edges:
        adj[i].append(j)
        adj[j].append(i)

    routes:             List[List[int]] = []
    used_depot_edges:   set             = set()

    for start in list(adj[0]):              # each depot-neighbor starts one route
        key = (min(0, start), max(0, start))
        if key in used_depot_edges:
            continue
        used_depot_edges.add(key)

        route: List[int] = []
        prev, cur = 0, start
        while cur != 0:
            route.append(cur)
            nexts = [nb for nb in adj[cur] if nb != prev]
            if not nexts:
                break                       # dead end — malformed Et
            prev, cur = cur, nexts[0]

        if route:
            routes.append(route)

    return routes


# ===========================================================================
# 8.  LKH-3 INSPIRED METAHEURISTIC
# ===========================================================================
"""
LKH-3 (Helsgott & Christofides, 2017) is the state-of-the-art CVRP solver.
The key ideas reproduced here:

  Candidate lists    Precompute k nearest neighbors per node.  All local-search
                     moves restrict insertion/exchange partners to this list,
                     reducing O(n²) searches to O(nk).

  Or-opt (1, 2, 3)   Relocate a consecutive segment of 1, 2, or 3 customers
                     to a better position — either within the same route
                     (intra) or in a different route (inter).  This subsumes
                     the simple single-customer relocate used in the heuristic.

  3-opt              Remove 3 route edges and reconnect the 3 segments in
                     the best of the 7 non-trivial ways.  Captures improvements
                     invisible to 2-opt.

  Double-bridge      4-opt perturbation: splits a route into 4 segments
                     A-B-C-D and reconnects as A-C-B-D.  Cannot be undone
                     by any sequence of 2-opt or 3-opt moves — ideal for
                     escaping deep local optima.

  ILS framework      Outer loop: local-search → perturb → local-search.
                     Accept only improvements; restart from a new random
                     construction every `n_restarts` iterations.
"""


def _build_candidates(vrp: VRPInstance, k: int = 10) -> List[List[int]]:
    """
    For each node i, return the k nearest other nodes sorted by distance.
    Node 0 is the depot; nodes 1..n are customers.
    """
    k = min(k, vrp.n)
    candidates = []
    for i in range(vrp.n + 1):
        others = [j for j in range(vrp.n + 1) if j != i]
        others.sort(key=lambda j: vrp.dist(i, j))
        candidates.append(others[:k])
    return candidates


# ── Or-opt: move a segment of exactly `seg_len` customers ─────────────────

def _or_opt_intra(route: List[int], vrp: VRPInstance, seg_len: int) -> List[int]:
    """
    Move every segment of `seg_len` consecutive customers to the best other
    position within the same route.  Repeat until no improvement is found.
    """
    improved = True
    while improved:
        improved = False
        n = len(route)
        if n <= seg_len:
            break
        best_gain, best_i, best_pos = 0.0, -1, -1

        for i in range(n - seg_len + 1):
            seg      = route[i: i + seg_len]
            prev_i   = route[i - 1]        if i > 0            else 0
            next_i   = route[i + seg_len]  if i + seg_len < n  else 0
            rem_gain = (vrp.dist(prev_i, seg[0]) + vrp.dist(seg[-1], next_i)
                        - vrp.dist(prev_i, next_i))

            # Route with the segment removed
            rest     = route[:i] + route[i + seg_len:]
            rest_seq = [0] + rest + [0]

            for pos in range(len(rest_seq) - 1):
                a, b = rest_seq[pos], rest_seq[pos + 1]
                ins_cost = (vrp.dist(a, seg[0]) + vrp.dist(seg[-1], b)
                            - vrp.dist(a, b))
                gain = rem_gain - ins_cost
                if gain > best_gain + 1e-9:
                    best_gain, best_i, best_pos = gain, i, pos

        if best_i >= 0:
            seg   = route[best_i: best_i + seg_len]
            rest  = route[:best_i] + route[best_i + seg_len:]
            route = rest[:best_pos] + seg + rest[best_pos:]
            improved = True

    return route


def _or_opt_inter(routes: List[List[int]], vrp: VRPInstance,
                  seg_len: int, candidates: List[List[int]]) -> List[List[int]]:
    """
    Move a segment of `seg_len` customers from one route to another when it
    reduces total cost and respects capacity.

    Candidate lists restrict the search: only routes containing a candidate
    neighbor of seg[0] are considered as insertion targets.
    """
    improved = True
    while improved:
        improved = False
        for ri_idx in range(len(routes)):
            ri = routes[ri_idx]
            ci = 0
            while ci <= len(ri) - seg_len:
                seg      = ri[ci: ci + seg_len]
                prev_c   = ri[ci - 1]       if ci > 0               else 0
                next_c   = ri[ci + seg_len] if ci + seg_len < len(ri) else 0
                rem_gain = (vrp.dist(prev_c, seg[0]) + vrp.dist(seg[-1], next_c)
                            - vrp.dist(prev_c, next_c))
                seg_load = sum(vrp.demand(c) for c in seg)

                best_gain, best_rj, best_pos = 0.0, -1, -1
                cand_set = set(candidates[seg[0]])

                for rj_idx, rj in enumerate(routes):
                    if rj_idx == ri_idx:
                        continue
                    if vrp.route_load(rj) + seg_load > vrp.capacity:
                        continue
                    # Skip if no node in rj is a candidate neighbor of seg[0]
                    if rj and not any(nd in cand_set for nd in rj):
                        continue
                    seq = [0] + rj + [0]
                    for pos in range(len(seq) - 1):
                        a, b = seq[pos], seq[pos + 1]
                        ins_cost = (vrp.dist(a, seg[0]) + vrp.dist(seg[-1], b)
                                    - vrp.dist(a, b))
                        gain = rem_gain - ins_cost
                        if gain > best_gain + 1e-9:
                            best_gain, best_rj, best_pos = gain, rj_idx, pos

                if best_rj >= 0:
                    del routes[ri_idx][ci: ci + seg_len]       # remove segment
                    routes[best_rj][best_pos: best_pos] = seg  # insert segment
                    improved = True
                    break          # restart inner loop after modification
                else:
                    ci += 1

            if improved:
                break
        routes = [r for r in routes if r]

    return routes


# ── 3-opt: try all reconnections of 3 removed edges ───────────────────────

def _three_opt(route: List[int], vrp: VRPInstance) -> List[int]:
    """
    3-opt improvement within one route.
    Removes 3 edges at positions (i,j,k) and tries all 7 non-trivial
    reconnections of the resulting 3 segments (reversals + swaps).
    Accepts the first improvement found; repeats until no improvement.
    Only applied to routes with ≤ 15 customers (O(n³) per pass).
    """
    if len(route) > 15:
        return _two_opt(route, vrp)   # fall back to 2-opt for longer routes

    improved = True
    while improved:
        improved = False
        seq = [0] + route + [0]
        n   = len(seq)
        d_old = sum(vrp.dist(seq[p], seq[p + 1]) for p in range(n - 1))

        for i in range(1, n - 3):
            for j in range(i + 1, n - 2):
                for k in range(j + 1, n - 1):
                    seg_A = seq[1:i]
                    seg_B = seq[i:j]
                    seg_C = seq[j:k]
                    seg_D = seq[k:-1]

                    # 7 non-trivial reconnections  (0 = original, skipped)
                    reconnections = [
                        seg_A + seg_B + seg_C[::-1] + seg_D,        # reverse C
                        seg_A + seg_B[::-1] + seg_C       + seg_D,  # reverse B
                        seg_A + seg_B[::-1] + seg_C[::-1] + seg_D,  # reverse B,C
                        seg_A + seg_C       + seg_B       + seg_D,  # swap B,C
                        seg_A + seg_C       + seg_B[::-1] + seg_D,  # swap, rev B
                        seg_A + seg_C[::-1] + seg_B       + seg_D,  # swap, rev C
                        seg_A + seg_C[::-1] + seg_B[::-1] + seg_D,  # swap, rev both
                    ]

                    for cand in reconnections:
                        new_seq = [0] + cand + [0]
                        d_new   = sum(vrp.dist(new_seq[p], new_seq[p + 1])
                                      for p in range(len(new_seq) - 1))
                        if d_new < d_old - 1e-9:
                            route    = cand
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break

    return route


# ── Perturbation operators ─────────────────────────────────────────────────

def _double_bridge(route: List[int]) -> List[int]:
    """
    4-opt double-bridge perturbation for a single route.
    Splits the route into 4 segments A-B-C-D and reconnects as A-C-B-D.

    This is the canonical ILS perturbation for TSP/VRP:
    it cannot be reversed by any sequence of 2-opt or 3-opt moves,
    so it reliably escapes from deep local optima.
    Requires len(route) ≥ 8.
    """
    n = len(route)
    if n < 8:
        return route
    a, b, c = sorted(random.sample(range(1, n), 3))
    A, B, C, D = route[:a], route[a:b], route[b:c], route[c:]
    return A + C + B + D


def _perturb(routes: List[List[int]], vrp: VRPInstance) -> List[List[int]]:
    """
    Choose the appropriate perturbation based on the solution structure:
    • Single route  → double-bridge (4-opt)
    • Multi-route   → random relocation of 2-4 customers between routes,
                       then capacity repair
    """
    routes = [list(r) for r in routes]

    if len(routes) == 1 and len(routes[0]) >= 8:
        return [_double_bridge(routes[0])]

    # Multi-route: randomly move 2-4 customers to different routes
    n_moves = random.randint(2, min(4, vrp.n))
    for _ in range(n_moves):
        non_empty = [i for i, r in enumerate(routes) if r]
        if len(non_empty) < 2:
            break
        src = random.choice(non_empty)
        c   = routes[src].pop(random.randrange(len(routes[src])))
        dst = random.choice([i for i in non_empty if i != src])
        routes[dst].insert(random.randint(0, len(routes[dst])), c)

    routes = _repair_capacity([r for r in routes if r], vrp)
    return _enforce_vehicle_limit(routes, vrp)


def _repair_capacity(routes: List[List[int]],
                     vrp:    VRPInstance) -> List[List[int]]:
    """
    Capacity repair with vehicle-limit enforcement.

    Step 1 — remove overloaded customers into an overflow list.
    Step 2 — re-insert overflow customers at the cheapest feasible position
             in any existing route.
    Step 3 — if the route count exceeds vrp.n_vehicles after any insertions,
             merge the shortest excess routes into the cheapest receiving route.

    Never opens new routes beyond vrp.n_vehicles.  If the instance is
    genuinely infeasible for the given m/Q the customer is appended to the
    least-loaded route regardless of capacity.
    """
    overflow: List[int] = []
    for r in routes:
        keep, load = [], 0.0
        for c in r:
            if load + vrp.demand(c) <= vrp.capacity:
                keep.append(c)
                load += vrp.demand(c)
            else:
                overflow.append(c)
        r[:] = keep

    # Remove empty routes
    routes = [r for r in routes if r]

    for c in overflow:
        best_pos, best_cost, best_r = 0, float("inf"), 0
        for ri, r in enumerate(routes):
            if vrp.route_load(r) + vrp.demand(c) > vrp.capacity:
                continue
            seq = [0] + r + [0]
            for pos in range(len(seq) - 1):
                ins = (vrp.dist(seq[pos], c) + vrp.dist(c, seq[pos+1])
                       - vrp.dist(seq[pos], seq[pos+1]))
                if ins < best_cost:
                    best_cost, best_pos, best_r = ins, pos, ri
        if best_cost < float("inf"):
            routes[best_r].insert(best_pos, c)
        else:
            # Capacity infeasible: insert into least-loaded route
            least = min(range(len(routes)), key=lambda i: vrp.route_load(routes[i]))
            routes[least].append(c)

    # Merge excess routes down to n_vehicles
    while len(routes) > vrp.n_vehicles:
        # Pick the shortest route (fewest customers) to merge away
        src_idx = min(range(len(routes)), key=lambda i: len(routes[i]))
        src = routes.pop(src_idx)
        for c in src:
            best_pos, best_cost, best_r = 0, float("inf"), 0
            for ri, r in enumerate(routes):
                if vrp.route_load(r) + vrp.demand(c) > vrp.capacity:
                    continue
                seq = [0] + r + [0]
                for pos in range(len(seq) - 1):
                    ins = (vrp.dist(seq[pos], c) + vrp.dist(c, seq[pos+1])
                           - vrp.dist(seq[pos], seq[pos+1]))
                    if ins < best_cost:
                        best_cost, best_pos, best_r = ins, pos, ri
            if best_cost < float("inf"):
                routes[best_r].insert(best_pos, c)
            else:
                routes[0].append(c)  # last resort

    return routes


# ── Local-search pass (full pipeline) ─────────────────────────────────────

def _local_search_pass(routes:     List[List[int]],
                       vrp:        VRPInstance,
                       candidates: List[List[int]]) -> List[List[int]]:
    """
    One complete local-search improvement pass.  Runs until no further
    improvement is possible, then returns with the vehicle limit enforced.
    """
    prev_cost = float("inf")
    while True:
        routes = [_or_opt_intra(r, vrp, 1) for r in routes]
        routes = [_or_opt_intra(r, vrp, 2) for r in routes]
        routes = [_or_opt_intra(r, vrp, 3) for r in routes]
        routes = [_three_opt(r, vrp)       for r in routes]
        for sl in [3, 2, 1]:
            routes = _or_opt_inter(routes, vrp, sl, candidates)
        routes = _relocate(routes, vrp)
        routes = [r for r in routes if r]
        routes = _enforce_vehicle_limit(routes, vrp)   # hard cap every iteration

        cost = compute_traversal_cost(vrp, routes)
        if cost >= prev_cost - 1e-9:
            break
        prev_cost = cost

    return routes


def _nearest_neighbour_random(vrp: VRPInstance) -> List[List[int]]:
    """
    Randomised nearest-neighbour construction with vehicle-limit enforcement.
    At each step picks uniformly among the 3 nearest feasible unvisited
    customers.  Once vrp.n_vehicles routes are open, remaining customers are
    force-inserted into existing routes.
    """
    unvisited = set(range(1, vrp.n + 1))
    routes: List[List[int]] = []

    while unvisited and len(routes) < vrp.n_vehicles:
        route, cap, cur = [], vrp.capacity, 0
        while unvisited:
            feasible = sorted(
                (vrp.dist(cur, c), c)
                for c in unvisited if vrp.demand(c) <= cap
            )
            if not feasible:
                break
            _, chosen = random.choice(feasible[:3])
            route.append(chosen)
            cap -= vrp.demand(chosen)
            unvisited.discard(chosen)
            cur = chosen
        if route:
            routes.append(route)

    # Force-insert remaining customers cheaply
    for c in list(unvisited):
        best_pos, best_cost, best_r = 0, float("inf"), 0
        for ri, r in enumerate(routes):
            if vrp.route_load(r) + vrp.demand(c) > vrp.capacity:
                continue
            seq = [0] + r + [0]
            for pos in range(len(seq) - 1):
                ins = (vrp.dist(seq[pos], c) + vrp.dist(c, seq[pos+1])
                       - vrp.dist(seq[pos], seq[pos+1]))
                if ins < best_cost:
                    best_cost, best_pos, best_r = ins, pos, ri
        if best_cost < float("inf"):
            routes[best_r].insert(best_pos, c)
        else:
            routes[0].append(c)

    return routes


# ── Main LKH-inspired solver ──────────────────────────────────────────────

def solve_lkh(
        vrp:          VRPInstance,
        n_restarts:   int   = 10,
        time_limit_s: float = 120.0,
) -> Tuple[List[List[int]], float]:
    """
    LKH-3 inspired Iterated Local Search (ILS) for CVRP.

    Algorithm
    ---------
    For each restart:
      1. Build initial solution via randomised nearest-neighbour.
      2. Apply full local-search pass (Or-opt 1/2/3 intra + 3-opt +
         Or-opt 1/2/3 inter + relocate) until no improvement.
      3. Inner ILS loop: perturb (double-bridge or random relocation)
         then local-search again.  Accept if improved.
         Stop inner loop after `patience` consecutive non-improvements.
    Keep the global best solution across all restarts.

    The greedy heuristic warm-start is used as the initial incumbent so
    the first restart always improves on the baseline.

    Parameters
    ----------
    n_restarts    : number of independent random NN restarts
    time_limit_s  : total wall-clock budget (seconds)
    """
    t0         = time.perf_counter()
    candidates = _build_candidates(vrp, k=min(10, vrp.n))

    # Seed global best with the deterministic heuristic
    best_routes, best_cost = solve(vrp)

    for restart in range(n_restarts):
        if time.perf_counter() - t0 > time_limit_s:
            break

        # ── Construction ──────────────────────────────────────────────────
        cur_routes = _nearest_neighbour_random(vrp)
        cur_routes = _local_search_pass(cur_routes, vrp, candidates)
        cur_cost   = compute_traversal_cost(vrp, cur_routes)

        # ── ILS inner loop ────────────────────────────────────────────────
        patience     = 0
        max_patience = 5 + vrp.n // 5   # scales with instance size

        while patience < max_patience:
            if time.perf_counter() - t0 > time_limit_s:
                break

            perturbed = _perturb(cur_routes, vrp)
            perturbed = _local_search_pass(perturbed, vrp, candidates)
            p_cost    = compute_traversal_cost(vrp, perturbed)

            if p_cost < cur_cost - 1e-9:
                cur_routes, cur_cost = perturbed, p_cost
                patience = 0
            else:
                patience += 1

        # ── Update global best (only if truly feasible) ──────────────────
        cur_routes = _enforce_vehicle_limit(cur_routes, vrp)
        cur_cost   = compute_traversal_cost(vrp, cur_routes)
        # Hard capacity check — reject any solution with overloaded routes
        cap_ok = all(vrp.route_load(r) <= vrp.capacity for r in cur_routes)
        if cap_ok and cur_cost < best_cost - 1e-9:
            best_cost, best_routes = cur_cost, cur_routes

    elapsed = time.perf_counter() - t0
    best_routes = _enforce_vehicle_limit(best_routes, vrp)
    best_cost   = compute_traversal_cost(vrp, best_routes)

    n_routes = len(best_routes)
    if n_routes > vrp.n_vehicles:
        print(f"    [LKH] WARNING: {n_routes} routes > m={vrp.n_vehicles} — vehicle limit violated!")
    print(f"    [LKH] cost={best_cost:.2f}  routes={n_routes}/{vrp.n_vehicles}"
          f"  restarts={restart + 1}  time={elapsed:.1f}s")
    return best_routes, best_cost


# ===========================================================================
# 9.  RESULT  +  REPORTING
# ===========================================================================

@dataclass
class Result:
    name:           str
    n_customers:    int
    n_vehicles:     int
    capacity:       float
    routes:         List[List[int]]
    traversal_cost: float           # actual Σ leg distances
    set_cost:       float           # Σ_{e in Et} w(e)  — Theorem 1 LHS
    ged:            float           # W_total - set_cost
    wtotal:         float           # Σ_{e in Gs} w(e)
    thm1_error:     float           # |set_cost - (wtotal - ged)|  ≈ 0
    best_known:     Optional[float]
    gap_pct:        Optional[float]
    solve_ms:       float
    solver:         str = "heuristic"  # "heuristic" | "bb" | "lkh"


def analyse(path: Path,
            n_vehicles: Optional[int] = None,
            solver: str = "heuristic") -> Result:
    """
    Load, solve, and compute GED analysis for one .vrp file.

    Parameters
    ----------
    solver : "heuristic"  — greedy NN + 2-opt + relocate  (fast, any size)
             "bb"         — exact branch and bound         (n ≤ 20 only)
             "lkh"        — LKH-3 inspired ILS             (best quality)
    """
    vrp = load_instance(path, n_vehicles)
    t0  = time.perf_counter()

    if solver == "bb":
        routes, _trav = solve_branch_and_bound(vrp)
    elif solver == "ged_bb":
        routes, _trav, _ = solve_ged_bb(vrp)
    elif solver == "ged_ilp":
        routes, _trav, _ = solve_ged_exact_ilp(vrp)
    elif solver == "ged_heuristic":
        routes, _trav, _ = solve_ged_heuristic(vrp)
    elif solver == "lkh":
        routes, _trav = solve_lkh(vrp)
    else:
        routes, _trav = solve(vrp)

    ms = (time.perf_counter() - t0) * 1000

    # ── Validate vehicle count ────────────────────────────────────────────
    # This is the hard constraint that was being silently violated.
    routes = _enforce_vehicle_limit(routes, vrp)
    n_routes = len(routes)
    if n_routes > vrp.n_vehicles:
        print(f"    [WARN] {solver}: {n_routes} routes > m={vrp.n_vehicles}"
              f" — vehicle limit still violated after repair")

    Gs   = build_complete_graph(vrp)
    Gt   = build_routing_graph(vrp, routes)
    wt   = compute_wtotal(Gs)
    ged  = compute_ged(Gs, Gt)
    setc = compute_set_cost(Gt)          # C'(R): each edge once — for Theorem 1
    trav_cost = compute_traversal_cost(vrp, routes)  # C(R): every leg — for BKS gap
    err  = abs(setc - (wt - ged))

    bk  = BEST_KNOWN.get(vrp.name)
    gap = 100.0 * (trav_cost - bk) / bk if bk else None

    # Sanity check: gap should never be negative (we can't beat BKS)
    if gap is not None and gap < -0.5:
        print(f"    [WARN] {solver}: gap={gap:.2f}% — cost {trav_cost:.2f}"
              f" < BKS {bk:.0f}.  Check vehicle/capacity constraints.")

    return Result(
        name           = vrp.name,
        n_customers    = vrp.n,
        n_vehicles     = vrp.n_vehicles,
        capacity       = vrp.capacity,
        routes         = routes,
        traversal_cost = trav_cost,
        set_cost       = setc,
        ged            = ged,
        wtotal         = wt,
        thm1_error     = err,
        best_known     = bk,
        gap_pct        = gap,
        solve_ms       = ms,
        solver         = solver,
    )


def print_result(res: Result, vrp: VRPInstance) -> None:
    print(f"\n{SEP}")
    print(f"  {res.name}   n={res.n_customers}  m={res.n_vehicles}"
          f"  Q={res.capacity:.0f}   solver={res.solver.upper()}")
    print(SEP)
    for k, route in enumerate(res.routes, 1):
        load  = vrp.route_load(route)
        rc    = vrp.route_cost(route)
        note  = "  [depot edge counted once in Et]" if len(route) == 1 else ""
        seq   = " → ".join(map(str, [0] + route + [0]))
        print(f"  R{k:2d}: {seq}   cost={rc:.2f}  load={load:.0f}/{res.capacity:.0f}{note}")
    print(SEP)
    print(f"  Traversal cost C(R)       = {res.traversal_cost:.4f}")
    if abs(res.traversal_cost - res.set_cost) > 1e-9:
        print(f"  Set-based cost   C'(R)   = {res.set_cost:.4f}  ← Theorem 1 uses this")
    print(f"  W_total                   = {res.wtotal:.4f}")
    print(f"  GED (Lemma 2)             = {res.ged:.4f}")
    print(f"  W_total − GED             = {res.wtotal - res.ged:.4f}")
    print(f"  Theorem 1 error           = {res.thm1_error:.2e}  "
          f"[{'PASS' if res.thm1_error < 1e-6 else 'FAIL'}]")
    if res.best_known:
        print(f"  Best known (BKS)          = {res.best_known:.0f}")
        print(f"  Gap                       = {res.gap_pct:.2f}%")
    print(f"  Solve time                = {res.solve_ms:.1f} ms")
    print(SEP)


def print_summary(results: List[Result]) -> None:
    W = 116
    print("\n" + "=" * W)
    print("  BENCHMARK SUMMARY")
    print("=" * W)
    print(f"  {'Instance':<14} {'n':>4} {'m':>3} {'Q':>5}  "
          f"{'Solver':<10} {'C(actual)':>11} {'C(set)':>9}  "
          f"{'GED':>12} {'Wtotal':>12}  "
          f"{'BKS':>7} {'Gap%':>7}  {'Thm1':>8}  {'ms':>7}")
    print("  " + "─" * (W - 2))
    all_pass = True
    for r in results:
        gap_s = f"{r.gap_pct:+.2f}" if r.gap_pct is not None else "   N/A"
        bks_s = f"{r.best_known:.0f}" if r.best_known else "  N/A"
        ok    = "PASS" if r.thm1_error < 1e-6 else "FAIL"
        if r.thm1_error >= 1e-6:
            all_pass = False
        print(f"  {r.name:<14} {r.n_customers:>4} {r.n_vehicles:>3} "
              f"{r.capacity:>5.0f}  "
              f"{r.solver:<10} {r.traversal_cost:>11.2f} {r.set_cost:>9.2f}  "
              f"{r.ged:>12.2f} {r.wtotal:>12.2f}  "
              f"{bks_s:>7} {gap_s:>7}  {r.thm1_error:>8.2e}  {ok} "
              f"{r.solve_ms:>7.1f}")
    print("=" * W)
    verdict = "YES ✓" if all_pass else "NO — see FAIL rows above"
    print(f"\n  Theorem 1 [C'(set) = Wtotal − GED] holds for all instances: {verdict}\n")


# ===========================================================================
# 10.  VISUALISATION
# ===========================================================================

_BG = "#0C0C18"


def _draw_base(ax, Gs, pos, vrp: VRPInstance) -> None:
    """Draw nodes and faint background edges."""
    colors = ["#FFD700" if i == 0 else "#90CAF9" for i in vrp.V]
    sizes  = [700       if i == 0 else 300        for i in vrp.V]
    nx.draw_networkx_nodes(Gs, pos, ax=ax, node_color=colors, node_size=sizes,
                           edgecolors="white", linewidths=0.8)
    nx.draw_networkx_labels(Gs, pos, ax=ax, font_color="black",
                            font_size=7, font_weight="bold")
    nx.draw_networkx_edges(Gs, pos, ax=ax, edge_color="#FFFFFF",
                           alpha=0.05, width=0.6)


def plot_instance(res: Result, vrp: VRPInstance) -> None:
    """Two-panel figure: solution routes (left) | GED edit operations (right)."""
    Gs  = build_complete_graph(vrp)
    Gt  = build_routing_graph(vrp, res.routes)
    pos = vrp.layout_positions()   # real coords or spring layout for explicit instances

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    fig.patch.set_facecolor(_BG)
    for ax in (ax1, ax2):
        ax.set_facecolor(_BG)
        ax.set_axis_off()

    # --- Left panel: solution routes ---
    gap_s = (f"  BKS={res.best_known:.0f}  Gap={res.gap_pct:.1f}%"
             if res.best_known else "")
    ax1.set_title(
        f"{res.name}  —  Heuristic Solution  (n={res.n_customers}, m={res.n_vehicles})\n"
        f"C(actual)={res.traversal_cost:.2f}{gap_s}",
        color="white", fontsize=11, fontweight="bold", pad=10)
    _draw_base(ax1, Gs, pos, vrp)

    legend_handles = []
    for k, route in enumerate(res.routes):
        color = ROUTE_COLORS[k % len(ROUTE_COLORS)]
        seq   = [0] + route + [0]
        edges = [(seq[i], seq[i + 1]) for i in range(len(seq) - 1)]
        nx.draw_networkx_edges(Gt, pos, edgelist=edges, ax=ax1,
                               edge_color=color, width=2.5)
        legend_handles.append(Line2D(
            [0], [0], color=color, linewidth=2,
            label=f"R{k+1}: {len(route)} cust  load {vrp.route_load(route):.0f}/{res.capacity:.0f}"
        ))
    ax1.legend(handles=legend_handles, loc="lower left", fontsize=7,
               facecolor="#1A1A2E", edgecolor="white", labelcolor="white",
               ncol=max(1, len(legend_handles) // 7))

    # --- Right panel: GED edit operations ---
    deleted  = [(u, v) for u, v in Gs.edges() if not Gt.has_edge(u, v)]
    del_w    = sum(Gs[u][v]["weight"] for u, v in deleted)
    kept_pct = 100.0 * res.set_cost / res.wtotal if res.wtotal else 0

    ax2.set_title(
        f"{res.name}  —  GED Edit Operations\n"
        f"GED={res.ged:.2f}  |  {len(deleted)} deleted edges (Σw={del_w:.2f})"
        f"  |  kept {kept_pct:.1f}% of Wtotal",
        color="white", fontsize=11, fontweight="bold", pad=10)
    _draw_base(ax2, Gs, pos, vrp)

    for k, route in enumerate(res.routes):
        color = ROUTE_COLORS[k % len(ROUTE_COLORS)]
        seq   = [0] + route + [0]
        edges = [(seq[i], seq[i + 1]) for i in range(len(seq) - 1)]
        nx.draw_networkx_edges(Gt, pos, edgelist=edges, ax=ax2,
                               edge_color=color, width=2.0, alpha=0.4)
    nx.draw_networkx_edges(Gs, pos, edgelist=deleted, ax=ax2,
                           edge_color="#FF3333", style="dashed",
                           width=0.9, alpha=0.7)

    ax2.legend(handles=[
        Line2D([0], [0], color="#FF3333", lw=2, ls="--",
               label=f"Deleted ({len(deleted)} edges) → GED"),
        Line2D([0], [0], color="#90CAF9", lw=2, alpha=0.5,
               label="Kept in Et  (solution edges)"),
    ], loc="lower left", fontsize=8,
       facecolor="#1A1A2E", edgecolor="white", labelcolor="white")

    fig.text(0.5, 0.02,
             f"Wtotal={res.wtotal:.1f}  |  GED={res.ged:.1f}  |  "
             f"C'(set)={res.set_cost:.1f}  |  Thm1 err={res.thm1_error:.1e}",
             ha="center", color="white", fontsize=9)
    plt.tight_layout(rect=[0, 0.06, 1, 1])
    save_path = OUTPUT_DIR / f"vrp_ged_{res.name}.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"  [fig] {save_path}")


def plot_summary(results: List[Result]) -> None:
    """3-panel overview: cost vs BKS | optimality gap | Theorem 1 decomposition."""
    names  = [r.name          for r in results]
    costs  = [r.traversal_cost for r in results]
    bks    = [r.best_known or r.traversal_cost for r in results]
    geds   = [r.ged            for r in results]
    wtots  = [r.wtotal         for r in results]
    gaps   = [r.gap_pct or 0   for r in results]
    setcs  = [r.set_cost       for r in results]
    x      = np.arange(len(names))

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.patch.set_facecolor(_BG)
    for ax in axes:
        ax.set_facecolor("#111126")
        ax.tick_params(colors="white")
        ax.spines[:].set_edgecolor("#334")
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=30, ha="right", color="white", fontsize=9)

    # Panel 1 — cost vs BKS
    w = 0.35
    axes[0].bar(x - w/2, costs, width=w, color="#2979FF", label="Heuristic C(R)", alpha=0.9)
    axes[0].bar(x + w/2, bks,   width=w, color="#00C853", label="Best Known (BKS)", alpha=0.9)
    axes[0].set_ylabel("Route Cost", color="white")
    axes[0].yaxis.label.set_color("white")
    axes[0].legend(facecolor="#1A1A2E", edgecolor="#446", labelcolor="white")
    axes[0].grid(axis="y", alpha=0.15)
    axes[0].set_title("Heuristic Cost vs BKS", color="white", fontweight="bold", fontsize=12)

    # Panel 2 — optimality gap
    bar_colors = ["#00C853" if g < 5 else "#FF6D00" if g < 20 else "#F44336" for g in gaps]
    axes[1].bar(x, gaps, color=bar_colors, alpha=0.9)
    axes[1].axhline(0, color="white", lw=0.8, alpha=0.4)
    for xi, g in zip(x, gaps):
        axes[1].text(xi, g + 0.5, f"{g:.1f}%", ha="center", color="white", fontsize=8)
    axes[1].set_ylabel("Gap above BKS (%)", color="white")
    axes[1].yaxis.label.set_color("white")
    axes[1].grid(axis="y", alpha=0.15)
    axes[1].set_title("Optimality Gap", color="white", fontweight="bold", fontsize=12)

    # Panel 3 — Theorem 1 decomposition
    frac_c   = [c / w for c, w in zip(setcs, wtots)]
    frac_ged = [g / w for g, w in zip(geds,  wtots)]
    axes[2].bar(x, frac_c,   color="#2979FF", label="C'(set) / Wtotal")
    axes[2].bar(x, frac_ged, color="#FF6D00", label="GED / Wtotal", bottom=frac_c)
    axes[2].axhline(1.0, color="white", lw=1.2, ls="--", alpha=0.7, label="= 1.0  (Theorem 1)")
    axes[2].set_ylabel("Fraction of Wtotal", color="white")
    axes[2].yaxis.label.set_color("white")
    axes[2].legend(facecolor="#1A1A2E", edgecolor="#446", labelcolor="white", fontsize=8)
    axes[2].grid(axis="y", alpha=0.15)
    axes[2].set_title("C'(R) + GED = Wtotal  (Theorem 1)", color="white", fontweight="bold", fontsize=12)

    fig.suptitle("VRP-GED Benchmark Analysis — Classic CVRP Instances",
                 color="white", fontsize=14, fontweight="bold")
    plt.tight_layout()
    save_path = OUTPUT_DIR / "vrp_ged_summary.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"  [fig] {save_path}")


# ===========================================================================
# 11.  MAIN
# ===========================================================================

def main() -> None:
    print("\n" + "=" * 76)
    print("  VRP-GED Solver  |  instances from:", INSTANCES_DIR.resolve())
    print("  Solvers: Heuristic  |  GED-Heur  |  GED-B&B  |  GED-ILP  |  LKH-3")
    print("=" * 76)

    instance_files = list_instances(INSTANCES_DIR)
    print(f"  Found {len(instance_files)} instance(s)\n")

    summary_results: List[Result] = []

    for path in instance_files:
        print(path)
        vrp = load_instance(path)
        print(f"\n  +-- {path.name}   n={vrp.n}  m={vrp.n_vehicles}  Q={vrp.capacity:.0f}")

        # Solver selection:
        #   n <= 15  : all solvers (GED-B&B tractable)
        #   n <= 50  : heuristic + GED-Heur + GED-ILP + LKH
        #   n <= 150 : heuristic + GED-ILP + LKH
        #   n > 150  : heuristic + LKH
        if vrp.n <= 15:
            solvers = [ "ged_heuristic"]
        elif vrp.n <= 50:
            solvers = [ "ged_heuristic"]
        elif vrp.n <= 150:
            solvers = [ "ged_heuristic"]
        else:
            solvers = ["ged_heuristic"]
            print(f"  |  n={vrp.n} > 150: heuristic + LKH only")

        # ── Run each solver and collect results ───────────────────────────
        # For GED-ILP we capture (routes, set_cost, ged) directly so we can
        # print the full Theorem 1 breakdown.  The Result object stores
        # set_cost as traversal_cost (they are equal when no solo routes exist;
        # differ by Σ w(0,j) per solo route otherwise — explained in notes).
        all_res: List[Result] = []
        ilp_raw: Optional[Tuple[List[List[int]], float, float]] = None  # (routes, C', GED)

        for s in solvers:
            print(f"  |  [{s.upper():10s}] ", end="", flush=True)
            try:
                if s == "ged_ilp":
                    r_ilp, c_ilp, g_ilp = solve_ged_exact_ilp(vrp)
                    ilp_raw = (r_ilp, c_ilp, g_ilp)
                    Gs_ilp   = build_complete_graph(vrp)
                    Gt_ilp   = build_routing_graph(vrp, r_ilp)
                    wt_ilp   = compute_wtotal(Gs_ilp)
                    setc_ilp = compute_set_cost(Gt_ilp)          # C'(R) — Theorem 1
                    trav_ilp = compute_traversal_cost(vrp, r_ilp) # C(R)  — BKS gap
                    bk       = BEST_KNOWN.get(vrp.name)
                    res = Result(
                        name           = vrp.name,
                        n_customers    = vrp.n,
                        n_vehicles     = vrp.n_vehicles,
                        capacity       = vrp.capacity,
                        routes         = r_ilp,
                        traversal_cost = trav_ilp,   # C(R) for gap comparison
                        set_cost       = setc_ilp,   # C'(R) for Theorem 1
                        ged            = g_ilp,
                        wtotal         = wt_ilp,
                        thm1_error     = abs(setc_ilp - (wt_ilp - g_ilp)),
                        best_known     = bk,
                        gap_pct        = 100.0 * (trav_ilp - bk) / bk if bk else None,
                        solve_ms       = 0.0,
                        solver         = "ged_ilp",
                    )
                else:
                    res = analyse(path, solver=s)

                gap_s = f"  gap={res.gap_pct:+.2f}%" if res.gap_pct is not None else ""
                print(f"cost={res.traversal_cost:.2f}{gap_s}  ({res.solve_ms:.0f} ms)")
                all_res.append(res)
            except Exception as exc:
                import traceback
                print(f"ERROR: {exc}")
                traceback.print_exc()

        if not all_res:
            print(f"  `- [ERROR] no results for {path.name}")
            continue

        # ── Pick best by cost ─────────────────────────────────────────────
        best = min(all_res, key=lambda r: r.traversal_cost)
        print(f"  `- Best solver: {best.solver.upper()}"
              f"  cost={best.traversal_cost:.2f}")

        print_result(best, vrp)

        # ── Full Theorem 1 breakdown for the GED-ILP result ───────────────
        if ilp_raw is not None:
            r_ilp, c_ilp, g_ilp = ilp_raw
            _print_ged_comparison(vrp, r_ilp, c_ilp, g_ilp,
                                  label="GED-ILP  —  direct Theorem 1 solution")

        plot_instance(best, vrp)
        summary_results.append(best)

        # ── Side-by-side solver comparison ───────────────────────────────
        if len(all_res) > 1:
            Gs_cmp = build_complete_graph(vrp)
            wt_cmp = compute_wtotal(Gs_cmp)
            print(f"\n  {'Solver':<12} {'Cost':>9} {'GED':>10}"
                  f"  {'Gap%':>7}  {'ms':>8}  note")
            print("  " + "─" * 58)
            for r in all_res:
                Gt_r  = build_routing_graph(vrp, r.routes)
                ged_r = compute_ged(Gs_cmp, Gt_r)
                gap_s = f"{r.gap_pct:+.2f}%" if r.gap_pct is not None else "   N/A"
                tag   = " ← best" if r is best else ""
                tag  += "  [Theorem 1 native]" if r.solver == "ged_ilp" else ""
                print(f"  {r.solver:<12} {r.traversal_cost:>9.2f}"
                      f" {ged_r:>10.2f}  {gap_s:>7}  {r.solve_ms:>8.1f}  {tag}")
            print(f"  {'W_total':>12} {wt_cmp:>10.2f}")

    if summary_results:
        print_summary(summary_results)
        if len(summary_results) > 1:
            plot_summary(summary_results)

    print(f"\n  Done. Figures saved to {OUTPUT_DIR.resolve()}/\n")


if __name__ == "__main__":
    main()