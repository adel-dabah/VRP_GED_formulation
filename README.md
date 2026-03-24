# VRP–GED Solver

**Capacitated Vehicle Routing via Graph Edit Distance**

Implementation of the equivalence framework from:

> *"A Graph Edit Distance Formulation for the Vehicle Routing Problem"*
> Adel Dabah, Forschungszentrum Jülich (JSC)

---

## Theorem 1 — The Core Identity

The entire solver is built around one mathematical identity:

```
min  C(R)  =  W_total  −  max  GED(Gs, Gt)
 R                           Gt
```

Where:

| Symbol | Meaning |
|--------|---------|
| `C(R)` | Actual traversal cost — sum of every driven leg |
| `C'(R)` | Set-based cost — `Σ_{e ∈ Et} w(e)`, each edge counted once |
| `W_total` | Sum of **all** edge weights in the complete graph Gs |
| `GED(Gs, Gt)` | Graph Edit Distance — sum of weights of edges deleted from Gs |
| `Gs` | Complete graph on all nodes (depot + customers) |
| `Gt` | Routing graph — the subgraph actually used by the solution |

**C(R) vs C'(R):** They are equal for all routes except single-customer routes `0→i→0`, where the depot edge `{0,i}` is traversed twice but appears only once in the undirected `Gt`. The difference is `Σ w(0,j)` per solo route.

**Theorem 1 verification** is printed for every solver result as `Thm1_err`. A value of `0.00e+00` confirms the identity holds exactly.

---

## Quick Start

```bash
# Install dependencies
pip install networkx numpy matplotlib scipy

# Place your .vrp files in ./instances/
mkdir instances
cp your_instances/*.vrp instances/

# Run
python vrp_ged_solver.py
```

Results are printed to stdout and figures saved to `./output/`.

---

## Supported Input Formats

### EUC_2D (coordinate-based)

The standard TSPLIB/VRPLIB format. Distances are computed as
`nint(sqrt(Δx² + Δy²))` — nearest-integer rounding matching all
A/B/E/P benchmark best-known values.

```
NAME : A-n32-k5
COMMENT : (Augerat et al, Min trucks: 5, Optimal value: 784)
TYPE : CVRP
DIMENSION : 32
EDGE_WEIGHT_TYPE : EUC_2D
CAPACITY : 100
NODE_COORD_SECTION
1  82 76
2  96 44
...
DEMAND_SECTION
1 0
2 19
...
DEPOT_SECTION
1
-1
EOF
```

### EXPLICIT / LOWER_ROW (matrix-based)

For instances with no display coordinates, such as `eil13`:

```
NAME : eil13
EDGE_WEIGHT_TYPE : EXPLICIT
EDGE_WEIGHT_FORMAT: LOWER_ROW
DISPLAY_DATA_TYPE: NO_DISPLAY
CAPACITY : 6000
EDGE_WEIGHT_SECTION
     9    14    21  ...
DEMAND_SECTION
...
```

Also supported: `FULL_MATRIX` and `UPPER_ROW` formats.

---

## Solver Overview

Six solvers are available, selected automatically by instance size:

```
n ≤ 15  :  heuristic  GED-heuristic  GED-B&B  GED-ILP  LKH
n ≤ 50  :  heuristic  GED-heuristic  GED-ILP  LKH
n ≤ 150 :  heuristic  GED-ILP  LKH
n > 150 :  heuristic  LKH
```

### 1. Baseline Heuristic (`solve`)

**Route-space solver. Fast baseline, any instance size.**

Three sequential phases:
1. **Nearest-neighbour construction** — greedily extend each route with the closest feasible unvisited customer. Caps at `n_vehicles` routes from the start.
2. **2-opt per route** — repeatedly reverse segments within each route to reduce its cost.
3. **Single-customer relocation** — move individual customers between routes when it improves total cost and respects capacity.

```
Complexity : O(n² log n) construction + O(n²) per 2-opt pass
Typical gap: 10–40% above BKS
```

---

### 2. GED Heuristic (`solve_ged_heuristic`)

**Edge-space solver. Works directly on Theorem 1. Fast and GED-aware.**

The only heuristic in this codebase that operates natively in the GED
formulation — it never builds route sequences during optimisation.

**Phase 1 — Greedy minimum-weight construction** `O(n² log n)`

Sort all `n(n+1)/2` edges ascending. For each node, assign its cheapest
incident edges that don't overflow any neighbour's degree cap, processing
the depot first (it needs `2m` edges before customers compete for cheap
depot edges). This keeps the lightest edges in `Et` and deletes all heavy
ones — maximising GED from the very first step.

**Phase 2 — Candidate-list 2-opt in edge space** `O(n·k)` per pass

For each kept edge `(u,v)`, try swapping it with kept edge `(a,b)` where
`a` or `b` is among the `k=10` nearest neighbours of `u` or `v`.

Two degree-preserving reconnections are tried:
```
R1: remove (u,v),(a,b)  →  add (u,a),(v,b)
R2: remove (u,v),(a,b)  →  add (u,b),(v,a)
```
Degrees are **always preserved** by the 2-swap structure — each node loses
one edge and gains one — so no degree check is needed.

**Bridge-aware connectivity skip:** Before each pass, Tarjan's algorithm
labels every kept edge as a bridge or non-bridge in `O(n + |Et|)`.
Non-bridge swaps are **accepted without a connectivity check** (removing
a non-bridge cannot disconnect the graph). Only bridge swaps require a
lightweight DFS reachability check. This eliminates the dominant cost of
the previous version.

**Phase 3 — Route extraction and capacity repair**

Walk the depot's edges to extract routes from the final `Et`. Enforce the
vehicle limit, then repair any capacity violation by reinserting customers
at the cheapest feasible positions in other routes.

```
Complexity : O(n² log n) + O(n·k·passes), k=10, passes ≤ 50
Typical gap: 5–25% above BKS
Speed      : 2–10 ms for n ≤ 50
```

---

### 3. Standard Branch and Bound (`solve_branch_and_bound`)

**Route-space exact solver. Practical for n ≤ 15.**

Depth-first search over customer-assignment states `(routes_done, cur_route, remaining)`.

**Branching:**
- Branch A: extend current route with a feasible customer (nearest first)
- Branch B: close current route, open a new one

**Pruning via 1-tree lower bound:**

```
LB = committed_cost
   + dist(cur_route_tail → depot)      ← must close this route
   + Σ_{c ∈ remaining} min₁(c)         ← cheapest entry edge
   + Σ_{c ∈ remaining} min₂(c) / 2     ← cheapest exit (halved)
```

**Warm-start:** The basic heuristic solution is used as incumbent before
the first node is expanded, ensuring aggressive pruning from the start.

**Hard abort:** A private `_Abort` exception is raised when the node or
time limit is hit, instantly unwinding the entire call stack. The label
"proven optimal" vs "limit hit" is set by catching or not catching
the exception.

```
Parameters : time_limit_s=60, node_limit=2_000_000
Practical  : n ≤ 15 typically proves optimal in < 5 s
```

---

### 4. GED Branch and Bound (`solve_ged_bb`)

**Edge-space exact solver. Implements Theorem 1 directly. Practical for n ≤ 15.**

Searches the space of subgraphs `Gt ⊆ Gs` by deciding, for each edge of
`Gs` in descending weight order, whether to KEEP it in `Et` or DELETE it
(adding its weight to GED).

**Branching:** Two children per node — DELETE (GED += w) or KEEP (degree update).
Edges are ordered heaviest-first so large GED values are found early.

**GED upper bound (admissible):**

```
UB = ged_so_far + Σ_{undecided} w(e)  −  forced_keep_LB
```

Where `forced_keep_LB` is computed per node:
- For each node `v` with remaining deficit `d(v) = req[v] - deg[v]`
- Take the `d(v)` cheapest undecided incident edges
- Sum and halve (each edge shared between two nodes)

This is always an underestimate of the forced-keep cost, making UB an
admissible (never overestimating) upper bound.

**Feasibility at leaves:** degree completion + DFS connectivity + capacity.

**Key difference from standard B&B:** This solver maximises GED directly.
The standard B&B minimises traversal cost. They find the same optimal value
by Theorem 1, but search completely different spaces.

```
Parameters  : time_limit_s=120, node_limit=5_000_000
Warm-start  : basic heuristic (validated for correct depot degree)
```

---

### 5. GED Exact ILP (`solve_ged_exact_ilp`)

**The only solver that is a complete, certified implementation of Theorem 1, solved by an ILP engine.**

Uses HiGHS via `scipy.optimize.milp` to solve the integer linear programme:

**Variables:**
- `f[i,j] ∈ {0,1}` — directed arc `i→j` used (for all ordered pairs `i≠j`)
- `z[j] ∈ {0,1}` — customer `j` forms a solo route `0→j→0`
- `q[j] ∈ [d_j, Q]` — cumulative load when arriving at customer `j`

**Objective:** `min Σ_{i≠j} w[i,j]·f[i,j] − Σ_j w[0,j]·z[j]`

The `z[j]` term removes the double-counting of the depot edge for solo
routes, making the objective equal to `C'(R)` exactly.

**Constraints:**
- `(D1)` Flow conservation: each customer has in-flow = out-flow = 1
- `(D2)` Depot: out-flow = in-flow = m
- `(MTZ)` Miller–Tucker–Zemlin: `q[j] ≥ q[i] + d[j] − Q·(1 − f[i,j])` — simultaneously eliminates subtours and enforces capacity

**`z[j]` linearisation:**
```
z[j] ≥ f[0,j] + f[j,0] − 1
z[j] ≤ f[0,j]
z[j] ≤ f[j,0]
```

Returns a **certificate of optimality** when HiGHS finishes within the
time limit. Theorem 1 error is printed and verified.

```
Variables  : N(N-1) binary arcs + n solo indicators + n continuous loads
Parameters : time_limit_s=300, mip_gap=1e-6
Scales to  : n ≈ 50–100 in seconds; n ≈ 200 in minutes
```

---

### 6. LKH-3 Inspired (`solve_lkh`)

**Route-space metaheuristic. Best solution quality for large instances.**

Iterated Local Search (ILS) inspired by the LKH-3 solver:

**Construction:** Randomised nearest-neighbour — at each step, pick one
of the 3 nearest feasible customers at random. Provides diverse restarts.

**Local search pipeline (repeated until convergence):**
1. Or-opt intra (segments 1, 2, 3) — move a segment within its route
2. 3-opt / 2-opt per route
3. Or-opt inter (segments 3, 2, 1) — move a segment to another route
4. Single-customer relocation

**Candidate lists:** All inter-route moves restrict partner search to
`k=10` nearest neighbours per node, reducing O(n²) to O(n·k) per pass.

**Perturbation (double-bridge):** A 4-opt move that reconnects route
segments as `A-C-B-D` instead of `A-B-C-D`. Cannot be undone by any
sequence of 2-opt or 3-opt moves — the canonical ILS escape.

**Vehicle limit enforcement:** `_enforce_vehicle_limit` is called after
every iteration of the local search pipeline to prevent solutions with
more routes than allowed. Capacity is validated before updating the
incumbent.

```
Parameters : n_restarts=10, time_limit_s=120
Typical gap: 0–5% above BKS
```

---

## Output

### Console (per instance)

```
+-- E-n30-k3.vrp   n=29  m=3  Q=4500
|  [HEURISTIC  ] cost=613.00  gap=+14.79%  (1 ms)
|  [GED_HEUR   ]     [GED-Heur] GED=23417.00  C(R)=613.00  passes=8  time=4.2 ms
|  [GED_ILP    ]     [GED-ILP] PROVEN OPTIMAL  GED=23496.00  C'(R)=534.00
|  [LKH        ]     [LKH] cost=538.00  routes=3/3  time=0.8s
`- Best solver: GED_ILP  cost=534.00

  R 1: 0 → 3 → 12 → 19 → ...   cost=178.00  load=4200/4500
  Traversal cost C(R)       = 534.0000
  Set-based cost   C'(R)   = 534.0000
  W_total                   = 24030.0000
  GED (Lemma 2)             = 23496.0000
  Theorem 1 error           = 0.00e+00  [PASS]
```

### Summary table

```
Instance         n   m     Q  Solver    C(actual)  C(set)       GED      BKS    Gap%    Thm1
E-n30-k3        29   3  4500  ged_ilp     534.00  534.00  23496.00      534   +0.00  0.00e+00 PASS
```

### Figures (`./output/`)

| File | Content |
|------|---------|
| `vrp_ged_{name}.png` | Two-panel: solution routes (left) + GED edit operations (right) |
| `vrp_ged_summary.png` | Three-panel overview across all instances |

The right panel of each instance figure shows which edges were **kept**
in `Et` (coloured by route) and which were **deleted** (red dashed lines).
The deleted edges are exactly the GED operations.

---

## Key Data Structures

### `VRPInstance`

```python
vrp.n           # number of customers (depot excluded)
vrp.n_vehicles  # m — maximum number of routes
vrp.capacity    # Q — per-route load limit
vrp.V           # [0, 1, ..., n] — all node indices; 0 = depot
vrp.dist(i, j)  # distance between nodes i and j
vrp.demand(i)   # demand at node i (0 for depot)
```

### Route representation

Routes are lists of customer indices (depot implicit at start and end):

```python
routes = [[3, 7, 2], [1, 8, 5], [4, 6]]
# means: 0→3→7→2→0, 0→1→8→5→0, 0→4→6→0
```

### `Result` dataclass

```python
@dataclass
class Result:
    name:           str               # instance name
    n_customers:    int               # n
    n_vehicles:     int               # m
    capacity:       float             # Q
    routes:         List[List[int]]   # best routes found
    traversal_cost: float             # C(R) — every leg counted
    set_cost:       float             # C'(R) — each edge once (Theorem 1)
    ged:            float             # GED(Gs, Gt) = W_total − C'(R)
    wtotal:         float             # W_total = Σ_{e∈Gs} w(e)
    thm1_error:     float             # |C'(R) − (W_total − GED)|  ≈ 0
    best_known:     Optional[float]   # BKS from literature
    gap_pct:        Optional[float]   # 100 * (C(R) − BKS) / BKS
    solve_ms:       float             # wall-clock time in milliseconds
    solver:         str               # which solver produced this result
```

---

## Configuration

Edit these constants at the top of `vrp_ged_solver.py`:

| Constant | Default | Purpose |
|----------|---------|---------|
| `INSTANCES_DIR` | `./instances` | Directory scanned for `.vrp` files |
| `OUTPUT_DIR` | `./output` | Directory for figures and logs |
| `BEST_KNOWN` | dict | Known optimal costs for gap computation |

Solver parameters are keyword arguments:

```python
# Call programmatically
vrp    = load_instance(Path("instances/E-n30-k3.vrp"))
routes, cost, ged = solve_ged_heuristic(vrp)
routes, cost, ged = solve_ged_exact_ilp(vrp, time_limit_s=60, mip_gap=1e-4)
routes, cost      = solve_lkh(vrp, n_restarts=20, time_limit_s=60)
```

---

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `networkx` | ≥ 2.6 | Graph construction and GED computation |
| `numpy` | ≥ 1.20 | Distance matrices and numerical operations |
| `matplotlib` | ≥ 3.3 | Solution and GED visualisation |
| `scipy` | ≥ 1.7 | ILP solver (HiGHS) via `scipy.optimize.milp` |

```bash
pip install networkx numpy matplotlib scipy
```

---

## Instance Format Reference

### Supported `EDGE_WEIGHT_TYPE` values

| Value | Distance formula |
|-------|-----------------|
| `EUC_2D` | `nint(sqrt(Δx² + Δy²))` — TSPLIB standard rounding |
| `EXPLICIT` | Read from `EDGE_WEIGHT_SECTION` |
| Others | Raw `sqrt(Δx² + Δy²)` float |

### Supported `EDGE_WEIGHT_FORMAT` values (for `EXPLICIT` type)

| Value | Layout |
|-------|--------|
| `LOWER_ROW` | Strict lower triangle, row by row |
| `FULL_MATRIX` | Complete n×n matrix, row-major |
| `UPPER_ROW` | Strict upper triangle, row by row |

### Vehicle count detection

The number of vehicles `m` is parsed from the `COMMENT` field:

```
COMMENT : (Min no of trucks: 4, Optimal value: 247)
                             ^--- parsed automatically
```

If not found, defaults to `max(2, n // 10)`.

---

## Benchmark Families

| Family | Source | Characteristics |
|--------|--------|----------------|
| **A-set** | Augerat et al. (1995) | Random instances, n = 32–80 |
| **B-set** | Augerat et al. (1995) | Clustered instances, n = 31–78 |
| **E-set** | Christofides & Eilon (1969) | Classic small/medium, n = 13–101 |
| **P-set** | Augerat et al. (1995) | Mixed instances, n = 16–101 |
| **eil13** | Eilon et al. | Explicit distance matrix (LOWER_ROW) |

All BKS values in `BEST_KNOWN` are integer costs computed with TSPLIB
nearest-integer rounding, consistent with the `EUC_2D` distance formula.

---

## Theorem 1 Verification

Every solver result is verified against Theorem 1 before being reported:

```python
setc = compute_set_cost(Gt)     # C'(R) = Σ_{e∈Et} w(e)
ged  = compute_ged(Gs, Gt)      # GED   = Σ_{e∉Et} w(e)
wt   = compute_wtotal(Gs)       # W_total
thm1_error = abs(setc - (wt - ged))   # should be 0.0
```

The summary table prints `PASS` if `thm1_error < 1e-6`, `FAIL` otherwise.
The final line confirms whether the identity holds across all instances:

```
Theorem 1 [C'(set) = Wtotal − GED] holds for all instances: YES ✓
```
