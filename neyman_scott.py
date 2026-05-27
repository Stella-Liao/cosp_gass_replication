"""
Neyman–Scott Poisson Cluster Point Process with Parent–Child Membership
========================================================================
Faithfully reproduces the generative logic of
    pointpats.process.PoissonClusterPointProcess
from PySAL pointpats (read from source: pointpats/process.py), then
extends it to preserve parent–child membership for marked point-process
modelling.

KEY LOGIC matched from pointpats source:
    1. runif_in_circle  — rejection sampling in SQUARE [-r,r]², NOT sqrt(r)
    2. Parents drawn from BOUNDING BOX uniform (np.random.uniform)
    3. children_per_parent = ceil(n / n_parents)
    4. realize() generates all children, then np.random.shuffle
    5. draw()  filters by window polygon, re-calls realize() if needed
    6. Uses legacy np.random API (seeded via np.random.seed)

The ONLY modification: parent_id is tracked alongside each child point.

Reference call replicated:
    PoissonClusterPointProcess(squwin, 2000, 20, 15, 1,
                               conditioning=False, asPP=False)
"""

import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from shapely.geometry import Point, box, Polygon

# ---------------------------------------------------------------------------
# Step 0 — Window construction
# ---------------------------------------------------------------------------

def make_square_window(xmin=0, ymin=0, xmax=100, ymax=100):
    """Return a Shapely polygon matching the PySAL Window used in squwin."""
    return box(xmin, ymin, xmax, ymax)


# ---------------------------------------------------------------------------
# runif_in_circle — EXACT reproduction from pointpats/process.py
# Uses rejection sampling in square (NOT the sqrt-radius trick).
# ---------------------------------------------------------------------------

def runif_in_circle(n, radius=1.0, center=(0.0, 0.0), burn=2, verbose=False):
    """
    Generate n points within a circle of given radius.
    COPIED from pointpats.process.runif_in_circle — uses rejection
    sampling inside square [-r,r] × [-r,r].

    Returns
    -------
    good : ndarray (n, 2)
        Coordinates of generated points, shifted to center.
    """
    good = np.zeros((n, 2), float)
    c = 0
    r = radius
    r2 = r * r
    it = 0
    while c < n:
        x = np.random.uniform(-r, r, (burn * n, 1))
        y = np.random.uniform(-r, r, (burn * n, 1))
        ids = np.where(x * x + y * y <= r2)
        candidates = np.hstack((x, y))[ids[0]]
        nc = candidates.shape[0]
        need = n - c
        if nc > need:
            good[c:] = candidates[:need]
        else:
            good[c:c + nc] = candidates
        c += nc
        it += 1
    if verbose:
        print("Iterations: {}".format(it))
    return good + np.asarray(center)


# ---------------------------------------------------------------------------
# Step 1 — Generate parent locations
#   From bounding box uniform — matching pointpats realize() method.
# ---------------------------------------------------------------------------

def generate_parents(bbox, n_parents):
    """
    Sample n_parents from uniform on bounding box.
    Matches pointpats.process.PoissonClusterPointProcess.realize():
        pxs = np.random.uniform(l, r, (int(n/self.children), 1))
        pys = np.random.uniform(b, t, (int(n/self.children), 1))
    """
    l, b, r, t = bbox
    pxs = np.random.uniform(l, r, (n_parents, 1))
    pys = np.random.uniform(b, t, (n_parents, 1))
    return np.hstack((pxs, pys))  # (n_parents, 2)


# ---------------------------------------------------------------------------
# Step 2 — realize() equivalent WITH parent_id tracking
#   Matches pointpats realize() but keeps parent assignment.
# ---------------------------------------------------------------------------

def realize_with_ids(n, n_parents, children_per_parent, radius, bbox):
    """
    Equivalent to PoissonClusterPointProcess.realize(n) but returns
    parent_id alongside child coordinates.

    1. Draw n_parents from bbox uniform  (same as pointpats)
    2. For each parent, draw children_per_parent via runif_in_circle
    3. Stack all children
    4. Shuffle children AND parent_ids in unison (pointpats shuffles)

    Returns
    -------
    child_xy  : ndarray (n_parents * children_per_parent, 2)
    parent_id : ndarray (n_parents * children_per_parent,)
    parent_xy : ndarray (n_parents, 2)
    """
    parent_xy = generate_parents(bbox, n_parents)

    all_children = []
    all_ids = []
    for g, center in enumerate(parent_xy):
        kids = runif_in_circle(children_per_parent, radius, center)
        all_children.append(kids)
        all_ids.append(np.full(children_per_parent, g, dtype=int))

    child_xy = np.vstack(all_children)
    parent_id = np.concatenate(all_ids)

    # pointpats does np.random.shuffle(res) — shuffle in unison
    shuffle_idx = np.random.permutation(len(child_xy))
    child_xy = child_xy[shuffle_idx]
    parent_id = parent_id[shuffle_idx]

    return child_xy, parent_id, parent_xy


# ---------------------------------------------------------------------------
# Step 3 — draw() equivalent WITH parent_id tracking
#   Matches pointpats draw() loop: filter by window, re-call realize()
#   if not enough points land inside.
# ---------------------------------------------------------------------------

def draw_with_ids(n, n_parents, children_per_parent, radius, window_poly):
    """
    Equivalent to PointProcess.draw() but preserves parent_id.

    Calls realize_with_ids repeatedly until n points inside window
    are collected.  Each call to realize generates NEW parents (matching
    pointpats behaviour where realize() is called afresh each iteration).

    For the final output, parents are taken from the FIRST realization
    that contributed points (this is necessary since pointpats discards
    parent info entirely).

    Returns
    -------
    child_xy  : ndarray (n, 2)
    parent_id : ndarray (n,)
    parent_xy : ndarray (n_parents, 2) — from first realize call
    """
    bbox = window_poly.bounds  # (minx, miny, maxx, maxy)
    sample_xy = []
    sample_id = []
    all_parent_xy = []        # collect parents from ALL realize calls
    parent_offset = 0         # offset parent_ids across multiple realize calls

    c = 0
    while c < n:
        child_xy, parent_id, parent_xy = realize_with_ids(
            n, n_parents, children_per_parent, radius, bbox)

        all_parent_xy.append(parent_xy)

        # offset parent_ids for subsequent realize calls
        parent_id = parent_id + parent_offset

        # window filter — matching pointpats window.filter_contained
        for i in range(len(child_xy)):
            pt = Point(child_xy[i])
            if window_poly.contains(pt):
                sample_xy.append(child_xy[i])
                sample_id.append(parent_id[i])
                c += 1
                if c >= n:
                    break

        parent_offset += n_parents  # new parents get new ids

    child_xy = np.array(sample_xy[:n])
    parent_id = np.array(sample_id[:n])
    all_parent_xy = np.vstack(all_parent_xy)

    return child_xy, parent_id, all_parent_xy

# ---------------------------------------------------------------------------
# Step 4 — Build GeoDataFrame with parent membership
# ---------------------------------------------------------------------------

def build_poi_gdf(child_xy, parent_id):
    """
    Create a GeoDataFrame with columns:
        Lon, Lat, geometry, parent_id, Name (formatted C0001 … C2000)
    """
    n = len(child_xy)
    names = [f"C{i+1:04d}" for i in range(n)]
    gdf = gpd.GeoDataFrame({
        "Name":      names,
        "Lon":       child_xy[:, 0],
        "Lat":       child_xy[:, 1],
        "parent_id": parent_id,
    }, geometry=[Point(xy) for xy in child_xy])
    return gdf


# ---------------------------------------------------------------------------
# Step 5 — Assign MAUP-clean marks (cluster strength + anchor effect)
# ---------------------------------------------------------------------------

def assign_marks(poi_gdf, rng,
                 cluster_mean_low=1.0, cluster_mean_high=10.0,
                 noise_sigma=0.3,
                 anchor_pct=0.04, anchor_boost=3.0):
    """
    Mark model — ADDITIVE, so raw values show visible cluster-level
    colour differences on a linear colourmap.

        S_g ~ Uniform(low, high)              cluster strength level
        mark_k = S_g + eps_k                  eps_k ~ Normal(0, noise_sigma)
        top 4 % within cluster get +anchor_boost  (anchor effect)

    Each cluster gets a distinct mean level → clusters appear as
    different colours without needing log transform.
    """
    poi_gdf = poi_gdf.copy()
    marks = np.zeros(len(poi_gdf))
    groups = poi_gdf.groupby("parent_id")

    for g, idx in groups.groups.items():
        idx_arr = idx.values if hasattr(idx, 'values') else np.array(list(idx))
        ng = len(idx_arr)

        # cluster-level mean drawn from wide uniform range
        S_g = rng.uniform(cluster_mean_low, cluster_mean_high)

        # within-cluster additive noise
        eps = rng.normal(0, noise_sigma, size=ng)
        base = S_g + eps

        # anchor effect: top 4 % get additive boost
        n_anchors = max(1, int(np.ceil(anchor_pct * ng)))
        top_idx = np.argsort(base)[-n_anchors:]
        base[top_idx] += anchor_boost

        marks[idx_arr] = base

    poi_gdf["poi_mark"] = marks
    return poi_gdf

# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_neyman_scott(
    window_poly=None,
    n_children=2000,
    n_parents=20,
    cluster_radius=15,
    seed=1,
    conditioning=False,
    # mark parameters
    cluster_mean_low=1.0,
    cluster_mean_high=10.0,
    noise_sigma=0.3,
    anchor_pct=0.04,
    anchor_boost=3.0,
):
    """
    Full pipeline matching:
        PoissonClusterPointProcess(squwin, 2000, 20, 15, 1,
                                   conditioning=False, asPP=False)
    with parent–child membership preserved.

    Uses legacy np.random (seeded via np.random.seed) to match
    pointpats internals. Mark generation uses np.random.default_rng
    for the separate mark model.
    """
    if window_poly is None:
        window_poly = make_square_window(0, 0, 100, 100)

    # pointpats uses legacy np.random
    np.random.seed(seed)

    # children_per_parent = ceil(n / parents) — matching pointpats
    children_per_parent = int(np.ceil(n_children * 1.0 / n_parents))

    # Steps 1-3: draw with parent tracking
    child_xy, parent_id, parent_xy = draw_with_ids(
        n_children, n_parents, children_per_parent,
        cluster_radius, window_poly)

    # Step 4: GeoDataFrame
    poi_gdf = build_poi_gdf(child_xy, parent_id)

    # Step 5: marks (use separate rng so marks don't depend on spatial seed)
    mark_rng = np.random.default_rng(seed + 1000)
    poi_gdf = assign_marks(poi_gdf, mark_rng,
                           cluster_mean_low, cluster_mean_high,
                           noise_sigma, anchor_pct, anchor_boost)

    return poi_gdf, parent_xy