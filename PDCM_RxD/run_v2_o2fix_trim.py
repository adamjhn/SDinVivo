from netpyne import sim
import numpy as np
import os
import sys
import pickle
import signal
import zlib
from neuron import h, rxd
import random
from stats import networkStatsFromSim
import json
from time import time
from netParams_v2_o2fix_trim import netParams, GLOBAL_CAPILLARY_LIST  # Phase C o2fix + v4 depth trim
import pandas as pd
from scipy.spatial import KDTree # THE SPEED UPGRADE
from scipy.stats import ks_2samp


def rand_uniform(gid, lb=0, ub=1):
    r = h.Random()
    r.Random123(gid, 1, 1)
    return r.uniform(lb, ub)


def rand_truncnorm(gid, mean=0, var=1, lb=None, ub=None):
    r = h.Random()
    r.Random123(gid, 1, 1)
    val = r.normal(mean, var)
    if lb is not None:
        val = max(lb, val)
    if ub is not None:
        val = min(val, ub)
    return val


cfg, netParams = sim.readCmdLineArgs(
    simConfigDefault="cfgPopWei.py", netParamsDefault="netParams_v2_o2fix_trim.py"
)

# Force all output files directly into the master folder
outdir = cfg.saveFolder

# Additional sim setup
# parallel context
pc = h.ParallelContext()
pcid = pc.id()
nhost = pc.nhost()
pc.timeout(0)
pc.set_maxstep(100)  # required when using multiple processes
random.seed(pcid + cfg.seeds["rec"])

# Forced-checkpoint-on-SIGUSR1: SLURM (via #SBATCH --signal=USR1@<secs>, no
# "B:" prefix) delivers this to every job-step task <secs> before the
# wall-clock time limit, so a `day`-partition run that's about to be killed
# gets one last checkpoint save instead of losing all progress since the
# last regular ssint boundary (up to 1000ms of sim time -- can be hours of
# wall-clock at this model's throughput). Every rank must independently save
# its own state file (see runSS()), so this can't rely on pcid==0 alone.
_checkpoint_signal_received = False


def _handle_checkpoint_signal(signum, frame):
    global _checkpoint_signal_received
    _checkpoint_signal_received = True


signal.signal(signal.SIGUSR1, _handle_checkpoint_signal)


def restoreSS():
    global lastss
    """restore sim state from saved files"""
    print(f"restore Save State from {cfg.restoredir}")
    svst = h.SaveState()
    f = h.File(os.path.join(cfg.restoredir, "save_test_" + str(pcid) + ".dat"))
    print("loaded file", f)
    svst.fread(f)
    print("read file", svst)
    svst.restore()
    print("restored", h.t)
    rvSeq = pickle.load(
        open(os.path.join(outdir, "save_randvar_" + str(pcid) + ".pkl"), "rb")
    )
    setRandSeq(rvSeq)
    lastss = h.t


def fi(cells):
    """set steady state RMP each cell
    when not restoring from a previous simulation"""

    # Don't run this -- it does not account for elevated K+
    """
    cfg.e_pas = {}
    for c in cells:
        # skip artificial cells
        if not hasattr(c.secs, "soma"):
            continue
        seg = c.secs.soma.hObj(0.5)
        isum = 0
        isum = (
            (seg.ina if h.ismembrane("na_ion") else 0)
            + (seg.ik if h.ismembrane("k_ion") else 0)
            + (seg.ica if h.ismembrane("ca_ion") else 0)
            + (seg.iother if h.ismembrane("other_ion") else 0)
        )
        seg.e_pas = cfg.hParams["v_init"] + isum / seg.g_pas
    """
    # restore from previous sim
    if cfg.restore:
        restoreSS()


def fi0(cells):
    for cell in cells:
        v = rand_truncnorm(
            cell.gid,
            cfg.cellPopsInit["mean"],
            cfg.cellPopsInit["std"] ** 2,
            ub=cfg.cellPopsInit["thresh"],
        )
        for sec in cell.secs.values():
            if "hObj" in sec:
                sec["hObj"].v = v


sim.initialize(
    simConfig=cfg, netParams=netParams
)  # create network object and set cfg and net params
sim.net.createPops()  # instantiate network populations
sim.net.createCells()  # instantiate network cells based on defined populations

# ==============================================================
# Empirical neuron-to-capillary distance matching
# ==============================================================
print("Adjusting cell positions to match empirical capillary distances...")

layer_mapping = {
    "L1e": 1, "L1i": 1, "L2e": 2, "L2i": 2, "L3e": 3, "L3i": 3,
    "L4e": 4, "L4i": 4, "L5e": 5, "L5i": 5, "L6e": 6, "L6i": 6,
}

all_cap_coords = []
for cap in GLOBAL_CAPILLARY_LIST:
    for z_idx, xy_pos in enumerate(cap):
        all_cap_coords.append(
            (
                xy_pos[0] * cfg.px,
                xy_pos[1] * cfg.px,
                z_idx * (cfg.sizeZ / len(cap)),
            )
        )

all_cap_coords = np.asarray(all_cap_coords, dtype=float)
print(f"Building KD-Tree for {len(all_cap_coords)} capillary points...")
cap_tree = KDTree(all_cap_coords)

print("Loading empirical distance distributions...")
empirical_by_layer = {}
for layer_num in sorted(set(layer_mapping.values())):
    csv_filename = f"all_Layer{layer_num}_distance.csv"
    if os.path.exists(csv_filename):
        df = pd.read_csv(csv_filename)
        distances = df["Nearest_Distance"].dropna().to_numpy(dtype=float)
        distances = distances[np.isfinite(distances)]
        empirical_by_layer[f"L{layer_num}"] = np.sort(distances)
    else:
        empirical_by_layer[f"L{layer_num}"] = np.asarray([30.8], dtype=float)

layer_y_ranges = {}
for pop_name, pop_data in netParams.popParams.items():
    for layer_name in ["L1", "L2", "L3", "L4", "L5", "L6"]:
        if layer_name in pop_name and "yRange" in pop_data:
            layer_y_ranges[layer_name] = pop_data["yRange"]


def layer_name_for_pop(pop):
    return f"L{layer_mapping[pop]}"


def get_soma_3d_center(cell):
    soma_sec = cell.secs.get("soma", {}).get("hObj")
    if soma_sec is None:
        return cell.tags["x"], -cell.tags["y"], cell.tags["z"]

    n3d = int(h.n3d(sec=soma_sec))
    if n3d == 0:
        return cell.tags["x"], -cell.tags["y"], cell.tags["z"]

    xs = [h.x3d(i, sec=soma_sec) for i in range(n3d)]
    ys = [h.y3d(i, sec=soma_sec) for i in range(n3d)]
    zs = [h.z3d(i, sec=soma_sec) for i in range(n3d)]
    return float(np.mean(xs)), float(np.mean(ys)), float(np.mean(zs))


def move_cell_to(cell, final_x, final_y, final_z):
    current_x, current_y, current_z = get_soma_3d_center(cell)
    dx_move = final_x - current_x
    # Cell tags use positive cortical depth, but RxD extracellular y spans
    # [-cfg.sizeY, 0], so section coordinates must live in negative-y space.
    dy_move = -final_y - current_y
    dz_move = final_z - current_z

    cell.tags["x"] = final_x
    cell.tags["y"] = final_y
    cell.tags["z"] = final_z

    for sec_dict in cell.secs.values():
        if "hObj" not in sec_dict:
            continue

        h_sec = sec_dict["hObj"]
        n3d = int(h.n3d(sec=h_sec))
        for i in range(n3d):
            h.pt3dchange(
                i,
                h.x3d(i, sec=h_sec) + dx_move,
                h.y3d(i, sec=h_sec) + dy_move,
                h.z3d(i, sec=h_sec) + dz_move,
                h.diam3d(i, sec=h_sec),
                sec=h_sec,
            )


def sample_layer_points(layer_name, n_points, rng):
    y_range = layer_y_ranges[layer_name]
    pts = np.column_stack(
        (
            rng.uniform(0.0, cfg.sizeX, n_points),
            rng.uniform(y_range[0], y_range[1], n_points),
            rng.uniform(0.0, cfg.sizeZ, n_points),
        )
    )
    dists, nearest_idx = cap_tree.query(pts)
    return pts, np.asarray(dists), np.asarray(nearest_idx)


def append_candidate_points(pool, layer_name, n_points, rng, max_points):
    remaining = max_points - pool["n_points"]
    if remaining <= 0:
        return False

    n_points = min(n_points, remaining)
    pts, dists, nearest_idx = sample_layer_points(layer_name, n_points, rng)
    if pool["points"] is None:
        pool["points"] = pts
        pool["distances"] = dists
        pool["nearest_idx"] = nearest_idx
    else:
        pool["points"] = np.vstack((pool["points"], pts))
        pool["distances"] = np.concatenate((pool["distances"], dists))
        pool["nearest_idx"] = np.concatenate((pool["nearest_idx"], nearest_idx))
    pool["n_points"] += n_points
    return True


def finalize_candidate_pool(pool):
    pool["sorted_idx"] = np.argsort(pool["distances"])
    pool["used"] = np.zeros(pool["n_points"], dtype=bool)
    pool["max_dist"] = float(np.max(pool["distances"]))


def build_candidate_pool(layer_name, max_target, rng, max_points=None):
    if max_points is None:
        max_points = MAX_CANDIDATE_POINTS_PER_LAYER
    pool = {"points": None, "distances": None, "nearest_idx": None, "n_points": 0}
    append_candidate_points(pool, layer_name, INITIAL_CANDIDATE_POINTS, rng, max_points)
    while (
        np.max(pool["distances"]) < max_target
        and pool["n_points"] < max_points
    ):
        append_candidate_points(pool, layer_name, CANDIDATE_BATCH_SIZE, rng, max_points)
    finalize_candidate_pool(pool)
    return pool


def grid_key(point):
    return tuple(np.floor(point / SPACING_GRID_SIZE_UM).astype(int))


def far_enough_from_placed(point, occupied_grid, min_distance):
    if min_distance <= 0.0:
        return True

    gx, gy, gz = grid_key(point)
    for ix in range(gx - 1, gx + 2):
        for iy in range(gy - 1, gy + 2):
            for iz in range(gz - 1, gz + 2):
                for other in occupied_grid.get((ix, iy, iz), []):
                    if np.linalg.norm(point - other) < min_distance:
                        return False
    return True


def add_to_occupied(point, occupied_grid):
    key = grid_key(point)
    occupied_grid.setdefault(key, []).append(point)


def ordered_window_indices(pool, target_dist, tolerance, rng):
    distances = pool["distances"]
    sorted_idx = pool["sorted_idx"]
    sorted_distances = distances[sorted_idx]
    lo = np.searchsorted(sorted_distances, target_dist - tolerance, side="left")
    hi = np.searchsorted(sorted_distances, target_dist + tolerance, side="right")
    window = sorted_idx[lo:hi]
    if window.size == 0:
        return window

    if window.size > MAX_CANDIDATE_TRIALS_PER_CELL:
        if window.size > 4 * MAX_CANDIDATE_TRIALS_PER_CELL:
            window = rng.choice(
                window,
                size=MAX_CANDIDATE_TRIALS_PER_CELL,
                replace=False,
            )
        else:
            order = np.argsort(np.abs(distances[window] - target_dist))
            window = window[order[:MAX_CANDIDATE_TRIALS_PER_CELL]]
    else:
        order = np.argsort(np.abs(distances[window] - target_dist))
        window = window[order]
    return window


def is_inside_layer(point, y_range):
    return (
        0.0 <= point[0] <= cfg.sizeX
        and y_range[0] <= point[1] <= y_range[1]
        and 0.0 <= point[2] <= cfg.sizeZ
    )


def project_point_to_distance(query_point, nearest_index, query_dist, target_dist):
    if query_dist <= 0.0:
        return None
    cap_point = all_cap_coords[int(nearest_index)]
    return cap_point + ((target_dist / query_dist) * (query_point - cap_point))


def fallback_candidates(pool, layer_name, target_dist, rng):
    """Yield (point, measured_dist) candidates for cells whose target distance
    isn't well covered by the exact-match window search.

    For targets beyond the sampled pool's max distance, each candidate is
    projected outward from a freshly sampled layer point (via
    sample_layer_points, not the pool's own static far tail), along the ray
    from its nearest capillary through itself, scaled to land exactly at
    target_dist. Voronoi cells are convex, so the projected point keeps the
    same nearest capillary. Drawing fresh anchors every call -- rather than
    reusing a fixed subset of the pool's farthest points -- avoids many
    far-target cells repeatedly projecting from the same handful of sites
    (the L5 "bulky tail"). Candidates that land outside the simulation box
    are discarded -- if every trial does, the box genuinely cannot reach that
    distance for this layer.
    """
    y_range = layer_y_ranges[layer_name]
    distances = pool["distances"]
    nearest_idx = pool["nearest_idx"]
    points = pool["points"]
    sorted_idx = pool["sorted_idx"]

    if target_dist >= pool["max_dist"]:
        fresh_pts, fresh_dists, fresh_nearest_idx = sample_layer_points(
            layer_name, FALLBACK_FRESH_SAMPLE_SIZE, rng
        )
        trial_order = rng.permutation(fresh_pts.shape[0])[:MAX_FALLBACK_CANDIDATES]
        for idx in trial_order:
            candidate = project_point_to_distance(
                fresh_pts[idx], fresh_nearest_idx[idx], fresh_dists[idx], target_dist
            )
            if candidate is None or not is_inside_layer(candidate, y_range):
                continue
            measured_dist, _ = cap_tree.query(candidate)
            yield candidate, float(measured_dist)
        return

    pos = np.searchsorted(distances[sorted_idx], target_dist)
    lo = max(0, pos - MAX_FALLBACK_CANDIDATES // 2)
    hi = min(sorted_idx.size, pos + MAX_FALLBACK_CANDIDATES // 2)
    window = sorted_idx[lo:hi]
    order = np.argsort(np.abs(distances[window] - target_dist))
    for idx in window[order]:
        yield points[idx], float(distances[idx])


def choose_candidate(pool, target_dist, occupied_grid, layer_name, rng):
    spacing_rejects = 0
    best_point = None
    best_dist = None
    best_error = float("inf")
    best_pool_idx = None

    for tolerance in DISTANCE_TOLERANCE_STEPS_UM:
        for idx in ordered_window_indices(pool, target_dist, tolerance, rng):
            if pool["used"][idx]:
                continue
            point = pool["points"][idx]
            error = abs(float(pool["distances"][idx]) - target_dist)
            if error < best_error:
                best_error = error
                best_point = point
                best_dist = float(pool["distances"][idx])
                best_pool_idx = int(idx)

            if far_enough_from_placed(point, occupied_grid, MIN_NEURON_SPACING_UM):
                pool["used"][idx] = True
                add_to_occupied(point, occupied_grid)
                return point, float(pool["distances"][idx]), error, spacing_rejects, False, False
            spacing_rejects += 1

    for point, measured_dist in fallback_candidates(pool, layer_name, target_dist, rng):
        error = abs(measured_dist - target_dist)
        if error < best_error:
            best_error = error
            best_point = point
            best_dist = measured_dist
            best_pool_idx = None

        if far_enough_from_placed(point, occupied_grid, MIN_NEURON_SPACING_UM):
            add_to_occupied(point, occupied_grid)
            return point, measured_dist, error, spacing_rejects, True, False
        spacing_rejects += 1

    # Last resort: keep all cells placed, but make the capacity failure explicit.
    # This is where the PI idea runs out of physically separated sampled sites.
    if best_point is None:
        unused = np.flatnonzero(~pool["used"])
        if unused.size == 0:
            raise RuntimeError("Candidate pool exhausted before all cells were placed")
        order = np.argsort(np.abs(pool["distances"][unused] - target_dist))
        best_pool_idx = int(unused[order[0]])
        best_point = pool["points"][best_pool_idx]
        best_dist = float(pool["distances"][best_pool_idx])
        best_error = abs(best_dist - target_dist)

    if best_pool_idx is not None:
        pool["used"][best_pool_idx] = True
    add_to_occupied(best_point, occupied_grid)
    return best_point, best_dist, best_error, spacing_rejects, True, True


DISTANCE_MATCH_GOAL_UM = 1.0
DISTANCE_TOLERANCE_STEPS_UM = (0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0)
CANDIDATE_BATCH_SIZE = 250000
INITIAL_CANDIDATE_POINTS = 500000
MAX_CANDIDATE_POINTS_PER_LAYER = 2500000
MAX_CANDIDATE_TRIALS_PER_CELL = 512
MAX_FALLBACK_CANDIDATES = 4096
FALLBACK_FRESH_SAMPLE_SIZE = 2000
# cfg.somaR (~81 um) is an RxD tissue-homogenization radius (Vtissue/NcellRxD),
# not a literal inter-soma spacing distance. Enforcing 2*cfg.somaR (~162 um) as a
# pairwise minimum separation is geometrically infeasible at the requested cell
# densities per layer, which is why most cells would fall back to
# spacing-ignoring last-resort placement. Use a purpose-built literal minimum
# inter-soma spacing distance instead (real cortical pyramidal soma diameters
# are roughly 10-25 um, not 162 um).
MIN_NEURON_SPACING_UM = 20.0
SPACING_GRID_SIZE_UM = MIN_NEURON_SPACING_UM

# === Self-validation / retry against empirical CSVs ===
# A single placement draw can land a layer's simulated distance distribution
# away from the empirical CSV by chance (greedy tail-first consumption order,
# spacing rejects, and candidate-pool sampling are all randomized). Rather than
# accept whatever one draw produces, each layer is placed and KS-tested against
# its empirical CSV; a bad draw (KS stat over tolerance) is retried from an
# independent seed. Only if repeated retries with reshuffled placement still
# fail does the layer escalate to resampling a larger candidate pool, which
# distinguishes genuine bad luck (fixed by retrying) from a structural capacity
# shortfall (the box cannot physically reach the empirical tail, no matter how
# many times it's retried -- see frac targets>candidate max diagnostic).
KS_STAT_TOLERANCE = 0.05
MAX_ATTEMPTS_PER_LAYER = 5
POOL_REBUILD_AFTER_ATTEMPT = 3
POOL_GROWTH_FACTOR = 2.0
POOL_HARD_CAP = MAX_CANDIDATE_POINTS_PER_LAYER * 4


def layer_seed(layer_name, attempt):
    return (
        int(cfg.seeds["loc"]),
        int(pcid),
        zlib.crc32(layer_name.encode()) & 0xFFFFFFFF,
        int(attempt),
    )


cells_by_layer = {}
for cell in sim.net.cells:
    pop = cell.tags.get("pop", "")
    if cell.tags.get("cellModel") == "NetStim" or pop not in layer_mapping:
        continue
    cells_by_layer.setdefault(layer_name_for_pop(pop), []).append(cell)

layer_target_values = {}
max_target_by_layer = {}
for layer_name, layer_cells in cells_by_layer.items():
    empirical = empirical_by_layer.get(layer_name, np.asarray([30.8], dtype=float))
    empirical = empirical[np.isfinite(empirical)]
    if empirical.size == 0:
        empirical = np.asarray([30.8], dtype=float)

    q = (np.arange(len(layer_cells)) + 0.5) / len(layer_cells)
    targets = np.quantile(empirical, q)
    layer_target_values[layer_name] = targets
    max_target_by_layer[layer_name] = float(np.max(targets))

print("Pre-sampling candidate sites for capacity-aware placement...")
candidate_pool_by_layer = {}
for layer_name in sorted(cells_by_layer):
    pool_rng = np.random.default_rng(layer_seed(layer_name, 0))
    candidate_pool_by_layer[layer_name] = build_candidate_pool(
        layer_name,
        max_target_by_layer[layer_name],
        pool_rng,
    )
    support_max = candidate_pool_by_layer[layer_name]["max_dist"]
    frac_over_capacity = float(
        np.mean(layer_target_values[layer_name] > support_max)
    )
    print(
        f"{layer_name}: target max={max_target_by_layer[layer_name]:.2f} um, "
        f"candidate max={support_max:.2f} um, "
        f"candidate sites={candidate_pool_by_layer[layer_name]['n_points']}, "
        f"frac targets>candidate max={frac_over_capacity:.3f}, "
        f"min neuron spacing={MIN_NEURON_SPACING_UM:.2f} um"
    )
    if support_max < max_target_by_layer[layer_name]:
        print(
            f"WARNING: {layer_name} empirical tail exceeds sampled candidate support. "
            "The far tail can only be matched up to the farthest available sites."
        )


def place_layer_attempt(layer_name, layer_cells, targets, pool, rng):
    """Run one full placement draw for a layer and return per-cell results.
    Does not touch real cell positions -- callers apply positions_by_gid only
    for whichever attempt is ultimately accepted."""
    shuffled_cells = list(layer_cells)
    rng.shuffle(shuffled_cells)
    target_by_gid = {}
    placement_order = np.argsort(targets)[::-1]
    for target_idx in placement_order:
        cell = shuffled_cells[int(target_idx)]
        target_by_gid[cell.gid] = float(targets[int(target_idx)])

    pool["used"][:] = False
    occupied_grid = {}
    ordered_cells = sorted(layer_cells, key=lambda c: target_by_gid[c.gid], reverse=True)

    n_cells = len(layer_cells)
    measured = np.empty(n_cells, dtype=float)
    errors = np.empty(n_cells, dtype=float)
    targets_ordered = np.empty(n_cells, dtype=float)
    positions_by_gid = {}
    fallback_count = 0
    far_fallback_count = 0
    spacing_failure_count = 0
    spacing_reject_total = 0

    for i, cell in enumerate(ordered_cells):
        target_distance = target_by_gid[cell.gid]
        point, measured_dist, measured_error, spacing_rejects, used_fallback, spacing_failed = (
            choose_candidate(pool, target_distance, occupied_grid, layer_name, rng)
        )

        positions_by_gid[cell.gid] = point
        targets_ordered[i] = target_distance
        measured[i] = measured_dist
        errors[i] = measured_error
        spacing_reject_total += spacing_rejects

        if used_fallback:
            fallback_count += 1
            if target_distance >= pool["max_dist"]:
                far_fallback_count += 1
        if spacing_failed:
            spacing_failure_count += 1

    return {
        "positions_by_gid": positions_by_gid,
        "measured": measured,
        "errors": errors,
        "targets_ordered": targets_ordered,
        "fallback_count": fallback_count,
        "far_fallback_count": far_fallback_count,
        "spacing_failure_count": spacing_failure_count,
        "spacing_reject_total": spacing_reject_total,
    }


success_count = 0
fail_count = 0
fallback_count = 0
spacing_failure_count = 0
spacing_reject_count = 0
layer_errors = {}
layer_measured = {}
layer_targets = {}
layer_fallbacks = {}
layer_far_fallbacks = {}
layer_spacing_failures = {}
layer_attempts_used = {}
layer_converged = {}

print(
    "Teleporting cells with tail-first, capacity-aware candidate placement "
    f"(self-validating each layer against its empirical CSV, KS tolerance={KS_STAT_TOLERANCE:.3f}, "
    f"max {MAX_ATTEMPTS_PER_LAYER} attempts/layer)..."
)
for layer_name in sorted(cells_by_layer):
    layer_cells = cells_by_layer[layer_name]
    targets = layer_target_values[layer_name]
    empirical = empirical_by_layer.get(layer_name, np.asarray([30.8], dtype=float))
    pool = candidate_pool_by_layer[layer_name]

    best_result = None
    best_ks_stat = float("inf")
    best_ks_pvalue = None
    best_attempt_num = None
    accepted = False
    attempt = 0

    for attempt in range(1, MAX_ATTEMPTS_PER_LAYER + 1):
        rng_attempt = np.random.default_rng(layer_seed(layer_name, attempt))

        if attempt > POOL_REBUILD_AFTER_ATTEMPT:
            grown = min(
                int(MAX_CANDIDATE_POINTS_PER_LAYER * POOL_GROWTH_FACTOR),
                POOL_HARD_CAP,
            )
            print(
                f"{layer_name}: escalating to a freshly resampled candidate pool "
                f"({grown} pts) for attempt {attempt} after repeated KS failures"
            )
            pool = build_candidate_pool(
                layer_name, max_target_by_layer[layer_name], rng_attempt, max_points=grown
            )
            candidate_pool_by_layer[layer_name] = pool

        result = place_layer_attempt(layer_name, layer_cells, targets, pool, rng_attempt)
        ks_stat, ks_pvalue = ks_2samp(result["measured"], empirical)
        passed = ks_stat <= KS_STAT_TOLERANCE

        status = "PASS" if passed else ("RETRY" if attempt < MAX_ATTEMPTS_PER_LAYER else "FAIL-KEEP-BEST")
        print(
            f"{layer_name} attempt {attempt}/{MAX_ATTEMPTS_PER_LAYER}: "
            f"KS stat={ks_stat:.4f} (p={ks_pvalue:.2e}, tolerance={KS_STAT_TOLERANCE:.3f}) -> {status}"
        )

        if ks_stat < best_ks_stat:
            best_ks_stat = ks_stat
            best_ks_pvalue = ks_pvalue
            best_result = result
            best_attempt_num = attempt

        if passed:
            accepted = True
            break

    layer_attempts_used[layer_name] = attempt
    layer_converged[layer_name] = accepted

    if not accepted:
        support_max = candidate_pool_by_layer[layer_name]["max_dist"]
        frac_over_capacity = float(np.mean(targets > support_max))
        print(
            f"WARNING: {layer_name} did not converge to KS stat <= {KS_STAT_TOLERANCE} in "
            f"{MAX_ATTEMPTS_PER_LAYER} attempts (best KS stat={best_ks_stat:.4f} on attempt "
            f"{best_attempt_num}); using best observed placement. "
            f"frac targets>candidate max={frac_over_capacity:.3f} -- if this is > 0, the "
            "empirical tail structurally exceeds this layer's box capacity and retrying "
            "cannot fix it (see the sizeX/sizeZ domain-size note)."
        )

    for cell in layer_cells:
        point = best_result["positions_by_gid"][cell.gid]
        move_cell_to(cell, point[0], point[1], point[2])

    layer_measured[layer_name] = best_result["measured"]
    layer_targets[layer_name] = best_result["targets_ordered"]
    layer_errors[layer_name] = best_result["errors"]
    layer_fallbacks[layer_name] = best_result["fallback_count"]
    layer_far_fallbacks[layer_name] = best_result["far_fallback_count"]
    layer_spacing_failures[layer_name] = best_result["spacing_failure_count"]

    fallback_count += best_result["fallback_count"]
    spacing_failure_count += best_result["spacing_failure_count"]
    spacing_reject_count += best_result["spacing_reject_total"]
    success_count += int(np.sum(best_result["errors"] <= DISTANCE_MATCH_GOAL_UM))
    fail_count += int(np.sum(best_result["errors"] > DISTANCE_MATCH_GOAL_UM))

print("\n--- TELEPORTATION RESULTS ---")
print(
    f"Candidate placements (<={DISTANCE_MATCH_GOAL_UM:.2f} um distance error): "
    f"{success_count} cells"
)
print(f"Best available / capacity-limited placements: {fail_count} cells")
print(f"Fallback candidate selections: {fallback_count} cells")
print(f"Spacing failures in last-resort placement: {spacing_failure_count} cells")
print(f"Rejected candidate sites due to spacing: {spacing_reject_count}")
for layer_name in sorted(layer_measured):
    measured = np.asarray(layer_measured[layer_name], dtype=float)
    targets = np.asarray(layer_targets[layer_name], dtype=float)
    errors = np.asarray(layer_errors[layer_name], dtype=float)
    if measured.size == 0:
        continue
    quantile_errors = np.abs(np.sort(measured) - np.sort(targets))
    n_cells = measured.size
    frac_fallback = layer_fallbacks[layer_name] / n_cells
    frac_far_fallback = layer_far_fallbacks[layer_name] / n_cells
    frac_spacing_failed = layer_spacing_failures[layer_name] / n_cells
    empirical = empirical_by_layer.get(layer_name, np.asarray([30.8], dtype=float))
    ks_stat, ks_pvalue = ks_2samp(measured, empirical)
    print(
        f"{layer_name}: converged={layer_converged[layer_name]} "
        f"(attempts={layer_attempts_used[layer_name]}/{MAX_ATTEMPTS_PER_LAYER}), "
        f"target mean={np.mean(targets):.2f}, "
        f"measured mean={np.mean(measured):.2f}, "
        f"target p95={np.percentile(targets, 95):.2f}, "
        f"measured p95={np.percentile(measured, 95):.2f}, "
        f"median placement error={np.median(errors):.3f} um, "
        f"p95 quantile error={np.percentile(quantile_errors, 95):.3f} um, "
        f"max quantile error={np.max(quantile_errors):.3f} um, "
        f"fallbacks={layer_fallbacks[layer_name]} (frac={frac_fallback:.3f}, "
        f"far-tail frac={frac_far_fallback:.3f}), "
        f"spacing failures={layer_spacing_failures[layer_name]} (frac={frac_spacing_failed:.3f}), "
        f"KS stat={ks_stat:.4f} (p={ks_pvalue:.2e}) vs empirical CSV (n={empirical.size})"
    )
print("=============================\n")

print("Spatial distribution complete. Finalizing setup...")
# ==============================================================

# ... (Continue with normal operations) ...
sim.net.connectCells()  # create connections between cells based on params
sim.net.addStims()  # add external stimulation to cells (IClamps etc)
sim.net.addRxD(nthreads=6)  # add reaction-diffusion (RxD)
"""clamps = []
for cell in sim.net.cells:
    if cell.tags['cellModel'] != "VecStim" and cell.tags['cellModel'] != "NetStim":
        vclamp = h.VClamp(cell.secs['soma']['hObj'](0.5))
        vclamp.dur[0] = 25
        vclamp.dur[1] = 0
        vclamp.dur[2] = 0
        vclamp.amp[0] = cfg.hParams["v_init"]
        clamps.append(vclamp)
"""
sim.setupRecording()  # setup variables to record for each cell
fih = h.FInitializeHandler(1, lambda: fi(sim.net.cells))
if not cfg.restore:
    fih0 = h.FInitializeHandler(0, lambda: fi0(sim.getCellsList(include=cfg.cellPops)))

# only single core stuff
if pcid == 0:
    # create output dir
    if not os.path.exists(outdir):
        try:
            os.makedirs(outdir)
        except:
            print("Unable to create the directory %r for the data and figures" % outdir)
            os._exit(1)

    # set variables for ecs concentrations
    k_ecs = sim.net.rxd["species"]["kk"]["hObj"][sim.net.rxd["regions"]["ecs"]["hObj"]]
    na_ecs = sim.net.rxd["species"]["na"]["hObj"][sim.net.rxd["regions"]["ecs"]["hObj"]]
    cl_ecs = sim.net.rxd["species"]["cl"]["hObj"][sim.net.rxd["regions"]["ecs"]["hObj"]]
    volume_ecs = sim.net.rxd["states"]["vol_ratio"]["hObj"][
        sim.net.rxd["regions"]["ecs"]["hObj"]
    ]
    o2_ecs = sim.net.rxd["species"]["o2_extracellular"]["hObj"][
        sim.net.rxd["regions"]["ecs_o2"]["hObj"]
    ]
    o2con = sim.net.rxd["states"]["o2_consumed"]["hObj"][
        sim.net.rxd["regions"]["ecs_o2"]["hObj"]
    ]

# manually record from cells from each layer
rng = np.random.default_rng(seed=pcid + cfg.seeds["rec"])
rec_cells = {}
# look up volume ratio species
rxd_volume = sim.net.rxd["states"]["vol_ratio"]["hObj"]

for lab, pop in sim.net.pops.items():
    if "xRange" in pop.tags:
        rec_cells[lab] = {
            "gid": (
                rng.choice(
                    pop.cellGids, size=int(max(1, cfg.nRec / nhost)), replace=False
                )
                if len(pop.cellGids) > min(1, cfg.nRec / nhost)
                else pop.cellGids
            )
        }
        rec_cells[lab]["pos"] = []
        for k in ["v", "ki", "nai", "cli", "ko", "nao", "clo", "o2o", "volume"]:
            rec_cells[lab][k] = []
        for idx in rec_cells[lab]["gid"]:
            cell = sim.cellByGid(idx)
            soma = cell.secs["soma"]["hObj"]
            rec_cells[lab]["pos"].append(cell.getSomaPos())
            for k in ["v", "ki", "nai", "cli", "ko", "nao", "clo", "o2o"]:
                rec_cells[lab][k].append(
                    h.Vector().record(getattr(soma(0.5), f"_ref_{k}"))
                )
            rec_cells[lab]["volume"].append(
                h.Vector().record(rxd_volume.nodes(soma(0.5))._ref_value)
            )
if pcid == 0:
    rec_cells["time"] = h.Vector().record(h._ref_t)


def getRandSeq():
    """Return a dict of gid->rv.get_seq for all cells with
    cellModel=='NetStim'
    In this model the NetStim are cells so are not listed in
    netParams.stimSourceParams, so cannot access them with
    include=['allNetStims'].
    """
    saveSeq = {"seq": {}, "ids": {}}
    for cell in sim.getCellsList(include=["all"]):
        if cell.tags["cellModel"] == "NetStim":
            saveSeq["seq"][cell.gid] = cell.hPointp.ranvar.get_seq()
            saveSeq["ids"][cell.gid] = list(cell.hPointp.ranvar.get_ids().as_numpy())
    return saveSeq


def setRandSeq(seqDict):
    """Take a dict of gid->seq and apply it to all cells"""
    done = []
    for cell in sim.getCellsList(include=["all"]):
        if cell.tags["cellModel"] == "NetStim":
            if cell.gid in seqDict["seq"]:
                cell.hPointp.ranvar.set_seq(seqDict["seq"][cell.gid])
                # cell.hPointp.ranvar.set_ids(*seqDict['ids'][cell.gid])
                done.append(cell.gid)
            else:
                raise Exception(
                    f"Failed to set randvar for cell {cell} gid {cell.gid} -- missing value"
                )
    for gid in seqDict["seq"]:
        if gid not in done:
            raise Exception(f"Failed to set randvar for cell gid {gid}")


def runSS():
    svst = h.SaveState()
    svst.save()
    f = h.File(os.path.join(outdir, "save_test_" + str(pcid) + ".dat"))
    svst.fwrite(f)
    rvSeq = getRandSeq()
    pickle.dump(
        rvSeq, open(os.path.join(outdir, "save_randvar_" + str(pcid) + ".pkl"), "wb")
    )


def saveconc():
    np.save(os.path.join(outdir, "k_%i.npy" % int(h.t)), k_ecs.states3d)
    np.save(os.path.join(outdir, "na_%i.npy" % int(h.t)), na_ecs.states3d)
    np.save(os.path.join(outdir, "cl_%i.npy" % int(h.t)), cl_ecs.states3d)
    np.save(os.path.join(outdir, "o2_%i.npy" % int(h.t)), o2_ecs.states3d)
    np.save(os.path.join(outdir, "o2con_%i.npy" % int(h.t)), o2con.states3d)
    np.save(os.path.join(outdir, "volume_%i.npy" % int(h.t)), volume_ecs.states3d)


def progress_bar(tstop, size=40):
    """report progress of the simulation"""
    prog = h.t / float(tstop)
    fill = int(size * prog)
    empt = size - fill
    progress = "#" * fill + "-" * empt
    sys.stdout.write(
        "[%s] %2.1f%% %6.1fms of %6.1fms\r" % (progress, 100 * prog, h.t, tstop)
    )
    sys.stdout.flush()


fout = None
lastss = 0
if pcid == 0:
    # record the wave progress
    fout = open(os.path.join(outdir, "wave_progress.txt"), "a")
    if cfg.k0Layer is None:
        yoff = cfg.sizeY / 2.0
    elif cfg.k0Layer == 2 or cfg.k0Layer == 3:
        yoff = sum(netParams.popParams["L2e"]["yRange"]) / 2
    elif cfg.k0Layer == 4:
        yoff = sum(netParams.popParams["L4e"]["yRange"]) / 2
    elif cfg.k0Layer == 5:
        yoff = sum(netParams.popParams["L5e"]["yRange"]) / 2
    elif cfg.k0Layer == 6:
        yoff = sum(netParams.popParams["L6e"]["yRange"]) / 2

cellSDOpen, cellSDClosed = {}, {}


durs = []
_rxd_threads_restored = False
_checkpoint_signal_saved = False


def runIntervalFunc(t):
    global _rxd_threads_restored, _checkpoint_signal_saved
    if _checkpoint_signal_received and not _checkpoint_signal_saved:
        # Latch so we save exactly once per signal, not on every remaining
        # 1ms tick before SLURM's hard kill actually lands.
        if pcid == 0:
            print(
                f"\nSIGUSR1 received near wall-clock limit at t={h.t:.1f}ms -- forcing checkpoint save",
                flush=True,
            )
        runSS()
        _checkpoint_signal_saved = True
    if not _rxd_threads_restored:
        # h.finitialize() (run inside runSimWithIntervalFunc, before this callback
        # ever fires) calls rxd's clear_rates(), which unconditionally resets the
        # RxD reaction thread pool to 1 (see clear_rates() / set_num_threads(1) in
        # rxd.cpp) as part of its normal structure-change rebuild -- not just on
        # teardown. Since addRxD(nthreads=6) runs long before the rest of the
        # network/structure is finalized, every reaction was silently running on
        # 1 thread for the whole sim. Structure is frozen by now, so redo it once.
        rxd.nthread(6)
        _rxd_threads_restored = True
    durs.append(time())
    """Write the wave_progress every 1ms"""
    global lastss, cellSDOpen, cellSDClosed
    saveint = 100  # save concentrations interval
    ssint = 1000  # save state interval
    lastss = 0
    if pcid == 0:
        if int(t) % saveint == 0:
            # plot extracellular concentrations averaged over depth every 100ms
            saveconc()
    for cell in sim.getCellsList(include=cfg.cellPops):
        v = cell.secs["soma"]["hObj"].v

        # check previously depolarized cells
        if cell.gid in cellSDOpen:
            if v <= cfg.SDThreshold:
                a = cellSDOpen[cell.gid]

                # if the cell was only depolarized for a single interval
                # it was probably just an AP -- remove it
                del cellSDOpen[cell.gid]
                if abs(a - h.t) > 2.5:
                    if cell.gid in cellSDClosed:
                        cellSDClosed[cell.gid].append((a, h.t))
                    else:
                        cellSDClosed[cell.gid] = [(a, h.t)]
        else:
            if v > cfg.SDThreshold:
                cellSDOpen[cell.gid] = h.t
    if ((int(t) % ssint == 0) and (h.t - lastss) > ssint) or (cfg.duration - t) < 1:
        runSS()
        lastss = t

        # sustained depolarization at current time
        cellSD = cellSDClosed.copy()
        for cell, a in cellSDOpen.items():
            if cell in cellSD:
                cellSD[cell].append([a, None])
            else:
                cellSD[cell] = [(a, None)]
        json.dump(cellSD, open(os.path.join(outdir, f"cellsSD_{pcid}.json"), "w"))

    if pcid == 0:
        progress_bar(cfg.duration)
        dist = 0
        dist1 = 1e9
        for nd in sim.net.rxd["species"]["kk"]["hObj"].nodes:
            if str(nd.region).split("(")[0] == "Extracellular":
                r = (
                    (nd.x3d - cfg.sizeX / 2.0) ** 2
                    + (nd.y3d + yoff) ** 2
                    + (nd.z3d - cfg.sizeZ / 2.0) ** 2
                ) ** 0.5
                if nd.concentration > cfg.Kceil and r > dist:
                    dist = r
                if nd.concentration <= cfg.Kceil and r < dist1:
                    dist1 = r
        fout.write("%g\t%g\t%g\n" % (h.t, dist, dist1))
        fout.flush()


sim.runSimWithIntervalFunc(1, runIntervalFunc)
# cfg.gatherOnlySimData=True (set in cfgPopWei.py) makes netpyne's gatherData()
# skip the real cross-rank cell gather -- per netpyne/sim/gather.py, rank 0's
# sim.net.allCells then just falls back to rank 0's OWN LOCAL sim.net.cells (a
# documented "avoid errors" workaround, not an actual merge), which would make
# the network_positions_*.json below only contain one rank's share of cells.
# Override gatherOnlySimData=False for this call only, forcing netpyne to
# merge netCells across all MPI ranks into sim.net.allCells; this is safe to
# do at full production scale because netpyne's gatherData() unconditionally
# clears cell.secs/cell.conns before building that per-cell payload whenever
# cfg.saveCellSecs/saveCellConns are False (as they are here) -- so the merged
# payload never actually contains compartment/connection data, only tags and
# other lightweight bookkeeping, regardless of this override. This call is
# the only safe place to do it: cell.secs is still needed by runIntervalFunc
# above (which just finished) and must not be touched any earlier, e.g. via a
# gatherData() call placed right after teleportation but before connectCells/
# addRxD -- that would silently wipe every cell's morphology before it's used.
# analyze=False: gatherOnlySimData=False also flips on gather.py's normally-
# dead "not gatherOnlySimData" analysis branch (popAvgRates() etc.), which has
# never run in this pipeline before. Skip it here (we don't need it) rather
# than exercise an untested code path for the first time at the end of a
# 12-hour production run; print the cell count ourselves instead.
sim.gatherData(gatherOnlySimData=False, analyze=False)
if pcid == 0:
    print(f"  Cells (full multi-rank merge): {len(sim.net.allCells)}")
    networkStatsFromSim(
        sim, filename=os.path.join(outdir, f"netstats_{cfg.duration/1000:0.2f}s.json")
    )

sim.saveData()

# ==============================================================
# ROUTE THE JSON FILE INTO THE CORRECT FOLDER (Master Node Only)
# ==============================================================
if pcid == 0:
    json_filename = f"network_positions_{cfg.ox}_{cfg.duration}ms.json"
    json_path = os.path.join(outdir, json_filename)

    # Extract just the raw X, Y, Z coordinates to keep the file lightweight
    positions_dict = {}
    for cell in sim.net.allCells:
        positions_dict[cell['gid']] = {
            'pop': cell['tags'].get('pop'),
            'x': cell['tags']['x'],
            'y': cell['tags']['y'],
            'z': cell['tags']['z']
        }

    with open(json_path, 'w') as f:
        json.dump(positions_dict, f)
    print(f"Saved all teleported cell positions to {json_path}!")
# ==============================================================
# ==============================================================

sim.analysis.plotData()
# merge rec_cells
rec_all = {}
for lab in rec_cells:
    # time only on pcid==0 - don't gather
    if lab == "time":
        rec_all["time"] = rec_cells["time"]
    else:
        rec_all[lab] = {}
        for k in rec_cells[lab]:
            # merge lists
            rec_all[lab][k] = pc.py_gather(rec_cells[lab][k], 0)

if pcid == 0:
    progress_bar(cfg.duration)
    fout.close()
    for lab in rec_all:
        if cfg.restore:
            try:
                rec_old = pickle.load(
                    open(os.path.join(outdir, f"recs_{lab}.pkl"), "rb")
                )
            except FileNotFoundError:
                # Pre-checkpoint segment never wrote recs_{lab}.pkl (e.g. it was
                # cut off by a forced SIGUSR1 checkpoint before reaching this
                # step) -- nothing to merge with, so just save what this segment
                # recorded instead of crashing before saveData()/the o2 heatmap.
                rec_old = None
            if rec_old is not None:
                if lab == "time":
                    rec_old.extend(rec_all["time"])
                else:
                    for k in rec_all[lab]:
                        if k != "pos" and k != "pop" and k != "gid":
                            for u, v in zip(rec_all[lab][k], rec_old[k]):
                                for x, y in zip(u, v):
                                    y.extend(x)
                pickle.dump(rec_old, open(os.path.join(outdir, f"recs_{lab}.pkl"), "wb"))
            else:
                pickle.dump(
                    rec_all[lab], open(os.path.join(outdir, f"recs_{lab}.pkl"), "wb")
                )
        else:
            pickle.dump(
                rec_all[lab], open(os.path.join(outdir, f"recs_{lab}.pkl"), "wb")
            )
    print("\nSimulation complete. Plotting membrane potentials")
    # ==============================================================
    # ROUTE THE HEATMAP INTO THE GRAPHS FOLDER
    # ==============================================================
    graph_folder = getattr(cfg, "graph_folder", f"./graphs/{cfg.ox}")
    os.makedirs(graph_folder, exist_ok=True)
    heatmap_name = f"o2_heatmap_{cfg.ox}_{cfg.duration}ms.png"

    sim.analysis.plotRxDConcentration(
        speciesLabel='o2_extracellular',
        regionLabel='ecs_o2',
        saveFig=os.path.join(graph_folder, heatmap_name)
    )

# Cleanly shut down the RxD reaction-thread pool before the process exits.
# rxd.nthread(6) in runIntervalFunc spawns worker threads that sit in
# TaskQueue_exe_tasks's pthread_cond_wait for the rest of the run. If they
# are still alive when the process's global destructors run at exit, one
# of libnrniv.so's destructors destroys that condition variable out from
# under the still-waiting threads -- undefined behavior in pthreads -- and
# pthread_cond_destroy hangs forever (observed: job 19502586 sat RUNNING,
# burning CPU, for 36+ hours after every real output was already saved).
# set_num_threads() (what rxd.nthread() calls) signals and joins worker
# threads properly when shrinking the pool, so dropping to 1 thread here
# lets the process exit normally instead of deadlocking in cleanup.
rxd.nthread(1)
# v0.0 - direct copy from ../uniformdensity/init.py
# v1.0 - added in o2 sources based on capillaries identified from histology
# v1.1 - set pas.e to maintain RMP and move restore state function
# v1.2 - replace centermembrane_potential with layer specific recordings
# v1.3 - fix save state (in NEURON 9) by restoring seq in NMODLRandom
