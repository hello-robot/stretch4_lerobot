"""A minimum spread for every normalization statistic of a LeRobot dataset.

LeRobot normalizes each state/action value by its spread in the dataset: max - min (MIN_MAX),
std (MEAN_STD), q99 - q01 or q90 - q10 (QUANTILES / QUANTILE10). A joint the demonstrations
barely moved has a spread of encoder noise, so normalization stretches that noise to the full
range and the policy can fit it.

floor_stats widens every such spread to at least the floor below, symmetrically around its
centre, so those values normalize to ~0 instead. A value whose spread is already above its
floor is left exactly as LeRobot computed it.
"""

# Half of the smallest spread each value may have, in its own units.
STATS_FLOOR = {
    "lift": 0.02, "arm": 0.02,                                     # m
    "wrist_yaw": 0.1, "wrist_pitch": 0.1, "wrist_roll": 0.1,       # rad
    "grip_mm": 5.0,                                                # mm
    "grip_effort": 5.0,                                            # % (0..100), state only
    "base_vx": 0.05, "base_vy": 0.05,                              # m/s
    "base_omega": 0.1,                                             # rad/s
}

# (low, high) statistics LeRobot normalizes by, widened together around their midpoint.
RANGE_PAIRS = (("min", "max"), ("q01", "q99"), ("q10", "q90"))


def floor_stats(stats, names, floor=None):
    """Floor one feature's statistics in place.

    stats: LeRobot's stats.json entry for one feature, {"min": [...], "max": [...], "std": [...],
           "q01": [...], ...} (missing keys are skipped); names: its value names, in order.
    Returns {name: {statistic: [spread before, spread after]}} of what was widened, for the
    record: e.g. {"base_vy": {"q10/q90": [0.0031, 0.1]}} (a std's spread is the std itself).
    """
    floor = STATS_FLOOR if floor is None else floor
    changed = {}
    for i, name in enumerate(names):
        half = floor.get(name)
        if half is None:
            continue
        for lo_key, hi_key in RANGE_PAIRS:
            if lo_key not in stats or hi_key not in stats:
                continue
            lo, hi = float(stats[lo_key][i]), float(stats[hi_key][i])
            if hi - lo < 2 * half:
                mid = (lo + hi) / 2
                stats[lo_key][i], stats[hi_key][i] = mid - half, mid + half
                changed.setdefault(name, {})[f"{lo_key}/{hi_key}"] = [hi - lo, 2 * half]
        # A uniform spread over [-half, half] has std half/sqrt(3); half/2 is the same order.
        if "std" in stats and float(stats["std"][i]) < half / 2:
            changed.setdefault(name, {})["std"] = [float(stats["std"][i]), half / 2]
            stats["std"][i] = half / 2
    return changed
