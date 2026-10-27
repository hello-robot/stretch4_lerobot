"""Keep the robot's state inside what the policy was trained on.

LeRobot normalizes each state value by its spread in the training dataset. A value the policy
never saw outside a narrow band (or never saw move at all) turns a small reading off that band
into a huge input. stretch4_fix_stats already holds the joints the demonstrations rarely moved
(a constant in training) and floors every spread; on the robot, the guard then:

  - pins each held joint's state to its training constant, and keeps the joint where it is
    instead of commanding the policy's value for it;
  - clamps every other state value to its training range [min, max] (meta/stats.json, already
    floored), and reports how far it had to clamp, so a robot outside the demos is visible.

When every value is inside its range, the guard changes nothing: the policy gets exactly the
measured state. Used by Stretch4 in control mode (--robot.preprocessing_from).
"""

import json
from pathlib import Path

from . import sidecar
from .stats_floor import STATS_FLOOR


class StateGuard:
    """names: the state's value names, in order; held: {joint: training constant};
    bounds: {name: (lo, hi)}."""

    def __init__(self, names, held=None, bounds=None):
        self.names = list(names)
        self.held = dict(held or {})
        self.bounds = dict(bounds or {})
        unknown = [n for n in self.held if n not in self.names]
        if unknown:
            raise ValueError(f"held joints {unknown} are not state values {self.names}")

    @classmethod
    def from_dataset(cls, root, names):
        """From a dataset's meta/stretch4.json "prepared" (held joints) and meta/stats.json
        (observation.state min/max). -> (guard, problem): problem is a sentence when the dataset
        was not prepared by stretch4_fix_stats, or changed since, else None."""
        root = Path(root)
        side = sidecar.read(root) or {}
        prepared = side.get("prepared")
        problem = None
        if not prepared:
            problem = (f"{root} was not prepared with stretch4_fix_stats: the policy may have "
                       "trained on amplified joint noise, and no joint is held")
        else:
            from .prepare_dataset import is_stale
            stale = is_stale(root, side)
            if stale:
                problem = (f"{root} changed after stretch4_fix_stats ({stale}); re-run it and "
                           "retrain")
        held = {j: float(h["value"]) for j, h in ((prepared or {}).get("held_joints") or {}).items()}
        bounds = {}
        stats_path = root / "meta" / "stats.json"
        info_path = root / "meta" / "info.json"
        if stats_path.exists() and info_path.exists():
            st = json.loads(stats_path.read_text()).get("observation.state") or {}
            stat_names = json.loads(info_path.read_text())["features"]["observation.state"]["names"]
            if "min" in st and "max" in st:
                bounds = {n: (float(st["min"][i]), float(st["max"][i]))
                          for i, n in enumerate(stat_names) if n in names}
        return cls(names, held, bounds), problem

    def apply(self, state):
        """state {name: value} -> (guarded state, {name: amount clamped beyond the floor})."""
        out = dict(state)
        far = {}
        for name, value in state.items():
            if name in self.held:
                out[name] = self.held[name]
                continue
            lo_hi = self.bounds.get(name)
            if lo_hi is None:
                continue
            lo, hi = lo_hi
            clamped = min(max(value, lo), hi)
            if clamped != value:
                out[name] = clamped
                if abs(value - clamped) > STATS_FLOOR.get(name, 0.0):
                    far[name] = value - clamped
        return out, far

    def hold_targets(self, targets, current):
        """Commanded joint targets with every held joint replaced by where it is now."""
        out = dict(targets)
        for joint in self.held:
            if joint in out and joint in current:
                out[joint] = current[joint]
        return out

    def describe(self):
        return ", ".join(f"{j}={v:.4g}" for j, v in self.held.items()) or "none"
