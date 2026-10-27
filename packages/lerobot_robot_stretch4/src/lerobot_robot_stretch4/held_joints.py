"""Which joints a policy should not learn: the decision stretch4_fix_stats makes.

The held joints are written into the data and recorded in meta/stretch4.json. A held joint is a
constant in training, in both the state and the action, and the robot keeps it where it is.
"""

import numpy as np

from .columns import POSE_JOINTS

# The joints a policy commands. The base is never held, and grip_effort is a state value only.
JOINTS = list(POSE_JOINTS)
# Within-episode range a joint must exceed to count as "moved" in that episode, in its
# own units. Well above encoder drift of a joint held still, well below any real motion.
DEFAULT_MIN_MOTION = {'lift': 0.005, 'arm': 0.005,                    # m
                      'wrist_yaw': 0.02, 'wrist_pitch': 0.02,         # rad (~1 deg)
                      'wrist_roll': 0.02,
                      'grip_mm': 1.0}                                 # mm
DEFAULT_MIN_EPISODE_FRAC = 0.1


def decide_held_joints(action, action_cols, episode_ends, episode_names=None,
                       min_episode_frac=DEFAULT_MIN_EPISODE_FRAC, min_motion=None,
                       hold_joints=(), keep_joints=(), auto=True):
    """Which joints the policy should not learn, and why.

    A joint moved in too few episodes cannot be learned -- one demonstration of *when* to
    use it is noise -- and it wrecks normalization: its few moving frames sit far outside
    the percentile range every other episode sets, at hundreds of times the normal scale.
    Such a joint is held: the dataset replaces it with a constant, and the robot keeps it
    where it is.

    A joint "moved" in an episode if its range there exceeds min_motion[joint]. It is held
    automatically if it moved in fewer than `min_episode_frac` of the episodes. hold_joints
    are held regardless; keep_joints are never held.

    Returns ({joint: {'value', 'reason'}}, report), report being one dict per joint.
    """
    min_motion = {**DEFAULT_MIN_MOTION, **(min_motion or {})}
    for name in [*hold_joints, *keep_joints]:
        if name not in JOINTS:
            raise ValueError(f"hold/keep joint {name!r} is not one of {JOINTS}")
    both = set(hold_joints) & set(keep_joints)
    if both:
        raise ValueError(f"{sorted(both)} are in both hold_joints and keep_joints")

    ends = np.asarray(episode_ends)
    starts = np.concatenate([[0], ends[:-1]])
    names = list(episode_names) if episode_names is not None else \
        [f'episode {i}' for i in range(len(ends))]
    n_eps = len(ends)
    held, report = {}, []
    for joint in JOINTS:
        x = action[:, action_cols.index(joint)].astype(np.float64)
        ranges = np.array([np.ptp(x[s:e]) if e > s else 0.0 for s, e in zip(starts, ends)])
        moved = [names[i] for i in np.nonzero(ranges > min_motion[joint])[0]]
        frac = len(moved) / max(n_eps, 1)
        entry = {
            'joint': joint,
            'episodes_moved': len(moved), 'episodes': n_eps,
            'fraction_moved': round(frac, 4),
            'min_motion': min_motion[joint],
            'q01': float(np.percentile(x, 1)), 'q99': float(np.percentile(x, 99)),
            'min': float(x.min()), 'max': float(x.max()), 'median': float(np.median(x)),
            # the names are only informative when few episodes moved it
            'moved_in': moved if len(moved) <= 10 else None,
        }
        if joint in hold_joints:
            decision, reason = 'held', 'listed in hold_joints'
        elif joint in keep_joints:
            decision, reason = 'learned', 'listed in keep_joints'
        elif auto and frac < min_episode_frac:
            decision = 'held'
            where = (f" ({', '.join(moved)})" if moved else '')
            reason = (f"moved more than {min_motion[joint]:g} in {len(moved)}/{n_eps} "
                      f"episodes{where}, below min_episode_frac {min_episode_frac:g}: too "
                      "rare to learn, and its moving frames would sit far outside the "
                      "normalization range")
        else:
            decision = 'learned'
            reason = (f"moved in {len(moved)}/{n_eps} episodes "
                      f"(>= min_episode_frac {min_episode_frac:g})" if auto else
                      'automatic holding is off')
        entry['decision'], entry['reason'] = decision, reason
        report.append(entry)
        if decision == 'held':
            held[joint] = {'value': entry['median'], 'reason': reason}
    return held, report


def format_hold_report(report):
    lines = ["[stretch] joint hold decisions (robot holds 'held' joints still):",
             f"[stretch]   {'joint':12s} {'moved':>9s}  {'q01..q99':>23s}  {'min..max':>23s}  decision"]
    for r in report:
        lines.append(
            f"[stretch]   {r['joint']:12s} {r['episodes_moved']:>4d}/{r['episodes']:<4d}  "
            f"{r['q01']:>11.5g}..{r['q99']:<11.5g} {r['min']:>11.5g}..{r['max']:<11.5g} "
            f"{r['decision'].upper()}")
        if r['decision'] == 'held':
            lines.append(f"[stretch]     -> {r['reason']}")
    return "\n".join(lines)
