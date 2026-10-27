#!/usr/bin/env python3
"""stretch4_fix_stats: get a Stretch 4 LeRobot dataset ready to train. Run it after recording.

    stretch4_fix_stats <dataset>              # every dataset, every policy
    stretch4_fix_stats <dataset> --relative   # pi0.5 with --policy.use_relative_actions=true
    lerobot-train --dataset.repo_id=<dataset> ...

<dataset> is a repo id under $HF_LEROBOT_HOME or a dataset folder. lerobot-record cannot run it
(LeRobot computes the stats itself), so run it after recording; stretch4_push_dataset offers to.
Re-run it after adding episodes (lerobot-record --resume): the robot warns when a dataset changed
after it was prepared.

1. Held joints (held_joints.py):
   a joint the demonstrations moved in fewer than --min-episode-frac of the episodes cannot be
   learned, and its few moving frames wreck normalization. It is written as a constant (its
   median) into observation.state and action, and the robot keeps it where it is.
2. Stats: observation.state and action recomputed from the data (in relative space with
   --relative, as LeRobot's pi0.5 relative actions see them), q01/q99 widened with --tail, then
   every spread floored (stats_floor.py) so a value the demos barely moved normalizes to
   ~0 instead of amplified encoder noise (or ~1e7 on the robot, for one that never moved).
3. meta/stretch4.json "prepared" records what was done; the robot plugin reads the held joints
   from it.

In place, and repeatable: the first run saves the original state/action columns to
meta/stretch4_original_columns.npz and every run starts from them, so re-running with other
options, or after more episodes, never compounds; --undo puts the originals back.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

from . import sidecar
from .columns import check_state_names
from .held_joints import (DEFAULT_MIN_EPISODE_FRAC, DEFAULT_MIN_MOTION,
                          decide_held_joints, format_hold_report)
from .stats_floor import floor_stats

STATE, ACTION = "observation.state", "action"
FEATURES = (STATE, ACTION)
ORIGINALS = "meta/stretch4_original_columns.npz"
QUANTILES = (1, 10, 50, 90, 99)
# Kept absolute by --relative: the gripper and the base velocity (a velocity minus a velocity
# is no target). Substrings of action names, as LeRobot's relative_exclude_joints.
RELATIVE_EXCLUDE = ("grip", "base_v", "base_omega")
VERSION = 1


def lerobot_home():
    """$HF_LEROBOT_HOME, as LeRobot resolves it."""
    return Path(os.environ.get("HF_LEROBOT_HOME", Path(os.environ.get(
        "HF_HOME", "~/.cache/huggingface")) / "lerobot")).expanduser()


def resolve(dataset):
    """A dataset folder, or a repo id under $HF_LEROBOT_HOME."""
    path = Path(dataset).expanduser()
    if path.is_dir():
        return path.resolve()
    home = lerobot_home()
    path = home / dataset
    if path.is_dir():
        return path.resolve()
    sys.exit(f"no dataset {dataset!r} (not a folder, and not under {home})")


# ---------------------------------------------------------------- data files ---

def _data_files(root):
    return sorted((root / "data").glob("*/*.parquet"))


def _fixed(col):
    """A fixed-size-list parquet column -> (rows, width) float array."""
    import pyarrow as pa
    arr = col.combine_chunks() if isinstance(col, pa.ChunkedArray) else col
    width = arr.type.list_size
    return arr.flatten().to_numpy(zero_copy_only=False).reshape(-1, width)


def _write_table(table, path):
    # Through a temp file and rename, so a hard-linked source is never written to.
    import pyarrow.parquet as pq
    tmp = path.with_suffix(".tmp")
    pq.write_table(table, tmp)
    os.replace(tmp, path)


def _set_fixed(table, name, values):
    import pyarrow as pa
    i = table.schema.get_field_index(name)
    field = table.schema.field(i)
    arr = pa.FixedSizeListArray.from_arrays(
        pa.array(np.ascontiguousarray(values).ravel(), type=field.type.value_type),
        field.type.list_size)
    return table.set_column(i, field, arr)


def _originals(root):
    """{relpath: {feature: (rows, width)}} as recorded, saving them on the first run."""
    import pyarrow.parquet as pq
    path = root / ORIGINALS
    if path.exists():
        saved = np.load(path)
        out = {}
        for key in saved.files:
            rel, feature = key.split("::")
            out.setdefault(rel, {})[feature] = saved[key]
        # Files written since (resumed recording) are still as recorded.
        for f in _data_files(root):
            rel = str(f.relative_to(root))
            if rel not in out:
                t = pq.read_table(f, columns=list(FEATURES))
                out[rel] = {k: _fixed(t[k]) for k in FEATURES}
            elif len(out[rel][STATE]) != pq.read_metadata(f).num_rows:
                sys.exit(f"{rel} has a different number of rows than when it was first prepared; "
                         f"restore it, or delete {ORIGINALS} to take the current data as recorded")
        return out
    out = {}
    for f in _data_files(root):
        t = pq.read_table(f, columns=list(FEATURES))
        out[str(f.relative_to(root))] = {k: _fixed(t[k]) for k in FEATURES}
    return out


def _save_originals(root, originals):
    path = root / ORIGINALS
    if path.exists():
        return
    np.savez_compressed(path, **{f"{rel}::{k}": v for rel, cols in originals.items()
                                 for k, v in cols.items()})


def _load_columns(root, originals):
    """All rows, in file order: {feature: (N, width)}, episode_index (N,), and the files'
    row counts."""
    import pyarrow.parquet as pq
    files = _data_files(root)
    cols = {k: np.concatenate([originals[str(f.relative_to(root))][k] for f in files])
            for k in FEATURES}
    episode = np.concatenate([pq.read_table(f, columns=["episode_index"])["episode_index"]
                              .to_numpy() for f in files])
    return cols, episode, [pq.read_metadata(f).num_rows for f in files]


def _write_columns(root, cols, rows):
    import pyarrow.parquet as pq
    start = 0
    for f, n in zip(_data_files(root), rows):
        table = pq.read_table(f)
        for k in FEATURES:
            table = _set_fixed(table, k, cols[k][start:start + n].astype(np.float32))
        _write_table(table, f)
        start += n


# --------------------------------------------------------------------- stats ---

def compute_stats(x):
    """LeRobot's stats for one feature, from its rows."""
    x = np.asarray(x, dtype=np.float64)
    out = {"min": x.min(0), "max": x.max(0), "mean": x.mean(0), "std": x.std(0),
           "count": np.array([len(x)])}
    for q in QUANTILES:
        out[f"q{q:02d}"] = np.percentile(x, q, axis=0)
    return {k: v.tolist() for k, v in out.items()}


def _episode_ends(episode):
    change = np.nonzero(np.diff(episode))[0] + 1
    return np.concatenate([change, [len(episode)]])


def _write_episode_stats(root, cols, episode):
    """meta/episodes: per-episode stats of state and action, from the data as written (LeRobot
    aggregates these when a dataset is extended)."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    for path in sorted((root / "meta" / "episodes").glob("*/*.parquet")):
        table = pq.read_table(path)
        index = table["episode_index"].to_pylist()
        per_episode = {e: {k: compute_stats(cols[k][episode == e]) for k in FEATURES}
                       for e in index}
        for i, field in enumerate(table.schema):
            parts = field.name.split("/")
            if len(parts) != 3 or parts[0] != "stats" or parts[1] not in FEATURES:
                continue
            stat = parts[2]
            if stat not in per_episode[index[0]][parts[1]]:
                continue
            rows = [per_episode[e][parts[1]][stat] for e in index]
            table = table.set_column(i, field, pa.array(rows, type=field.type))
        _write_table(table, path)


def _relative_chunks(cols, episode, names, exclude, chunk_size):
    """Every action chunk in relative space, as LeRobot's relative action stats see them."""
    from lerobot.datasets.compute_stats import (_compute_relative_chunk_batch,
                                                _get_valid_chunk_starts)
    from lerobot.processor.relative_action_processor import RelativeActionsProcessorStep

    action = cols[ACTION].astype(np.float32)
    mask = RelativeActionsProcessorStep(enabled=True, exclude_joints=list(exclude),
                                        action_names=list(names))._build_mask(action.shape[1])
    starts = _get_valid_chunk_starts(episode, chunk_size)
    return _compute_relative_chunk_batch(starts, action, cols[STATE].astype(np.float32),
                                         chunk_size, np.asarray(mask, dtype=np.float32))


# ------------------------------------------------------------------- prepare ---

def prepare(dataset, relative=False, chunk_size=50, exclude=RELATIVE_EXCLUDE, tail=None,
            min_episode_frac=DEFAULT_MIN_EPISODE_FRAC, min_motion=None, hold=(), keep=(),
            auto_hold=True, verbose=True):
    """Prepare the dataset in place (see the module docstring). -> the "prepared" record."""
    root = resolve(dataset)
    side = sidecar.require_current(root)
    info = json.loads((root / "meta" / "info.json").read_text())
    names = {k: list(info["features"][k]["names"]) for k in FEATURES}
    problem = check_state_names(names[STATE], names[ACTION])
    if problem:
        sys.exit(f"{root}: {problem}; this dataset predates the shared columns")

    originals = _originals(root)
    cols, episode, rows = _load_columns(root, originals)
    ends = _episode_ends(episode)

    held, report = decide_held_joints(
        cols[ACTION], names[ACTION], ends,
        episode_names=[f"episode {int(episode[e - 1])}" for e in ends],
        min_episode_frac=min_episode_frac, min_motion=min_motion,
        hold_joints=list(hold), keep_joints=list(keep), auto=auto_hold)
    if verbose:
        print(format_hold_report(report))
    for joint, h in held.items():
        for k in FEATURES:
            cols[k][:, names[k].index(joint)] = h["value"]

    _save_originals(root, originals)
    _write_columns(root, cols, rows)
    _write_episode_stats(root, cols, episode)

    # Stats: from the data as written, relative actions as LeRobot will see them, then tails
    # and floors. Image stats are left as they are.
    tail = (0.1 if relative else 1.0) if tail is None else tail
    action_rows = (_relative_chunks(cols, episode, names[ACTION], exclude, chunk_size)
                   if relative else cols[ACTION])
    stats_path = root / "meta" / "stats.json"
    stats = json.loads(stats_path.read_text()) if stats_path.exists() else {}
    floored = {}
    for k, x in ((STATE, cols[STATE]), (ACTION, action_rows)):
        st = compute_stats(x)
        st["q01"] = np.percentile(np.asarray(x, np.float64), tail, axis=0).tolist()
        st["q99"] = np.percentile(np.asarray(x, np.float64), 100 - tail, axis=0).tolist()
        changed = floor_stats(st, names[k])
        if changed:
            floored[k] = changed
        stats[k] = st
    tmp = stats_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(stats, indent=4))
    os.replace(tmp, stats_path)

    record = {
        "tool": "stretch4_fix_stats", "version": VERSION,
        "frames": int(info.get("total_frames", len(episode))),
        "episodes": int(info.get("total_episodes", len(ends))),
        "relative": bool(relative),
        "relative_exclude": list(exclude) if relative else None,
        "chunk_size": int(chunk_size) if relative else None,
        "tail": float(tail),
        "hold": {"auto": bool(auto_hold), "min_episode_frac": float(min_episode_frac),
                 "min_motion": {**DEFAULT_MIN_MOTION, **(min_motion or {})},
                 "hold_joints": list(hold), "keep_joints": list(keep)},
        "held_joints": held,
        "floored": floored,
        "joints": report,
    }
    side["prepared"] = record
    sidecar.save(root, side)
    if verbose:
        print(f"{root}: {len(episode)} frames, {len(ends)} episodes"
              f"{', relative action stats' if relative else ''}; held: "
              f"{', '.join(held) or 'none'}")
        for k, joints in floored.items():
            print(f"  {k} stats floored (spread before -> after):")
            for n, fixed in joints.items():
                print(f"    {n:12s} " + ", ".join(f"{stat} {before:.3g} -> {after:.3g}"
                                                  for stat, (before, after) in fixed.items()))
    return record


def undo(dataset):
    """Put the recorded state/action columns and stats back; drop "prepared"."""
    root = resolve(dataset)
    if not (root / ORIGINALS).exists():
        sys.exit(f"{root} was not prepared (no {ORIGINALS})")
    originals = _originals(root)
    cols, episode, rows = _load_columns(root, originals)
    _write_columns(root, cols, rows)
    _write_episode_stats(root, cols, episode)
    stats_path = root / "meta" / "stats.json"
    stats = json.loads(stats_path.read_text())
    for k in FEATURES:
        stats[k] = compute_stats(cols[k])
    stats_path.write_text(json.dumps(stats, indent=4))
    side = sidecar.read(root)
    if side:
        side.pop("prepared", None)
        sidecar.save(root, side)
    (root / ORIGINALS).unlink()
    print(f"{root}: restored as recorded")


def is_stale(root, side=None):
    """Why a prepared dataset no longer matches its data, or None. Also None when unprepared."""
    side = side if side is not None else sidecar.read(root)
    prepared = (side or {}).get("prepared")
    if not prepared:
        return None
    info = json.loads((Path(root) / "meta" / "info.json").read_text())
    now = (int(info.get("total_frames", -1)), int(info.get("total_episodes", -1)))
    then = (prepared.get("frames"), prepared.get("episodes"))
    if now != then:
        return (f"it has {now[0]} frames / {now[1]} episodes, but was prepared with "
                f"{then[0]} / {then[1]}")
    return None


def options_from(record):
    """prepare() keyword arguments that redo what a "prepared" record did, so a dataset that
    changed since is prepared again the same way (--relative above all). {} for none."""
    record = record or {}
    hold = record.get("hold") or {}
    opts = {"relative": record.get("relative"), "chunk_size": record.get("chunk_size"),
            "exclude": record.get("relative_exclude"), "tail": record.get("tail"),
            "min_episode_frac": hold.get("min_episode_frac"), "min_motion": hold.get("min_motion"),
            "hold": hold.get("hold_joints"), "keep": hold.get("keep_joints"),
            "auto_hold": hold.get("auto")}
    out = {k: v for k, v in opts.items() if v is not None}
    for k in ("exclude", "hold", "keep"):
        if k in out:
            out[k] = tuple(out[k])
    return out


def _min_motion(items):
    out = {}
    for item in items or []:
        joint, _, value = item.partition("=")
        if joint not in DEFAULT_MIN_MOTION or not value:
            sys.exit(f"--min-motion takes joint=value with joint one of {list(DEFAULT_MIN_MOTION)}")
        out[joint] = float(value)
    return out


def main(argv=None):
    p = argparse.ArgumentParser(prog="stretch4_fix_stats", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("dataset", help="repo id under $HF_LEROBOT_HOME, or a dataset folder")
    p.add_argument("--relative", action="store_true",
                   help="action stats in relative space, for pi0.5 --policy.use_relative_actions")
    p.add_argument("--chunk-size", type=int, default=50, help="--relative: the policy's chunk_size")
    p.add_argument("--exclude-joints", nargs="+", default=list(RELATIVE_EXCLUDE),
                   help="--relative: action names kept absolute, as substrings "
                        f"(default {RELATIVE_EXCLUDE}, as pi0.5's relative_exclude_joints)")
    p.add_argument("--tail", type=float, default=None,
                   help="percent each tail may leave out of q01/q99 (default 1, LeRobot's own; "
                        "0.1 with --relative)")
    p.add_argument("--min-episode-frac", type=float, default=DEFAULT_MIN_EPISODE_FRAC,
                   help="hold a joint moved in fewer than this fraction of the episodes")
    p.add_argument("--min-motion", nargs="+", metavar="JOINT=VALUE",
                   help=f"within-episode range that counts as moved (defaults {DEFAULT_MIN_MOTION})")
    p.add_argument("--hold", nargs="+", default=[], help="always hold these joints")
    p.add_argument("--keep", nargs="+", default=[], help="never hold these joints")
    p.add_argument("--no-auto-hold", action="store_true", help="hold only --hold joints")
    p.add_argument("--undo", action="store_true", help="restore the dataset as recorded")
    a = p.parse_args(argv)
    if a.undo:
        undo(a.dataset)
        return
    prepare(a.dataset, relative=a.relative, chunk_size=a.chunk_size, exclude=a.exclude_joints,
            tail=a.tail, min_episode_frac=a.min_episode_frac, min_motion=_min_motion(a.min_motion),
            hold=a.hold, keep=a.keep, auto_hold=not a.no_auto_hold)


if __name__ == "__main__":
    main()
