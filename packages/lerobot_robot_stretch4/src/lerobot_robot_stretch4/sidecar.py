"""meta/stretch4.json: how a Stretch 4 LeRobot dataset was made.

Stretch4.describe() fills it in at connect, into the dataset being recorded; stretch4_fix_stats
adds "prepared". --robot.preprocessing_from=<dataset> reads it back to rebuild the exact cameras
(physical eye, crop, rotation; the size comes from meta/info.json's image shapes) and the held
joints. It lives in meta/ so push_to_hub uploads it with the dataset:

    {
      "schema": "stretch4 dataset v3",
      "source": "lerobot-record",    # the tool that wrote the dataset
      "robot_type": "stretch4",
      "cameras": {"head_left": {"camera": "head_left", "crop": [x, y, w, h] | null,
                                "size": [h, w], "rotate": 1, "height": h, "width": w}, ...},
      "state_names": [...],          # columns.POLICY_STATE_COLUMNS
      "action_names": [...],         # columns.VELOCITY_ACTION_COLUMNS
      "action_mode": "absolute",
      "base_action": "velocity",
      "lookahead": 3,                # action[t] = measured value at t + lookahead
      "sync": {"clock": "head_left", "max_skew_ms": 20.0, "match_timeout_ms": 120.0},
      "gripper_tool": "stretch_gripper" | "parallel_gripper",   # SG4 | PG4; absent if unknown
      "prepared": {...}              # stretch4_fix_stats; absent until it has run
    }

v3 = the shared state order (base velocity first), no observation.base_odom /
observation.gripper columns, the gripper frame waited for. Older sidecars are refused.
"""

import inspect
import json
import logging
from pathlib import Path

from .columns import DEFAULT_GRIPPER_TOOL

SIDECAR = "meta/stretch4.json"
SCHEMA = "stretch4 dataset v3"

log = logging.getLogger(__name__)


def make(source, cameras, state_names, action_names, lookahead, sync,
         action_mode="absolute", base_action="velocity", gripper_tool=None):
    """A v3 sidecar. `cameras` is {column: {"camera", "crop", "size", "rotate"}}; height and
    width are filled in from size."""
    cams = {}
    for name, cam in cameras.items():
        size = list(cam["size"]) if cam.get("size") else None
        cams[name] = {"camera": cam["camera"], "crop": list(cam["crop"]) if cam.get("crop") else None,
                      "size": size, "rotate": cam.get("rotate"),
                      "height": size[0] if size else None, "width": size[1] if size else None}
    return {
        "schema": SCHEMA, "source": source, "robot_type": "stretch4",
        "cameras": cams,
        "state_names": list(state_names), "action_names": list(action_names),
        "action_mode": action_mode, "base_action": base_action,
        "lookahead": int(lookahead),
        "sync": dict(sync),
        **({"gripper_tool": gripper_tool} if gripper_tool else {}),
    }


def read(root):
    """The sidecar of the dataset at `root`, or None."""
    path = Path(root) / SIDECAR
    return json.loads(path.read_text()) if path.exists() else None


def save(root, side):
    """Write atomically, so a reader never sees half a file. Writes `side` as it is; write()
    is the recording path's, which also keeps what stretch4_fix_stats recorded."""
    path = Path(root) / SIDECAR
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(side, indent=2))
    tmp.replace(path)
    return path


def require_current(root):
    """The dataset's sidecar, or an error naming what to do if it is missing or older than v3."""
    side = read(root)
    if side is None:
        raise FileNotFoundError(f"{root} has no {SIDECAR}: not a Stretch 4 dataset, or recorded "
                                "before the sidecar existed")
    if side.get("schema") != SCHEMA:
        raise ValueError(f"{root}/{SIDECAR} is {side.get('schema')!r}, this code needs {SCHEMA!r} "
                         "(shared state order, synced gripper frames). Re-record the dataset; "
                         "old checkpoints run from the pre-conformance tag.")
    return side


def write(root, info: dict) -> Path:
    """Write atomically, keeping what stretch4_fix_stats recorded. Warn if the dataset already
    records different cameras: resuming a dataset with other crops/eyes would mix two
    preprocessings in one dataset. Raise ValueError if the dataset was recorded with the other
    gripper (SG4 vs PG4): one dataset holds one gripper."""
    old = read(root)
    if old and info.get("gripper_tool"):
        # A sidecar from before the PG4 has no tag; those were all SG4.
        had = old.get("gripper_tool") or DEFAULT_GRIPPER_TOOL
        if had != info["gripper_tool"]:
            raise ValueError(f"{Path(root) / SIDECAR}: this dataset was recorded with {had}, the "
                             f"robot has {info['gripper_tool']}. Record into a new dataset.")
    if old and old.get("cameras") != info.get("cameras"):
        log.error("%s: this session's cameras differ from the ones the dataset was started with; "
                  "episodes in this dataset now use different preprocessing.\n  before: %s\n  now:    %s",
                  Path(root) / SIDECAR, old.get("cameras"), info.get("cameras"))
    if old and "prepared" in old:
        info = {**info, "prepared": old["prepared"]}
    return save(root, info)


def recording_dataset_root(max_depth: int = 12) -> Path | None:
    """Root of the LeRobotDataset the calling script is recording into, if any.

    LeRobot does not tell a Robot which dataset it records into; lerobot-record (and the
    lerobot-rollout recording strategies) hold it in a local variable of the function that
    calls robot.connect(), so it is found by looking up the call stack."""
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ImportError:
        return None
    frame = inspect.currentframe()
    try:
        for _ in range(max_depth):
            frame = frame.f_back
            if frame is None:
                return None
            for value in frame.f_locals.values():
                if isinstance(value, LeRobotDataset):
                    return Path(value.root)
        return None
    finally:
        del frame
