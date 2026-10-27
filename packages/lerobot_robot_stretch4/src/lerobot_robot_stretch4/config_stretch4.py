"""Config for the Stretch 4 LeRobot Robot (--robot.type=stretch4)."""

import dataclasses
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

from lerobot.cameras import CameraConfig
from lerobot.robots import RobotConfig

from lerobot_camera_oak import CAMERAS, OAKCameraConfig

log = logging.getLogger(__name__)

# Recorded (height, width) of every camera. Equal for all cameras because LeRobot's diffusion
# policy requires every image to have the same shape (it checks this before any resize). 1.6:1
# is the gripper camera's own aspect, so the gripper is recorded whole; the head is cropped to it.
DEFAULT_IMAGE_SIZE = [240, 384]

# action[t] = the measured value at t + ACTION_LOOKAHEAD: 3 frames (100 ms at 30 fps) is about
# the teleop-command-to-motion lag. Every Stretch 4 dataset uses the same lead.
ACTION_LOOKAHEAD = 3
# Append grip_effort to observation.state when recording.
INCLUDE_GRIP_EFFORT = False
# Camera pairing per mode. Recording can wait longer for the matching frame; control needs a
# fresh observation every tick.
MAX_SKEW_MS = {"observe": 20.0, "control": 20.0}
MATCH_TIMEOUT_MS = {"observe": 120.0, "control": 60.0}


def default_cameras() -> dict[str, CameraConfig]:
    # The right head eye and the right gripper eye. The robot gives them its image_size.
    return {
        "head_right": OAKCameraConfig(camera="head_right"),
        "gripper": OAKCameraConfig(camera="gripper_right"),
    }


def _source(name, camera):
    """The physical camera behind a dataset image column."""
    if camera in CAMERAS:
        return camera
    if name in CAMERAS:
        return name
    return "gripper_right" if name.startswith("gripper") else "head_right"


CAMERAS_FILE_SCHEMA = "stretch4 cameras v1"


def cameras_from_file(path: Path) -> dict[str, CameraConfig]:
    """Camera configs from a cameras file (stretch4_pick_cameras writes them):

        {"schema": "stretch4 cameras v1",
         "cameras": {"head_right": {"camera": "head_right", "crop": [x, y, w, h], "size": [240, 384]},
                     "gripper":    {"camera": "gripper_right", "crop": null, "size": [240, 384]}}}

    Keys are the dataset columns (observation.images.<key>). crop is in upright native pixels and
    has the aspect of size; null means the largest centred crop of that aspect."""
    data = json.loads(Path(path).expanduser().read_text())
    if data.get("schema") != CAMERAS_FILE_SCHEMA or not isinstance(data.get("cameras"), dict):
        raise ValueError(f"{path} is not a Stretch 4 cameras file (schema {CAMERAS_FILE_SCHEMA!r}); "
                         "make one with stretch4_pick_cameras")
    cameras = {}
    for name, cam in data["cameras"].items():
        unknown = set(cam) - {"camera", "crop", "size", "rotate"}
        if unknown:
            raise ValueError(f"{path}: camera {name!r} has unknown fields {sorted(unknown)}")
        cameras[name] = OAKCameraConfig(camera=_source(name, cam.get("camera")), crop=cam.get("crop"),
                                        size=cam.get("size"), rotate=cam.get("rotate"))
    if not cameras:
        raise ValueError(f"{path} lists no cameras")
    return cameras


def _feature_names(info, key):
    names = info["features"].get(key, {}).get("names") or []
    return next(iter(names.values()), []) if isinstance(names, dict) else list(names)


def _action_names(info):
    return _feature_names(info, "action")


def check_dataset_columns(root: Path, info: dict) -> None:
    """Refuse a dataset this robot cannot feed: the shared columns, state in the action's order
    (optionally followed by grip_effort)."""
    from .columns import VELOCITY_ACTION_COLUMNS, check_state_names
    action, state = _action_names(info), _feature_names(info, "observation.state")
    problem = (f"action {action} is not {VELOCITY_ACTION_COLUMNS}"
               if action != VELOCITY_ACTION_COLUMNS else check_state_names(state, action))
    if problem:
        raise ValueError(
            f"{root}: {problem}. These are not the shared Stretch 4 columns, and the policy would "
            "read the state scrambled. Re-record it and retrain, or run old checkpoints from the pre-conformance tag.")


def dataset_grip_effort(root: Path) -> bool:
    """Whether the dataset's observation.state ends with grip_effort."""
    from .columns import GRIP_EFFORT
    info = json.loads((root / "meta" / "info.json").read_text())
    return _feature_names(info, "observation.state")[-1:] == [GRIP_EFFORT]


def cameras_from_dataset(root: Path) -> dict[str, CameraConfig]:
    """Camera configs that reproduce a dataset's images exactly, from its meta/stretch4.json
    (eye, crop, rotation) and meta/info.json (the recorded shape)."""
    from .sidecar import SIDECAR, require_current

    info = json.loads((root / "meta" / "info.json").read_text())
    check_dataset_columns(root, info)
    side = require_current(root)
    fps = int(info.get("fps", 30))
    cameras = {}
    for key, ft in info["features"].items():
        if not key.startswith("observation.images."):
            continue
        name = key.removeprefix("observation.images.")
        cam = side["cameras"].get(name)
        if cam is None:
            raise ValueError(f"{root}: {SIDECAR} does not describe camera {name!r}")
        cameras[name] = OAKCameraConfig(camera=_source(name, cam.get("camera")), crop=cam.get("crop"),
                                        size=list(ft["shape"][:2]), rotate=cam.get("rotate"), fps=fps)
    return cameras


# Smallest padding per side, so a joint the demos barely moved is not pinned to one value.
MIN_PAD = {"lift": 0.02, "arm": 0.02, "wrist_yaw": 0.1, "wrist_pitch": 0.1, "wrist_roll": 0.1,
           "grip_mm": 5.0}


def _action_range(root: Path):
    """(min, max) per action value, absolute: from the per-episode stats in meta/episodes/.
    meta/stats.json is no source for this: after stretch4_fix_stats --relative its action stats
    are action - state. None if the dataset has no per-episode action stats."""
    import pyarrow.parquet as pq
    lo = hi = None
    for path in sorted((root / "meta" / "episodes").glob("*/*.parquet")):
        if "stats/action/min" not in pq.read_schema(path).names:
            return None
        t = pq.read_table(path, columns=["stats/action/min", "stats/action/max"])
        mins = [min(col) for col in zip(*t["stats/action/min"].to_pylist())]
        maxs = [max(col) for col in zip(*t["stats/action/max"].to_pylist())]
        lo = mins if lo is None else [min(a, b) for a, b in zip(lo, mins)]
        hi = maxs if hi is None else [max(a, b) for a, b in zip(hi, maxs)]
    return None if lo is None else (lo, hi)


def bounds_from_dataset(root: Path, names, margin=0.05):
    """{joint: [lo, hi]} from the dataset's absolute action range (meta/episodes/), padded by
    `margin` of the range and at least MIN_PAD."""
    info = json.loads((root / "meta" / "info.json").read_text())
    action_names = _action_names(info)
    action_range = _action_range(root)
    if action_range is None or not action_names:
        log.warning("%s has no per-episode action stats (meta/episodes/): no joint bounds from "
                    "it", root)
        return None
    out = {}
    for j in names:
        if j in action_names:
            i = action_names.index(j)
            lo, hi = float(action_range[0][i]), float(action_range[1][i])
            pad = max((hi - lo) * margin, MIN_PAD.get(j, 0.0))
            out[j] = [lo - pad, hi + pad]
    return out


@RobotConfig.register_subclass("stretch4")
@dataclass(kw_only=True)
class Stretch4Config(RobotConfig):
    """Hello Robot Stretch 4, through stretch_body_server.

    Action = base_vx, base_vy, base_omega (body-frame velocity) + lift, arm, wrist yaw/pitch/roll,
    grip_mm (absolute). observation.state = those 6 joints + the base velocity. Any LeRobot
    policy trains on this as-is.

    mode="observe" (recording): never commands the robot. Drive it with any teleop running as
    its own process (gamepad, SpaceMouse, Quest, ...) and record with
    --teleop.type=stretch4_observer. Observations come action_lookahead ticks late, so each
    recorded action is the measured value action_lookahead frames after its observation
    (lookahead.py): the action leads the state, like a teleop command.
    mode="control" (rollout): send_action drives the arm, gripper and base.
    """

    cameras: dict[str, CameraConfig] = field(default_factory=default_cameras)
    # Recorded [height, width] of every camera that does not set its own size. Keep it equal
    # across cameras for LeRobot's diffusion policy, which assumes every image has one shape.
    image_size: list[int] = field(default_factory=lambda: list(DEFAULT_IMAGE_SIZE))
    # A cameras file from stretch4_pick_cameras (eye, crop and size per camera). Overrides
    # `cameras`; for recording. Deployment takes the cameras from the dataset instead
    # (preprocessing_from), which records whatever this file set.
    cameras_file: str | None = None
    mode: str = "observe"
    # A trained dataset's root: take cameras (eye, crop, size, rotation) and joint bounds from
    # it, so deployment preprocesses exactly like training did. Overrides `cameras`.
    preprocessing_from: str | None = None
    # The OAK camera whose frames clock each observation; the others are matched to it by
    # timestamp and the joints
    # are interpolated to it. Default: the first OAK camera.
    sync_camera: str | None = None
    # Pair frames at most this far apart in capture time; a pair further apart is skipped and the
    # next sync frame taken. None: MAX_SKEW_MS (20 ms).
    max_skew_ms: float | None = None
    # Wait at most this long for the other cameras' matching frames. A ceiling, not a delay. 
    # None: MATCH_TIMEOUT_MS (120 ms observe, 60 ms control).
    match_timeout_ms: float | None = None
    joint_poll_hz: float = 200.0
    # observe mode: record action[t] = the value at t + action_lookahead ticks, by handing
    # lerobot-record each observation that many ticks late. None: ACTION_LOOKAHEAD (3, as
    # every Stretch 4 dataset) in observe mode, 0 in
    # control mode (a policy must see the current observation; >0 there is an error).
    action_lookahead: int | None = None
    # Append the gripper's effort (%, 0..100) as the last value of observation.state. None:
    # INCLUDE_GRIP_EFFORT in observe mode; the preprocessing_from dataset's state
    # in control mode (an explicit value that disagrees with it is an error).
    grip_effort: bool | None = None
    # LeRobot's spoken announcements: "stretch" = the Stretch 4 voice (Google TTS, cached;
    # see voice.py), "system" = LeRobot's default (spd-say). --play_sounds=false silences both.
    voice: str = "stretch"

    # ---- control mode ----
    use_base: bool = True
    use_gripper: bool = True
    priority: int = 2
    # False: respect the control lock, so a teleop or e-stop tool can take over.
    ignore_control_lock: bool = False
    # move_to speed/acceleration limits (m/s, m/s^2, rad/s, rad/s^2).
    limits: dict[str, float] = field(default_factory=lambda: {
        "v_m": 0.10, "a_m": 0.25, "v_r": 1.5, "a_r": 4.0})
    # Caps on the commanded base velocity (m/s, rad/s).
    base_max_lin: float = 0.3
    base_max_ang: float = 0.6
    # {joint: [lo, hi]} clamp for commanded joints. Default: from preprocessing_from's actions.
    joint_bounds: dict[str, list[float]] | None = None
    # Halt the base if send_action stops arriving for this long.
    base_timeout_s: float = 0.3
    # Keep the state inside the training range:
    # pin the joints stretch4_fix_stats held, hold them still, clamp the rest to the
    # preprocessing_from dataset's range. Changes nothing while the robot is inside it.
    state_guard: bool = True

    def __post_init__(self):
        if self.mode not in ("observe", "control"):
            raise ValueError(f"mode must be 'observe' or 'control', got {self.mode!r}")
        if self.action_lookahead is None:
            self.action_lookahead = ACTION_LOOKAHEAD if self.mode == "observe" else 0
        if self.max_skew_ms is None:
            self.max_skew_ms = MAX_SKEW_MS[self.mode]
        if self.match_timeout_ms is None:
            self.match_timeout_ms = MATCH_TIMEOUT_MS[self.mode]
        if self.action_lookahead < 0:
            raise ValueError(f"action_lookahead must be >= 0, got {self.action_lookahead}")
        if self.mode == "control" and self.action_lookahead:
            raise ValueError("action_lookahead delays observations and is for recording "
                             "(mode='observe'); in control mode the policy must see the current "
                             "observation. Leave it unset or pass 0.")
        if self.voice not in ("stretch", "system"):
            raise ValueError(f"voice must be 'stretch' or 'system', got {self.voice!r}")
        if len(self.image_size) != 2 or min(self.image_size) < 1:
            raise ValueError(f"image_size must be [height, width], got {self.image_size}")
        if self.cameras_file and self.preprocessing_from:
            raise ValueError("pass either cameras_file (recording) or preprocessing_from "
                             "(deployment: the training dataset's cameras), not both")
        if self.cameras_file:
            self.cameras = cameras_from_file(Path(self.cameras_file))
        if self.preprocessing_from:
            root = Path(self.preprocessing_from).expanduser()
            self.cameras = cameras_from_dataset(root)
            trained = dataset_grip_effort(root)
            if self.grip_effort is not None and self.grip_effort != trained:
                raise ValueError(f"grip_effort={self.grip_effort}, but {root} was recorded "
                                 f"{'with' if trained else 'without'} grip_effort in the state")
            self.grip_effort = trained
            if self.joint_bounds is None:
                from .columns import POSE_JOINTS
                self.joint_bounds = bounds_from_dataset(root, POSE_JOINTS)
        if self.grip_effort is None:
            self.grip_effort = INCLUDE_GRIP_EFFORT if self.mode == "observe" else False
        # Cameras without their own size get image_size (cropped to its aspect, then resized).
        self.cameras = {
            k: (dataclasses.replace(c, size=list(self.image_size), width=None, height=None)
                if isinstance(c, OAKCameraConfig) and c.size is None else c)
            for k, c in self.cameras.items()}
        oak = [k for k, c in self.cameras.items() if isinstance(c, OAKCameraConfig)]
        if not oak:
            raise ValueError("stretch4 needs at least one OAK camera to clock observations")
        shapes = {(c.height, c.width) for c in self.cameras.values()}
        if len(shapes) > 1:
            log.warning("cameras have different shapes %s: LeRobot's diffusion policy needs one shape "
                        "for every camera", sorted(shapes))
        if self.sync_camera is None:
            self.sync_camera = oak[0]
        elif self.sync_camera not in oak:
            raise ValueError(f"sync_camera {self.sync_camera!r} is not one of the OAK cameras {oak}")
        super().__post_init__()
