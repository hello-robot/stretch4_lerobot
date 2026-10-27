"""The plugin pairs camera frames by capture time (waits for the slower camera's matching frame,
skips a pair that never matches), and in control mode keeps the state inside
the training range (state_guard.py): held joints pinned and kept where they are."""

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from lerobot_camera_oak import OAKCamera, OAKCameraConfig
from lerobot_camera_oak.capture import FrameBuffer
from lerobot_robot_stretch4 import Stretch4, Stretch4Config, sidecar
from lerobot_robot_stretch4.columns import POLICY_STATE_COLUMNS


class FakeCam(OAKCamera):
    """An OAK camera whose frames come from a FrameBuffer the test fills; decode is identity."""

    def __init__(self, buf):
        self.buf = buf

    def _buffer(self):
        return self.buf

    def latest_raw(self):
        t, frame, _ = self.buf.latest()
        return t, frame

    def decode(self, jpeg):
        return jpeg


JOINTS = {"lift": {"pos": 0.8}, "arm": {"pos": 0.2}, "wrist_yaw": {"pos": 0.0},
          "wrist_pitch": {"pos": 0.6}, "wrist_roll": {"pos": 0.07},
          "gripper": {"aperture_mm": 41.0, "pos": 0.0},
          "omnibase": {"x_vel": 0.0, "y_vel": 0.0, "theta_vel": 0.0}}


def head_and_gripper(**config):
    cameras = {"head_left": OAKCameraConfig(camera="head_left"),
               "gripper": OAKCameraConfig(camera="gripper_right")}
    robot = Stretch4(Stretch4Config(cameras=cameras, **config))
    head, grip = FrameBuffer(), FrameBuffer()
    robot.cameras = {"head_left": FakeCam(head), "gripper": FakeCam(grip)}
    robot._decoder = ThreadPoolExecutor(2)
    robot._joints_at = lambda t, timeout=1.0: JOINTS
    return robot, head, grip


def test_waits_for_the_matching_gripper_frame():
    robot, head, grip = head_and_gripper(mode="control")
    assert robot.config.max_skew_ms == 20 and robot.config.match_timeout_ms == 60
    head.put(1.000, "head@1.000")
    grip.put(0.958, "grip@0.958")             # already here, but 42 ms older: not a match
    # the matching gripper frame (9 ms before the head's) arrives 30 ms later, as on the robot
    threading.Timer(0.03, lambda: grip.put(0.991, "grip@0.991")).start()
    t0 = time.monotonic()
    obs, state, t_ref = robot._sample()
    assert obs["gripper"] == "grip@0.991" and obs["head_left"] == "head@1.000"
    assert 0.02 < time.monotonic() - t0 < 0.06    # waited for it, no longer
    assert robot._skew_drops == 0
    assert list(state) == POLICY_STATE_COLUMNS


def test_no_matching_frame_skips_the_pair():
    robot, head, grip = head_and_gripper(mode="control", match_timeout_ms=20)
    head.put(1.000, "head")
    grip.put(0.958, "stale gripper")
    assert robot._sample() is None            # get_observation takes the next head frame
    assert robot._skew_drops == 1


# ------------------------------------------------------------------ guard ---

def prepared_dataset(root, held):
    """meta/ of a v3 dataset prepared by stretch4_fix_stats: state range lift 0.7..0.9."""
    (root / "meta").mkdir(parents=True)
    feats = {"observation.images.head_left": {"dtype": "video", "shape": [240, 384, 3]},
             "observation.images.gripper": {"dtype": "video", "shape": [240, 384, 3]},
             "observation.state": {"names": POLICY_STATE_COLUMNS},
             "action": {"names": POLICY_STATE_COLUMNS}}
    (root / "meta" / "info.json").write_text(json.dumps(
        {"fps": 30, "total_frames": 100, "total_episodes": 2, "features": feats}))
    lo = {n: -1.0 for n in POLICY_STATE_COLUMNS}
    hi = {n: 1.0 for n in POLICY_STATE_COLUMNS}
    lo["lift"], hi["lift"] = 0.7, 0.9
    lo["grip_mm"], hi["grip_mm"] = 35.0, 45.0
    rng = {"min": [lo[n] for n in POLICY_STATE_COLUMNS], "max": [hi[n] for n in POLICY_STATE_COLUMNS]}
    (root / "meta" / "stats.json").write_text(json.dumps({"observation.state": rng, "action": rng}))
    side = sidecar.make("lerobot-record",
                        {"head_left": {"camera": "head_left", "crop": None, "size": [240, 384]},
                         "gripper": {"camera": "gripper_right", "crop": None, "size": [240, 384]}},
                        POLICY_STATE_COLUMNS, POLICY_STATE_COLUMNS, 3, {"clock": "head_left"})
    side["prepared"] = {"frames": 100, "episodes": 2,
                        "held_joints": {j: {"value": v, "reason": "test"} for j, v in held.items()}}
    sidecar.save(root, side)
    return root


def guarded_robot(tmp_path, held):
    root = prepared_dataset(tmp_path / "ds", held)
    robot = Stretch4(Stretch4Config(mode="control", preprocessing_from=str(root)))
    robot._connect_guard()
    return robot


def test_guard_pins_held_joints_and_clamps_the_rest(tmp_path):
    robot = guarded_robot(tmp_path, {"grip_mm": 40.0})
    inside = {n: 0.0 for n in POLICY_STATE_COLUMNS}
    inside.update(lift=0.8, grip_mm=40.0)
    assert robot._guarded({**inside, "head_left": "img"}) == {**inside, "head_left": "img"}

    out = robot._guarded({**inside, "lift": 1.2, "grip_mm": 41.3})
    assert out["grip_mm"] == 40.0             # held: the training constant, not the reading
    assert out["lift"] == 0.9                 # clamped to the training range
    assert robot._guard_far == 1              # 0.3 m outside: reported


def test_held_joints_are_kept_where_they_are(tmp_path):
    robot = guarded_robot(tmp_path, {"grip_mm": 40.0})
    sent = []
    robot._arm = SimpleNamespace(last={"lift": 0.8}, send=lambda pose: sent.append(pose) or pose)
    robot._tracker = robot._cmd_robot = SimpleNamespace(push_command=lambda **k: None,
                                                        command_velocity=lambda v: None)
    robot._joints_at = lambda t, timeout=1.0: JOINTS
    robot.send_action({**{n: 0.0 for n in POLICY_STATE_COLUMNS}, "lift": 0.85, "grip_mm": 10.0})
    assert sent[0]["lift"] == 0.85            # learned: as the policy says
    assert sent[0]["grip_mm"] == 41.0         # held: where the gripper is now, not 10 mm


def test_unprepared_dataset_warns(tmp_path, caplog):
    root = prepared_dataset(tmp_path / "ds", {})
    side = sidecar.read(root)
    del side["prepared"]
    sidecar.save(root, side)
    robot = Stretch4(Stretch4Config(mode="control", preprocessing_from=str(root)))
    with caplog.at_level("WARNING"):
        robot._connect_guard()
    assert "stretch4_fix_stats" in caplog.text


def test_sync_defaults_per_mode():
    # recording may wait longer for the matching frame; control needs a fresh observation
    rec = Stretch4Config(mode="observe")
    assert (rec.action_lookahead, rec.max_skew_ms, rec.match_timeout_ms) == (3, 20.0, 120.0)
    run = Stretch4Config(mode="control")
    assert (run.action_lookahead, run.max_skew_ms, run.match_timeout_ms) == (0, 20.0, 60.0)


# ------------------------------------------------------------- grip effort ---

def test_grip_effort_observed_when_on():
    robot, head, grip = head_and_gripper(mode="control", grip_effort=True)
    robot._joints_at = lambda t, timeout=1.0: {
        **JOINTS, "gripper": {**JOINTS["gripper"], "effort": 37.0}}
    head.put(1.000, "head")
    grip.put(0.995, "grip")
    _, state, _ = robot._sample()
    assert list(state) == [*POLICY_STATE_COLUMNS, "grip_effort"] and state["grip_effort"] == 37.0
    assert "grip_effort" not in robot.action_names


def test_grip_effort_follows_the_training_dataset(tmp_path):
    root = prepared_dataset(tmp_path / "ds", {})
    info = json.loads((root / "meta" / "info.json").read_text())
    info["features"]["observation.state"]["names"] = [*POLICY_STATE_COLUMNS, "grip_effort"]
    (root / "meta" / "info.json").write_text(json.dumps(info))
    cfg = Stretch4Config(mode="control", preprocessing_from=str(root))
    assert cfg.grip_effort is True
    with pytest.raises(ValueError, match="grip_effort"):
        Stretch4Config(mode="control", preprocessing_from=str(root), grip_effort=False)
    assert Stretch4Config(mode="control", preprocessing_from=str(prepared_dataset(
        tmp_path / "plain", {}))).grip_effort is False


def _tagged(tmp_path, tool):
    root = prepared_dataset(tmp_path / "ds", {})
    side = sidecar.read(root)
    if tool:
        side["gripper_tool"] = tool
    sidecar.save(root, side)
    return root


@pytest.mark.parametrize("trained, live", [("parallel_gripper", "stretch_gripper"),
                                           (None, "parallel_gripper")])   # untagged is an SG4
def test_policy_from_another_gripper_is_refused(tmp_path, trained, live):
    robot = Stretch4(Stretch4Config(mode="control", preprocessing_from=str(_tagged(tmp_path, trained))))
    robot._gripper_tool = live
    with pytest.raises(RuntimeError, match="the robot has"):
        robot._check_gripper()


def test_same_gripper_connects(tmp_path):
    robot = Stretch4(Stretch4Config(mode="control",
                                    preprocessing_from=str(_tagged(tmp_path, "parallel_gripper"))))
    robot._gripper_tool = "parallel_gripper"
    robot._check_gripper()
