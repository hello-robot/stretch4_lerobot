"""meta/stretch4.json: written by both recording paths, read by --robot.preprocessing_from."""

import json

import pytest

from lerobot_robot_stretch4 import Stretch4Config
from lerobot_robot_stretch4.config_stretch4 import cameras_from_dataset
from lerobot_robot_stretch4 import sidecar as shared
from lerobot_robot_stretch4.sidecar import recording_dataset_root

VELOCITY_ACTION = ["base_vx", "base_vy", "base_omega", "lift", "arm", "wrist_yaw", "wrist_pitch",
                   "wrist_roll", "grip_mm"]
JOINTS_FIRST = VELOCITY_ACTION[3:] + VELOCITY_ACTION[:3]   # the pre-conformance state order


def _info(root, shapes, action=VELOCITY_ACTION, state=VELOCITY_ACTION):
    (root / "meta").mkdir(parents=True, exist_ok=True)
    feats = {f"observation.images.{k}": {"dtype": "video", "shape": list(v)} for k, v in shapes.items()}
    feats["action"] = {"names": action}
    feats["observation.state"] = {"names": state}
    (root / "meta" / "info.json").write_text(json.dumps({"fps": 30, "features": feats}))


def write(root, side):
    """A v3 sidecar with these cameras (the rest as lerobot-record writes it)."""
    cams = {k: {"size": [240, 384], **v} for k, v in side["cameras"].items()}
    shared.save(root, shared.make("lerobot-record", cams, VELOCITY_ACTION, VELOCITY_ACTION, 3,
                                   {"clock": "head_right"}))


def test_finds_the_dataset_being_recorded(tmp_path):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    dataset = LeRobotDataset.create("local/t", fps=30, root=tmp_path / "ds", use_videos=False,
                                    features={"x": {"dtype": "float32", "shape": (1,), "names": None}})

    def robot_connect():   # the Robot is called from the function holding `dataset`
        return recording_dataset_root()

    assert robot_connect() == tmp_path / "ds"
    assert dataset is not None


def test_no_dataset_no_root():
    assert recording_dataset_root() is None


def test_preprocessing_from_sidecar(tmp_path):
    # The recorded shape comes from meta/info.json, the eye/crop/rotation from the sidecar.
    _info(tmp_path, {"head_right": (240, 384, 3), "gripper": (240, 384, 3)})
    write(tmp_path, {"cameras": {
        "head_right": {"camera": "head_right", "crop": [0, 700, 1200, 750], "rotate": -1},
        "gripper": {"camera": "gripper_right", "crop": None, "rotate": 0}}})
    cams = cameras_from_dataset(tmp_path)
    assert {k: (c.camera, c.crop, c.height, c.width) for k, c in cams.items()} == {
        "head_right": ("head_right", [0, 700, 1200, 750], 240, 384),
        "gripper": ("gripper_right", [0, 0, 640, 400], 240, 384)}
    cfg = Stretch4Config(preprocessing_from=str(tmp_path))
    assert set(cfg.cameras) == {"head_right", "gripper"}


def test_sidecar_crop_that_does_not_fit_the_shape_is_an_error(tmp_path):
    _info(tmp_path, {"head_right": (240, 384, 3)})
    write(tmp_path, {"cameras": {"head_right": {"camera": "head_right", "crop": [0, 0, 600, 600]}}})
    with pytest.raises(ValueError, match="stretch"):
        cameras_from_dataset(tmp_path)


def test_pose_base_dataset_is_refused(tmp_path):
    _info(tmp_path, {"head_right": (240, 384, 3)}, action=["base_x", "base_y", "base_theta"])
    write(tmp_path, {"cameras": {"head_right": {"camera": "head_right"}}})
    with pytest.raises(ValueError, match="not the shared Stretch 4 columns"):
        cameras_from_dataset(tmp_path)


def test_joints_first_state_is_refused(tmp_path):
    # the policy would read a scrambled state: refused, not silently reordered
    _info(tmp_path, {"head_right": (240, 384, 3)}, state=JOINTS_FIRST)
    write(tmp_path, {"cameras": {"head_right": {"camera": "head_right"}}})
    with pytest.raises(ValueError, match="Re-record it"):
        cameras_from_dataset(tmp_path)


def test_old_sidecar_is_refused(tmp_path):
    _info(tmp_path, {"head_right": (240, 384, 3)})
    shared.save(tmp_path, {"schema": "stretch4 dataset v1",
                            "cameras": {"head_right": {"camera": "head_right"}}})
    with pytest.raises(ValueError, match="pre-conformance"):
        cameras_from_dataset(tmp_path)


def test_plugin_write_keeps_the_prepared_record(tmp_path):
    # resuming a recording rewrites the sidecar; what stretch4_fix_stats recorded stays
    from lerobot_robot_stretch4.sidecar import read, write as plugin_write
    _info(tmp_path, {"head_right": (240, 384, 3)})
    write(tmp_path, {"cameras": {"head_right": {"camera": "head_right"}}})
    side = read(tmp_path)
    shared.save(tmp_path, {**side, "prepared": {"held_joints": {}}})
    plugin_write(tmp_path, side)
    assert read(tmp_path)["prepared"] == {"held_joints": {}}


def _plugin_side(gripper_tool):
    cams = {"head_right": {"camera": "head_right", "size": [240, 384]}}
    return shared.make("lerobot-record", cams, VELOCITY_ACTION, VELOCITY_ACTION, 3,
                       {"clock": "head_right"}, gripper_tool=gripper_tool)


def test_gripper_tool_recorded(tmp_path):
    from lerobot_robot_stretch4.sidecar import read, write as plugin_write
    plugin_write(tmp_path, _plugin_side("parallel_gripper"))
    assert read(tmp_path)["gripper_tool"] == "parallel_gripper"


def test_resuming_with_another_gripper_is_refused(tmp_path):
    from lerobot_robot_stretch4.sidecar import read, write as plugin_write
    plugin_write(tmp_path, _plugin_side("stretch_gripper"))
    with pytest.raises(ValueError, match="recorded with stretch_gripper"):
        plugin_write(tmp_path, _plugin_side("parallel_gripper"))
    assert read(tmp_path)["gripper_tool"] == "stretch_gripper"     # left as it was


def test_untagged_dataset_is_sg4(tmp_path):
    # a dataset recorded before the PG4: resumable on an SG4, refused on a PG4
    from lerobot_robot_stretch4.sidecar import write as plugin_write
    plugin_write(tmp_path, _plugin_side(None))
    plugin_write(tmp_path, _plugin_side("stretch_gripper"))
    plugin_write(tmp_path, _plugin_side(None))
    with pytest.raises(ValueError, match="recorded with stretch_gripper"):
        plugin_write(tmp_path, _plugin_side("parallel_gripper"))
