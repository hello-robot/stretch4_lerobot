"""The Stretch 4 dataset contract: column names and order, the sidecar format, and the defaults
every dataset is recorded with. Changing any of these makes existing datasets and checkpoints
incompatible, so they are pinned here."""

from lerobot_robot_stretch4 import Stretch4, Stretch4Config, sidecar
from lerobot_robot_stretch4 import config_stretch4, columns

JOINTS = ["lift", "arm", "wrist_yaw", "wrist_pitch", "wrist_roll", "grip_mm"]
VELOCITY_ACTION = ["base_vx", "base_vy", "base_omega", *JOINTS]


def test_columns():
    assert list(columns.POSE_JOINTS) == JOINTS
    assert columns.VELOCITY_ACTION_COLUMNS == VELOCITY_ACTION
    assert columns.POSE_ACTION_COLUMNS == ["base_x", "base_y", "base_theta", *JOINTS]
    # state[i] and action[i] are the same quantity; grip_effort is state only, last
    assert columns.POLICY_STATE_COLUMNS == VELOCITY_ACTION
    assert columns.policy_state_columns(False) == VELOCITY_ACTION
    assert columns.policy_state_columns(True) == [*VELOCITY_ACTION, "grip_effort"]


def test_robot_names():
    robot = Stretch4(Stretch4Config())
    assert robot.state_names == VELOCITY_ACTION
    assert robot.action_names == VELOCITY_ACTION
    assert not {"base_x", "base_y", "base_theta"} & set(robot.state_names + robot.action_names)


def test_observer_names_match_the_robot():
    from lerobot_teleoperator_stretch4.stretch4_observer import ACTION_NAMES
    assert ACTION_NAMES == Stretch4(Stretch4Config()).action_names


def test_check_state_names():
    assert columns.check_state_names(VELOCITY_ACTION, VELOCITY_ACTION) is None
    assert columns.check_state_names([*VELOCITY_ACTION, "grip_effort"], VELOCITY_ACTION) is None
    assert columns.check_state_names(JOINTS + VELOCITY_ACTION[:3], VELOCITY_ACTION)
    assert columns.check_state_names(VELOCITY_ACTION, JOINTS + VELOCITY_ACTION[:3])


def test_sidecar_format():
    assert (sidecar.SIDECAR, sidecar.SCHEMA) == ("meta/stretch4.json", "stretch4 dataset v3")
    side = sidecar.make("lerobot-record", {"head_left": {"camera": "head_left", "crop": [0, 0, 10, 16],
                                                         "size": [240, 384], "rotate": 1}},
                        VELOCITY_ACTION, VELOCITY_ACTION, 3, {"clock": "head_left"},
                        gripper_tool="parallel_gripper")
    assert side["cameras"]["head_left"] == {"camera": "head_left", "crop": [0, 0, 10, 16],
                                            "size": [240, 384], "rotate": 1,
                                            "height": 240, "width": 384}
    assert (side["schema"], side["robot_type"], side["action_mode"], side["base_action"],
            side["lookahead"], side["gripper_tool"]) == (
        sidecar.SCHEMA, "stretch4", "absolute", "velocity", 3, "parallel_gripper")


def test_recording_defaults():
    assert config_stretch4.ACTION_LOOKAHEAD == 3
    assert config_stretch4.INCLUDE_GRIP_EFFORT is False
    assert config_stretch4.MAX_SKEW_MS == {"observe": 20.0, "control": 20.0}
    assert config_stretch4.MATCH_TIMEOUT_MS == {"observe": 120.0, "control": 60.0}
