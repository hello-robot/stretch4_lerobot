"""The velocity base: what lerobot-record stores, what control mode commands, and stock LeRobot
policies building on the resulting features."""

from itertools import count
from types import SimpleNamespace

from lerobot.configs import FeatureType, PolicyFeature
from lerobot.processor import RelativeActionsProcessorStep
from lerobot_robot_stretch4 import Stretch4, Stretch4Config
from lerobot_robot_stretch4 import stretch4 as stretch4_module
from lerobot_teleoperator_stretch4 import Stretch4Observer, Stretch4ObserverConfig

BASE_VEL = ["base_vx", "base_vy", "base_omega"]
JOINTS = ["lift", "arm", "wrist_yaw", "wrist_pitch", "wrist_roll", "grip_mm"]


def stub_robot(monkeypatch, **config):
    robot = Stretch4(Stretch4Config(**config))
    ticks = count()

    def sample():
        i = next(ticks)
        state = {n: float(i) for n in robot.state_names}
        return {"image": i, **state}, state, i / 30

    monkeypatch.setattr(robot, "_sample", sample)
    monkeypatch.setattr(Stretch4, "is_connected", property(lambda self: True))
    monkeypatch.setitem(stretch4_module._LATEST, "robot", robot)
    return robot


def test_recording_has_no_base_pose(monkeypatch):
    robot = stub_robot(monkeypatch)
    observer = Stretch4Observer(Stretch4ObserverConfig())
    assert list(observer.action_features) == robot.action_names == BASE_VEL + JOINTS
    for _ in range(10):
        obs, action = robot.get_observation(), observer.get_action()
        assert not {"base_x", "base_y", "base_theta"} & (set(obs) | set(action))
        assert all(action[n] == obs[n] + 3 for n in action)   # 3 ticks ahead, base velocity too


class Fake:
    def __init__(self):
        self.calls = []

    def __getattr__(self, name):
        def record(*args, **kwargs):
            self.calls.append((name, args))
            return {}
        return record


def test_control_commands_the_base_velocity():
    robot = Stretch4(Stretch4Config(mode="control"))
    robot._arm = SimpleNamespace(last={"lift": 0.0}, send=lambda pose: pose)
    robot._tracker, robot._cmd_robot = Fake(), Fake()
    robot.send_action({**dict(zip(BASE_VEL, (0.2, -0.05, 0.1))), **{j: 0.5 for j in JOINTS}})
    assert robot._tracker.calls == [("command_velocity", ([0.2, -0.05, 0.1],))]
    assert robot._last_action_t > 0   # keeps the base watchdog fed


def features():
    return ({"observation.state": PolicyFeature(type=FeatureType.STATE, shape=(9,)),
             "observation.images.head_right": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 240, 384)),
             "observation.images.gripper": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 240, 384))},
            {"action": PolicyFeature(type=FeatureType.ACTION, shape=(9,))})


def test_stock_act_builds():
    from lerobot.policies.act.configuration_act import ACTConfig
    cfg = ACTConfig(device="cpu")
    cfg.input_features, cfg.output_features = features()
    cfg.validate_features()


def test_stock_diffusion_builds_with_equal_camera_shapes():
    # LeRobot's diffusion refuses cameras of different shapes; the default cameras are equal.
    from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
    cfg = DiffusionConfig(device="cpu")
    cfg.input_features, cfg.output_features = features()
    cfg.validate_features()


def test_stock_pi05_relative_mask_for_stretch():
    # --policy.relative_exclude_joints for pi0.5: joints relative; gripper and base velocity
    # absolute. LeRobot matches the tokens as substrings of the action names.
    step = RelativeActionsProcessorStep(enabled=True, exclude_joints=["grip", "base_v", "base_omega"],
                                        action_names=BASE_VEL + JOINTS)
    assert step._build_mask(9) == [False, False, False, True, True, True, True, True, False]
