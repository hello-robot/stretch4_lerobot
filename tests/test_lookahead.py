"""Record-time lookahead: lerobot-record's (get_observation, then get_action) pairs must be
(observation[t], action = pose[t+N]).

"""

from itertools import count

import pytest

from lerobot_robot_stretch4 import Stretch4, Stretch4Config
from lerobot_robot_stretch4 import stretch4 as stretch4_module
from lerobot_robot_stretch4.lookahead import ObservationDelay
from lerobot_teleoperator_stretch4 import Stretch4Observer, Stretch4ObserverConfig


# ---------------------------------------------------------------- ObservationDelay ---

def test_delay_returns_the_push_from_n_ago():
    d = ObservationDelay(3)
    assert [d.push(i, i / 30) for i in range(3)] == [None, None, None]
    assert [d.push(i, i / 30) for i in range(3, 7)] == [0, 1, 2, 3]


def test_delay_zero_passes_through():
    d = ObservationDelay(0)
    assert [d.push(i, i / 30) for i in range(3)] == [0, 1, 2]


def test_delay_refills_after_a_gap():
    # E.g. save_episode: nothing observed for seconds. An observation from before the gap must
    # not be paired with a pose after it.
    d = ObservationDelay(2)
    for i in range(5):
        d.push(i, i / 30)
    assert d.push(100, 10.0) is None
    assert d.push(101, 10.0 + 1 / 30) is None
    assert d.push(102, 10.0 + 2 / 30) == 100


def test_negative_delay_rejected():
    with pytest.raises(ValueError):
        ObservationDelay(-1)


# ------------------------------------------------------------ robot + observer pairing ---

def stub_robot(monkeypatch, **config):
    robot = Stretch4(Stretch4Config(**config))
    ticks = count()

    def sample():
        i = next(ticks)
        state = {n: float(i) for n in robot.state_names}   # pose i on every column
        return {"image": f"frame {i}", **state}, state, i / 30

    monkeypatch.setattr(robot, "_sample", sample)
    monkeypatch.setattr(Stretch4, "is_connected", property(lambda self: True))
    monkeypatch.setitem(stretch4_module._LATEST, "robot", robot)
    return robot


def record(robot, ticks):
    """lerobot-record's per-tick order: get_observation, then teleop.get_action."""
    observer = Stretch4Observer(Stretch4ObserverConfig())
    frames = []
    for _ in range(ticks):
        obs = robot.get_observation()
        action = observer.get_action()
        frames.append((obs, action))
    return frames


def test_observe_mode_saves_the_pose_three_ticks_ahead(monkeypatch):
    robot = stub_robot(monkeypatch)                        # observe, default lookahead
    assert robot.config.action_lookahead == 3
    for obs, action in record(robot, 20):
        assert obs["image"] == f"frame {int(obs['lift'])}"  # images and state stay paired
        for name in robot.action_names:
            assert action[name] == obs[name] + 3


def test_lookahead_is_configurable(monkeypatch):
    robot = stub_robot(monkeypatch, action_lookahead=5)
    assert all(a["lift"] == o["lift"] + 5 for o, a in record(robot, 10))


def test_control_mode_is_not_delayed(monkeypatch):
    robot = stub_robot(monkeypatch, mode="control")
    assert robot.config.action_lookahead == 0
    obs = robot.get_observation()
    assert obs["lift"] == robot.latest_state["lift"]


def test_control_mode_rejects_a_lookahead():
    with pytest.raises(ValueError, match="action_lookahead"):
        Stretch4Config(mode="control", action_lookahead=3)

