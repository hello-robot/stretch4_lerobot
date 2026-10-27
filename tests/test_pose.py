"""stretch4_pose: save the current pose, show it, go back to it (fake robot, nothing moves)."""

import json

import pytest

from lerobot_robot_stretch4 import pose as cli
from lerobot_robot_stretch4.columns import POSE_JOINTS
from lerobot_robot_stretch4.pose import load_pose

POSE = {"lift": 0.8, "arm": 0.1, "wrist_yaw": 0.2, "wrist_pitch": -0.3, "wrist_roll": 0.0,
        "grip_mm": 40.0}


class FakeRobot:
    def __init__(self, pose=POSE):
        self.pose, self.stopped = dict(pose), False

    def stop(self):
        self.stopped = True

    def pull_status(self, blocking=True):
        pass


@pytest.fixture
def robot(monkeypatch):
    r = FakeRobot()
    monkeypatch.setattr(cli, "connect", lambda client_id: r)
    monkeypatch.setattr(cli, "pose_from_status", lambda robot: dict(robot.pose))
    return r


def test_save_writes_the_current_pose(robot, tmp_path):
    out = tmp_path / "start.json"
    assert cli.main(["save", str(out)]) == 0
    assert load_pose(out) == POSE                      # what stretch4_pose goto reads
    assert json.loads(out.read_text())["columns"] == list(POSE_JOINTS)
    assert robot.stopped


def test_save_does_not_overwrite_without_force(robot, tmp_path):
    out = tmp_path / "start.json"
    cli.main(["save", str(out)])
    robot.pose["lift"] = 0.5
    with pytest.raises(SystemExit, match="--force"):
        cli.main(["save", str(out)])
    assert load_pose(out)["lift"] == 0.8
    cli.main(["save", str(out), "--force"])
    assert load_pose(out)["lift"] == 0.5


def test_show_needs_no_robot(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(cli, "connect", lambda client_id: pytest.fail("show connected"))
    out = tmp_path / "start.json"
    cli.save_pose(out, POSE)
    assert cli.main(["show", str(out)]) == 0
    assert "lift=0.800" in capsys.readouterr().out


def test_goto_moves_to_the_file_pose(robot, monkeypatch, tmp_path):
    out = tmp_path / "start.json"
    cli.save_pose(out, POSE)
    robot.pose["lift"] = 0.3
    calls = []

    def fake_goto(r, pose, conv, read_current, **kw):
        calls.append((pose, kw))
        r.pose = dict(pose)
        return True, {j: 0.0 for j in pose}

    monkeypatch.setattr(cli, "goto_pose", fake_goto)
    monkeypatch.setattr(cli, "gripper_conversion", lambda r: None)
    assert cli.main(["goto", str(out), "--yes", "--no-gripper"]) == 0
    (pose, kw), = calls
    assert pose == POSE and kw["gripper"] is False and kw["priority"] == 2
    assert robot.stopped


def test_goto_asks_first(robot, monkeypatch, tmp_path):
    out = tmp_path / "start.json"
    cli.save_pose(out, POSE)
    monkeypatch.setattr(cli, "goto_pose", lambda *a, **k: pytest.fail("moved without a yes"))
    monkeypatch.setattr("builtins.input", lambda prompt: "n")
    assert cli.main(["goto", str(out)]) == 1


def test_goto_without_a_file_says_how_to_make_one(tmp_path):
    with pytest.raises(SystemExit, match="stretch4_pose save"):
        cli.main(["goto", str(tmp_path / "missing.json")])
