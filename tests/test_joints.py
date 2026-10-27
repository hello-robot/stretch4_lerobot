"""The joint thread: a bad status (a missing field, a gripper the tool config does not name)
is reported and skipped, not the end of the thread."""

import threading
import time
from collections import deque
from types import SimpleNamespace

from lerobot_robot_stretch4 import joints, pose

JOINT = {"pos": 0.1, "vel": 0.0, "effort": 0.0, "timestamp_pc": 0.0}


class FakeRobot:
    def __init__(self, bad=1):
        self.bad, self.polls = bad, 0

    def pull_status(self, blocking=True):
        self.polls += 1
        return True

    @property
    def status(self):
        t = time.time()
        joint = {**JOINT, "timestamp_pc": t}
        eoa = {j: joint for j in joints.WRIST_JOINTS}
        if self.polls > self.bad:
            eoa["stretch_gripper"] = joint
        return {"arm": joint, "lift": joint, "end_of_arm": eoa,
                "omnibase": {"pose_time_s": t, "x_vel": 0.0, "y_vel": 0.0, "theta_vel": 0.0}}


def test_a_bad_status_does_not_end_the_thread(monkeypatch, capsys):
    grip = SimpleNamespace(name="stretch_gripper", aperture_mm=lambda status: 40.0)
    monkeypatch.setattr(pose, "gripper_conversion", lambda robot: grip)
    history, stop = joints.JointHistory(), threading.Event()
    t = threading.Thread(target=joints.joint_thread, daemon=True,
                         args=(FakeRobot(), history, 0.0, stop, 200.0, deque(maxlen=8)))
    t.start()
    deadline = time.monotonic() + 2.0
    while "gripper" not in history.sample_at(time.monotonic()) and time.monotonic() < deadline:
        time.sleep(0.01)
    stop.set()
    t.join(timeout=1.0)
    assert history.sample_at(time.monotonic())["gripper"]["aperture_mm"] == 40.0
    assert "status read failed (KeyError" in capsys.readouterr().out
