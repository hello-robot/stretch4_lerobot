"""Passive teleoperator: the action is the robot's newest measured pose.

Why: lerobot-record saves whatever the teleoperator returns as the action. A velocity device
(gamepad) and a pose device (leader arm) return different things, so datasets would depend on
the device. The observer returns the robot's measured pose instead, so every device gives the
same dataset.

It reads robot.latest_state from the Stretch4 robot in the same process (lerobot-record calls
get_observation, then get_action). In observe mode that robot hands out each observation
action_lookahead ticks late (default 3), so the saved action[t] is the pose at t+3: it leads
the state, like a teleop command on other LeRobot robots.
"""

import time

from lerobot.teleoperators import Teleoperator

from lerobot_robot_stretch4 import latest_robot

from .config_stretch4_observer import Stretch4ObserverConfig

# Before a robot is connected. Once one is, its action_names (velocity or pose base) are used.
ACTION_NAMES = ["base_vx", "base_vy", "base_omega", "lift", "arm",
                "wrist_yaw", "wrist_pitch", "wrist_roll", "grip_mm"]


def _action_names():
    robot = latest_robot()
    return list(robot.action_names) if robot is not None else ACTION_NAMES


class Stretch4Observer(Teleoperator):
    config_class = Stretch4ObserverConfig
    name = "stretch4_observer"

    def __init__(self, config: Stretch4ObserverConfig):
        super().__init__(config)
        self.config = config
        self._connected = False

    @property
    def action_features(self) -> dict:
        return {n: float for n in _action_names()}

    @property
    def feedback_features(self) -> dict:
        return {}

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def is_calibrated(self) -> bool:
        return True

    def connect(self, calibrate: bool = True) -> None:
        self._connected = True

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    def get_action(self) -> dict:
        deadline = time.monotonic() + self.config.timeout_s
        while True:
            robot = latest_robot()
            if robot is not None and robot.latest_state is not None:
                if robot.config.mode != "observe":
                    raise RuntimeError("stretch4_observer records an externally driven robot; "
                                       "use it with --robot.mode=observe")
                return {n: robot.latest_state[n] for n in robot.action_names}
            if time.monotonic() > deadline:
                raise RuntimeError("stretch4_observer found no connected --robot.type=stretch4 "
                                   "observation in this process; it only works together with "
                                   "that robot (lerobot-record runs both in one process)")
            time.sleep(0.01)

    def send_feedback(self, feedback: dict) -> None:
        pass

    def disconnect(self) -> None:
        self._connected = False
