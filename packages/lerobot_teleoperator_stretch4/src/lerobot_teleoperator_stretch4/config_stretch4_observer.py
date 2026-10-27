"""Config for the passive Stretch 4 teleoperator (--teleop.type=stretch4_observer)."""

from dataclasses import dataclass

from lerobot.teleoperators import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("stretch4_observer")
@dataclass(kw_only=True)
class Stretch4ObserverConfig(TeleoperatorConfig):
    """Records what the robot does instead of driving it.

    Run your usual teleop (gamepad, SpaceMouse, Quest, marionnette...) as its own process;
    lerobot-record with --robot.type=stretch4 --teleop.type=stretch4_observer then saves the
    measured pose as the action, whatever device moved the robot."""

    # Seconds to wait for the robot's first observation in get_action().
    timeout_s: float = 2.0
