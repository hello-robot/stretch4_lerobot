"""LeRobot robot plugin for Hello Robot Stretch 4 (--robot.type=stretch4)."""

from .config_stretch4 import Stretch4Config
from .stretch4 import Stretch4, latest_robot

__all__ = ["Stretch4", "Stretch4Config", "latest_robot"]
