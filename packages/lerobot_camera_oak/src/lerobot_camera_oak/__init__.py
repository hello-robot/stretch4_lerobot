"""LeRobot camera plugin for the Hello Robot Stretch 4 OAK cameras (--robot.cameras type: oak)."""

from .camera_oak import OAKCamera
from .configuration_oak import CAMERAS, OAKCameraConfig

__all__ = ["OAKCamera", "OAKCameraConfig", "CAMERAS"]
