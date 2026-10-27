"""Config for the Stretch 4 OAK cameras (head stereo pair, gripper OAK-D-SR)."""

from dataclasses import dataclass
from functools import cache

from lerobot.cameras.configs import CameraConfig, ColorMode

# Used for tests, actual configs are imported from stretch4_bofy
CAMERAS = {
    "head_left": ("head", "left", (1920, 1200), 1),
    "head_right": ("head", "right", (1920, 1200), -1),
    "gripper_left": ("gripper", "left", (640, 400), 0),
    "gripper_right": ("gripper", "right", (640, 400), 0),
}


@cache
def native(camera):
    """(native (width, height), quarter turns) of a camera, from stretch4_body only on the robot, else from CAMERAS."""
    try:
        from stretch4_body.subsystem.cameras.enums.rgb_camera import RGBCameras
    except (ImportError, KeyError):
        return CAMERAS[camera][2], CAMERAS[camera][3]
    cfg = RGBCameras[camera].config
    h, w = cfg.image_size
    return (w, h), int(cfg.rotate_number_of_times)


def upright_wh(native_wh, rotate):
    """(width, height) of the frame after the quarter turns."""
    w, h = native_wh
    return (h, w) if rotate % 2 else (w, h)


def center_crop(upright, size):
    """The largest crop of an upright (width, height) frame with the aspect of size (H, W),
    centred: what a camera records when it has a size and no crop of its own."""
    fw, fh = upright
    target = size[1] / size[0]
    if fw / fh > target:                       # frame wider than the target: trim the sides
        w, h = int(round(fh * target)), fh
    else:                                      # taller: trim top and bottom
        w, h = fw, int(round(fw / target))
    return [(fw - w) // 2, (fh - h) // 2, w, h]


def output_hw(native_wh, rotate, crop, size):
    """(height, width) after rotation -> crop -> resize: the size if given, else the crop's."""
    if size:
        return int(size[0]), int(size[1])
    w, h = upright_wh(native_wh, rotate)
    if crop:
        w, h = crop[2], crop[3]
    return h, w


@CameraConfig.register_subclass("oak")
@dataclass(kw_only=True)
class OAKCameraConfig(CameraConfig):
    """One eye of a Stretch 4 OAK camera.

    Every frame is rotated upright, cropped, and only then resized to `size` (H, W), so the
    crop is always in native pixels and the image is never stretched: the crop must have the
    aspect of `size`. With a size and no crop, it is the largest centred crop of that aspect.
    `width`/`height` (the LeRobot fields) are the output size and are filled in from
    `size`/`crop`; leave them unset.
    """

    camera: str = "head_right"
    # (x, y, w, h) in upright (post-rotation) native pixels, applied before the resize.
    crop: list[int] | None = None
    # Recorded (height, width). None: the (cropped) native size. The robot sets every camera to
    # its image_size unless the camera sets its own.
    size: list[int] | None = None
    # Quarter turns. None: the robot's own configuration for this camera, read at connect().
    rotate: int | None = None
    color_mode: ColorMode = ColorMode.RGB
    fps: int | None = 30
    warmup_s: float = 1.0
    # Kept frames per eye, for timestamp lookups (FrameBuffer).
    buffer: int = 32

    def __post_init__(self) -> None:
        if self.camera not in CAMERAS:
            raise ValueError(f"camera must be one of {sorted(CAMERAS)}, got {self.camera!r}")
        self.color_mode = ColorMode(self.color_mode)
        native_wh, robot_rot = native(self.camera)
        rot = robot_rot if self.rotate is None else self.rotate
        if self.size is not None:
            self.size = [int(v) for v in self.size]
            if len(self.size) != 2 or min(self.size) < 1:
                raise ValueError(f"{self.camera}: size must be [height, width], got {self.size}")
            if self.crop is None:
                self.crop = center_crop(upright_wh(native_wh, rot), self.size)
            else:
                aspect, target = self.crop[2] / self.crop[3], self.size[1] / self.size[0]
                if abs(aspect / target - 1) > 0.02:
                    raise ValueError(
                        f"{self.camera}: crop {self.crop} is {aspect:.3f} wide per unit of height "
                        f"but size {self.size} is {target:.3f}; resizing would stretch the image. "
                        "Pick the crop with stretch4_pick_cameras, which keeps the aspect.")
        h, w = output_hw(native_wh, rot, self.crop, self.size)
        if (self.width, self.height) not in ((None, None), (w, h)):
            raise ValueError(f"{self.camera}: width/height {self.width}x{self.height} do not match "
                             f"size/crop, which give {w}x{h}; set only size/crop")
        self.width, self.height = w, h

    @property
    def device(self) -> str:
        return CAMERAS[self.camera][0]

    @property
    def eye(self) -> str:
        return CAMERAS[self.camera][1]
