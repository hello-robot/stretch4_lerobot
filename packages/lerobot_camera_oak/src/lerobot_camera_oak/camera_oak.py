"""LeRobot Camera for the Stretch 4 OAK cameras, through stretch4_body.

Both head eyes come from one device (head_left_right_compressed) and both gripper eyes from
another (gripper_rgbd_compressed). A device can only be opened once, so each is a shared
session: one controller and one capture thread, fanned out to a FrameBuffer per eye, opened by
the first OAKCamera that needs it and closed by the last. Frames stay the camera's own JPEG
until read() decodes them.

"""

import threading
import time
from typing import Any

import cv2
import numpy as np

from lerobot.cameras.camera import Camera
from lerobot.cameras.configs import ColorMode

from .configuration_oak import CAMERAS, OAKCameraConfig, native


class _Session:
    """One opened stretch4_body camera device, shared by the OAKCameras on it."""

    _lock = threading.Lock()
    _open: dict[str, "_Session"] = {}

    def __init__(self, device: str, buffer: int):
        from .capture import FrameBuffer, open_with_retry
        from stretch4_body.subsystem.cameras.controllers.camera_pipeline_controller import (
            RGBPipelineController)
        from stretch4_body.subsystem.cameras.enums.rgb_camera import RGBCameras

        self.device, self.refs = device, 0
        camera_type = (RGBCameras.head_left_right_compressed if device == "head"
                       else RGBCameras.gripper_rgbd_compressed)
        extra = {} if device == "head" else {"enable_pointcloud": False}
        self.controller = open_with_retry(f"{device} camera", lambda: RGBPipelineController(
            camera_type=camera_type, recording_directory=None, show_image_in=None,
            is_rotate=False, is_rectify=False, is_crop=False, ai_models_to_use=[],
            detect_aruco_marker_size=None, **extra))
        self.buffers = {"left": FrameBuffer(buffer), "right": FrameBuffer(buffer)}
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True, name=f"oak-{device}")
        self.thread.start()

    def _run(self):
        # Drain the device as fast as it produces (depthai queues hold only the newest frame);
        # each eye goes to its own buffer.
        try:
            for frame in self.controller.get_frame_synced(is_run_pipeline=False):
                if self.stop.is_set():
                    break
                if frame is None:
                    continue
                for eye, buf in self.buffers.items():
                    img = getattr(frame, eye, None)
                    if img is not None and img.timestamp > 0:
                        buf.put(img.timestamp, img.image)
        except Exception as e:
            if not self.stop.is_set():
                print(f"{self.device} camera thread stopped: {e}")

    @classmethod
    def acquire(cls, device: str, buffer: int) -> "_Session":
        with cls._lock:
            session = cls._open.get(device) or cls(device, buffer)
            cls._open[device] = session
            session.refs += 1
            return session

    def release(self):
        from .capture import close_quietly
        with self._lock:
            self.refs -= 1
            if self.refs > 0:
                return
            self._open.pop(self.device, None)
        # Leave the capture generator before closing the device; closing under a blocked
        # read is what wedges it for the next open.
        self.stop.set()
        self.thread.join(timeout=2.0)
        close_quietly(f"{self.device} camera", self.controller.stop)
        time.sleep(1.0)   # the device reboots on close


class OAKCamera(Camera):
    """One eye of a Stretch 4 OAK camera as a LeRobot Camera."""

    def __init__(self, config: OAKCameraConfig):
        super().__init__(config)
        self.config = config
        self._session: _Session | None = None
        self._last_t = None
        self._rot = config.rotate   # resolved at connect() when None

    def __str__(self) -> str:
        return f"OAKCamera({self.config.camera})"

    @property
    def is_connected(self) -> bool:
        return self._session is not None

    @staticmethod
    def find_cameras() -> list[dict[str, Any]]:
        return [{"type": "oak", "id": name, "name": name} for name in CAMERAS]

    def connect(self, warmup: bool = True) -> None:
        if self._session is not None:
            return
        self._session = _Session.acquire(self.config.device, self.config.buffer)
        # The same stretch4_body setting the config was sized with (configuration_oak.native).
        robot_rot = native(self.config.camera)[1]
        self._rot = robot_rot if self.config.rotate is None else self.config.rotate
        if self.config.rotate is not None and self.config.rotate != robot_rot:
            print(f"note: {self.config.camera} rotated {self.config.rotate} quarter turns as "
                  f"configured; this robot's own setting is {robot_rot}")
        if warmup:
            deadline = time.monotonic() + max(self.config.warmup_s, 5.0)
            while self._buffer().latest()[1] is None:
                if time.monotonic() > deadline:
                    raise TimeoutError(f"{self}: no frames after connect")
                time.sleep(0.02)

    def _buffer(self):
        if self._session is None:
            raise RuntimeError(f"{self} is not connected")
        return self._session.buffers[self.config.eye]

    def _decode(self, jpeg) -> np.ndarray:
        from .image import preprocess
        bgr = cv2.imdecode(np.frombuffer(jpeg, np.uint8) if not isinstance(jpeg, np.ndarray)
                           else jpeg, cv2.IMREAD_COLOR)
        # Crop in native pixels first, then resize to the recorded size.
        rgb = preprocess(bgr, self._rot, self.config.crop, None)
        size = (self.config.width, self.config.height)
        if rgb.shape[1::-1] != size and self.config.size:
            shrink = rgb.shape[1] > size[0]
            rgb = cv2.resize(rgb, size, interpolation=cv2.INTER_AREA if shrink else cv2.INTER_LINEAR)
        if rgb.shape[:2] != (self.config.height, self.config.width):
            raise RuntimeError(f"{self}: frame is {rgb.shape[:2]}, expected "
                               f"{(self.config.height, self.config.width)}")
        return rgb if self.config.color_mode == ColorMode.RGB else cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    # ---- timestamped access, for the Stretch 4 robot's camera/joint sync ----

    def latest_raw(self):
        """(timestamp, jpeg) of the newest frame; timestamp is time.monotonic() at capture."""
        t, jpeg, _ = self._buffer().latest()
        return t, jpeg

    def nearest_raw(self, t_query):
        """(timestamp, jpeg, skew_s) of the frame closest to t_query, among those already here."""
        return self._buffer().nearest(t_query)

    def match_raw(self, t_query, tol_s, timeout_s):
        """(timestamp, jpeg, skew_s) of the frame closest to t_query, after waiting up to
        timeout_s for one captured at or after t_query - tol_s to arrive. Returns as soon as
        that frame is here; a camera with more pipeline latency is waited for, not paired
        with an older frame."""
        from .capture import match_nearest
        return match_nearest(self._buffer(), t_query, tol_s, timeout_s)

    def decode(self, jpeg) -> np.ndarray:
        return self._decode(jpeg)

    def read_with_timestamp(self):
        t, jpeg = self.latest_raw()
        if jpeg is None:
            raise RuntimeError(f"{self}: no frame yet")
        return self._decode(jpeg), t

    # ---- LeRobot Camera API ----

    def read(self, color_mode: ColorMode | None = None) -> np.ndarray:
        return self.read_with_timestamp()[0]

    def async_read(self, timeout_ms: float = 200) -> np.ndarray:
        """The next frame newer than the last one returned."""
        deadline = time.monotonic() + timeout_ms / 1e3
        while True:
            t, jpeg = self.latest_raw()
            if jpeg is not None and t != self._last_t:
                self._last_t = t
                return self._decode(jpeg)
            if time.monotonic() > deadline:
                raise TimeoutError(f"{self}: no new frame within {timeout_ms} ms")
            time.sleep(0.002)

    def read_latest(self, max_age_ms: int = 500) -> np.ndarray:
        t, jpeg = self.latest_raw()
        if jpeg is None:
            raise RuntimeError(f"{self}: no frame yet")
        age_ms = (time.monotonic() - t) * 1e3
        if age_ms > max_age_ms:
            raise TimeoutError(f"{self}: newest frame is {age_ms:.0f} ms old")
        return self._decode(jpeg)

    def disconnect(self) -> None:
        if self._session is not None:
            self._session.release()
            self._session = None
