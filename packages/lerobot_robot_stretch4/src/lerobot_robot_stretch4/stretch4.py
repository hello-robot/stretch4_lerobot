"""Hello Robot Stretch 4 as a LeRobot Robot.

observation.state = base_vx, base_vy, base_omega, lift, arm, wrist yaw/pitch/roll, grip_mm
(columns.POLICY_STATE_COLUMNS: the action's order, in every Stretch 4 dataset),
then grip_effort with --robot.grip_effort=true,
plus one image per camera, rotated upright, cropped and then resized to the camera's size. The
joints are interpolated to the sync camera's capture time and the other cameras' matching frames
waited for, instead of LeRobot's usual "latest of everything".

action = base_vx, base_vy, base_omega (body-frame velocity) + the 6 joints (absolute)
(columns.VELOCITY_ACTION_COLUMNS). No base pose anywhere, so any LeRobot policy trains on it.

Observe mode (recording): get_observation() returns the observation action_lookahead ticks late
while latest_state, which stretch4_observer returns as the action, is the newest one, so
lerobot-record stores action[t] = value[t + action_lookahead] (see lookahead.py).

Control mode: the arm and gripper get rate-limited move_to targets and the base the commanded
velocity, capped (control.py). With --robot.preprocessing_from, the state guard (state_guard.py)
keeps the state inside the training range: the joints stretch4_fix_stats held are pinned to their
training value and kept where they are, the rest clamped to the dataset's range.
"""

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from functools import cached_property


from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.robots import Robot

from lerobot_camera_oak import OAKCamera

from .config_stretch4 import Stretch4Config

log = logging.getLogger(__name__)

BASE_VELOCITY = ("base_vx", "base_vy", "base_omega")

# The robot most recently connected in this process, for stretch4_observer.
_LATEST = {"robot": None}


def latest_robot():
    return _LATEST["robot"]


class Stretch4(Robot):
    config_class = Stretch4Config
    name = "stretch4"

    def __init__(self, config: Stretch4Config):
        super().__init__(config)
        from .columns import VELOCITY_ACTION_COLUMNS, policy_state_columns
        self.config = config
        self.action_names = list(VELOCITY_ACTION_COLUMNS)
        self.state_names = policy_state_columns(config.grip_effort)
        self.cameras = make_cameras_from_configs(config.cameras)
        self._robot = None           # status client, polled by the joint thread
        self._cmd_robot = None       # command client (control mode)
        self._gripper_tool = None    # stretch_gripper (SG4) | parallel_gripper (PG4), at connect
        self._history = None
        self._stop = threading.Event()
        self._threads = []
        self._last_sync_t = None
        self._skew_drops = 0
        self._guard = None            # control mode with preprocessing_from: state_guard.StateGuard
        self._guard_far = 0           # observations clamped well outside the training range
        self.latest_state = None     # the newest state (observe mode: ahead of what get_observation returns)
        from .lookahead import ObservationDelay
        self._delay = ObservationDelay(config.action_lookahead if config.mode == "observe" else 0)
        self._arm = self._tracker = None
        self._last_action_t = 0.0
        # Cameras are decoded in parallel: a full-res head JPEG takes ~20 ms on the NUC, so two
        # head eyes decoded one after the other do not fit in a 30 Hz frame. cv2 releases the GIL.
        self._decoder = None

    # ---------------------------------------------------------------- features ---

    @cached_property
    def observation_features(self) -> dict:
        cams = {k: (c.height, c.width, 3) for k, c in self.config.cameras.items()}
        return {**{n: float for n in self.state_names}, **cams}

    @cached_property
    def action_features(self) -> dict:
        return {n: float for n in self.action_names}

    @property
    def is_connected(self) -> bool:
        return self._robot is not None and all(c.is_connected for c in self.cameras.values())

    @property
    def is_calibrated(self) -> bool:
        return True   # Stretch homes itself (stretch_robot_home); nothing to calibrate here

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    # ----------------------------------------------------------------- connect ---

    def _client(self, client_id):
        from stretch4_body.robot.robot_client import RobotClient
        robot = RobotClient(client_id=client_id)
        if not robot.startup(verbose=False):
            raise ConnectionError("Could not connect to the robot server. Try: stretch_body_server --launch")
        return robot

    def connect(self, calibrate: bool = True) -> None:
        from .joints import JointHistory, joint_thread, measure_mono_to_wall_offset
        from collections import deque

        self._robot = self._client("lerobot_stretch4_status")
        self._robot.pull_status()
        if not self._robot.is_homed():
            self.disconnect()
            raise RuntimeError("Robot is not homed. Run stretch_robot_home first.")
        from .pose import gripper_conversion
        self._gripper_tool = gripper_conversion(self._robot).name   # before joint_thread owns the socket
        self._check_gripper()

        # Status only (never pushes), so it takes no lease and coexists with any teleop.
        self._history = JointHistory()
        self._stop.clear()
        t = threading.Thread(target=joint_thread, daemon=True, name="stretch4-joints", args=(
            self._robot, self._history, measure_mono_to_wall_offset(), self._stop,
            self.config.joint_poll_hz, deque(maxlen=64)))
        t.start()
        self._threads.append(t)

        for cam in self.cameras.values():
            cam.connect()
        self._decoder = ThreadPoolExecutor(max_workers=max(1, len(self.cameras)),
                                           thread_name_prefix="stretch4-decode")

        if self.config.mode == "control":
            self._connect_guard()
            self._connect_control()
        if self.config.voice == "stretch":
            from . import voice
            voice.install()
        self._write_sidecar()
        _LATEST["robot"] = self
        log.info("%s connected (mode=%s)", self, self.config.mode)

    def describe(self) -> dict:
        """This robot's dataset columns, as meta/stretch4.json records them (sidecar.py)."""
        from . import sidecar
        cameras, others = {}, {}
        for key, cfg in self.config.cameras.items():
            cam = self.cameras[key]
            if isinstance(cam, OAKCamera):
                cameras[key] = {"camera": cfg.camera, "crop": cfg.crop,
                                "size": [cfg.height, cfg.width], "rotate": cam._rot}
            else:   # another LeRobot camera: record its config as LeRobot has it
                others[key] = {"type": cfg.type, "height": cfg.height, "width": cfg.width}
        side = sidecar.make(
            "lerobot-record", cameras, self.state_names, self.action_names,
            self.config.action_lookahead,
            {"clock": self.config.sync_camera, "wait_for_match": True,
             "max_skew_ms": self.config.max_skew_ms,
             "match_timeout_ms": self.config.match_timeout_ms},
            gripper_tool=self._gripper_tool)
        side["cameras"].update(others)
        return side

    def _write_sidecar(self):
        """Record the camera preprocessing into the dataset being recorded, if there is one."""
        from .sidecar import recording_dataset_root, write
        root = recording_dataset_root()
        if root is None:
            return   # not recording (e.g. lerobot-rollout without a dataset, or a script)
        try:
            path = write(root, self.describe())
            log.info("wrote %s", path)
        except OSError as e:
            log.warning("could not write the Stretch 4 sidecar into %s: %s", root, e)
        except ValueError:
            self.disconnect()   # another gripper than the dataset's: record nothing into it
            raise

    def _check_gripper(self):
        """Refuse a robot whose gripper is not the one the preprocessing_from dataset was
        recorded with (its policy would see a gripper it never trained on)."""
        if not self.config.preprocessing_from:
            return
        from pathlib import Path
        from .columns import gripper_mismatch
        from .sidecar import read
        side = read(Path(self.config.preprocessing_from).expanduser()) or {}
        problem = gripper_mismatch(side.get("gripper_tool"), self._gripper_tool)
        if problem:
            self.disconnect()
            raise RuntimeError(problem)

    def _connect_guard(self):
        """The state guard from the training dataset (--robot.preprocessing_from)."""
        if not (self.config.state_guard and self.config.preprocessing_from):
            if self.config.state_guard:
                log.warning("no --robot.preprocessing_from: the state is not kept inside the "
                            "training range (state_guard needs the training dataset)")
            return
        from pathlib import Path
        from .state_guard import StateGuard
        self._guard, problem = StateGuard.from_dataset(
            Path(self.config.preprocessing_from).expanduser(), self.state_names)
        if problem:
            log.warning("%s", problem)
        log.info("held joints (pinned in the state, kept where they are): %s",
                 self._guard.describe())

    def _connect_control(self):
        from .control import ArmCommander, BaseTracker
        from .pose import gripper_conversion

        # A second client for commands: the status client is polled from the joint thread.
        self._cmd_robot = self._client("lerobot_stretch4_cmd")
        bounds = {k: tuple(v) for k, v in self.config.joint_bounds.items()} if self.config.joint_bounds else None
        if bounds and self._guard:
            # A held joint is kept where it already is; its trained range is a single value,
            # which as a clamp would drag it there instead.
            bounds = {k: b for k, b in bounds.items() if k not in self._guard.held}
        if bounds is None:
            log.warning("no joint_bounds: only the per-step rate limit protects the joints. Pass "
                        "--robot.preprocessing_from=<dataset> or --robot.joint_bounds.")
        self._arm = ArmCommander(self._cmd_robot, gripper_conversion(self._cmd_robot),
                                 dict(self.config.limits), gripper=self.config.use_gripper,
                                 bounds=bounds)
        self._tracker = BaseTracker(self._cmd_robot, max_lin=self.config.base_max_lin,
                                    max_ang=self.config.base_max_ang)
        if self.config.use_base:
            t = threading.Thread(target=self._base_watchdog, daemon=True, name="stretch4-base-watchdog")
            t.start()
            self._threads.append(t)

    # ------------------------------------------------------------- observation ---

    def _joints_at(self, t, timeout=1.0):
        from .joints import WRIST_JOINTS
        needed = ("arm", "lift", "gripper", "omnibase") + WRIST_JOINTS
        deadline = time.monotonic() + timeout
        while True:
            j = self._history.sample_at(t)
            if all(k in j for k in needed):
                return j
            if time.monotonic() > deadline:
                raise TimeoutError(f"no joint state for {[k for k in needed if k not in j]}")
            time.sleep(0.005)

    def _state_from(self, j):
        b, g = j["omnibase"], j["gripper"]
        values = {"lift": j["lift"]["pos"], "arm": j["arm"]["pos"], "wrist_yaw": j["wrist_yaw"]["pos"],
                  "wrist_pitch": j["wrist_pitch"]["pos"], "wrist_roll": j["wrist_roll"]["pos"],
                  "grip_mm": g["aperture_mm"], "base_vx": b["x_vel"], "base_vy": b["y_vel"],
                  "base_omega": b["theta_vel"]}
        if self.config.grip_effort:
            values["grip_effort"] = g["effort"]
        return {n: float(values[n]) for n in self.state_names}

    def get_observation(self) -> dict:
        """The synced observation from action_lookahead ticks ago (observe mode) or now.

        latest_state is always the newest pose. While the delay fills (first call, or after a
        pause such as saving an episode) this samples further frames, ~N frames once per
        episode, so every returned observation has a pose N ticks after it. A sync frame whose
        other cameras have no frame close enough in time is skipped for the next one."""
        if not self.is_connected:
            raise ConnectionError(f"{self} is not connected")
        while True:
            sample = self._sample()
            if sample is None:
                continue
            obs, state, t_ref = sample
            self.latest_state = state
            if self._guard is not None:
                obs = self._guarded(obs)
            delayed = self._delay.push(obs, t_ref)
            if delayed is not None:
                return delayed

    def _guarded(self, obs):
        state = {n: obs[n] for n in self.state_names}
        guarded, far = self._guard.apply(state)
        if far:
            self._guard_far += 1
            if self._guard_far in (1, 10) or self._guard_far % 100 == 0:
                log.warning("state outside the training range, clamped: %s; %d observations so far",
                            ", ".join(f"{n} by {d:+.3g}" for n, d in far.items()), self._guard_far)
        return {**obs, **guarded}

    def _sample(self):
        """-> (observation, state, t_ref): the next sync-camera frame, the other cameras'
        frames matched to it (waited for, not just the latest), and the joints
        interpolated to its capture time. None when another camera has no frame within
        max_skew_ms of it: the caller takes the next sync frame."""
        sync_key = self.config.sync_camera
        sync = self.cameras[sync_key]

        # The next sync-camera frame drives the clock.
        deadline = time.monotonic() + 1.0
        while True:
            t_ref, jpeg = sync.latest_raw()
            if jpeg is not None and t_ref != self._last_sync_t:
                break
            if time.monotonic() > deadline:
                raise TimeoutError(f"no new frame from {sync_key}")
            time.sleep(0.002)
        self._last_sync_t = t_ref

        tol = self.config.max_skew_ms / 1e3
        timeout = self.config.match_timeout_ms / 1e3
        frames = {}
        for key, cam in self.cameras.items():
            if key == sync_key or not isinstance(cam, OAKCamera):
                continue
            _, frame, skew = cam.match_raw(t_ref, tol, timeout)
            if frame is None:
                raise TimeoutError(f"no frame from {key}")
            if abs(skew) > tol:
                self._skew_drops += 1
                if self._skew_drops in (1, 10) or self._skew_drops % 100 == 0:
                    log.warning("%s: no frame within %.0f ms of %s (nearest %+.0f ms); skipped "
                                "that %s frame, %d so far", key, self.config.max_skew_ms, sync_key,
                                skew * 1e3, sync_key, self._skew_drops)
                return None
            frames[key] = frame

        jobs = {sync_key: self._decoder.submit(sync.decode, jpeg)}
        for key, cam in self.cameras.items():
            if key == sync_key:
                continue
            jobs[key] = (self._decoder.submit(cam.decode, frames[key]) if key in frames
                         else self._decoder.submit(cam.async_read))
        obs = {key: job.result() for key, job in jobs.items()}

        state = self._state_from(self._joints_at(t_ref))
        obs.update(state)
        return obs, state, t_ref

    # ------------------------------------------------------------------ action ---

    def send_action(self, action: dict) -> dict:
        if self.config.mode != "control":
            return action   # observe mode: an external teleop drives the robot
        from .columns import POSE_JOINTS
        from .pose import pose_from_joints

        targets = {j: float(action[j]) for j in POSE_JOINTS}
        if self._arm.last is None or (self._guard and self._guard.held):
            now = pose_from_joints(self._joints_at(time.monotonic()))
            if self._arm.last is None:
                self._arm.reset(now)
            if self._guard:
                # held joints stay where they are (the policy was trained with them constant)
                targets = self._guard.hold_targets(targets, now)
        sent = self._arm.send(targets)
        out = dict(action)
        out.update(sent)
        if self.config.use_base:
            # Body-frame velocity, capped at base_max_lin/base_max_ang; the watchdog halts the
            # base if actions stop arriving.
            self._tracker.command_velocity([float(action[n]) for n in BASE_VELOCITY])
            self._last_action_t = time.monotonic()
        self._cmd_robot.push_command(ignore_control_lock=self.config.ignore_control_lock,
                                     priority=self.config.priority)
        return out

    def current_pose(self):
        """The arm's pose now (pose.py's joints), from the joint thread."""
        from .pose import pose_from_joints
        return pose_from_joints(self._joints_at(time.monotonic()))

    def is_runstopped(self) -> bool:
        """True while the run-stop is active, from the status the joint thread pulls
        (_cmd_robot never pulls, so its status would be stale)."""
        return bool(self._robot.is_runstopped())

    def goto_pose(self, pose, timeout=40.0, cancel=None):
        """Drive the arm slowly to `pose` (control mode), the base stopped. -> (arrived, last_error).

        The next send_action is rate-limited from where the arm then is, not from the last
        policy target."""
        from .pose import goto_pose, gripper_conversion, pose_from_joints
        if self.config.mode != "control":
            raise RuntimeError("goto_pose needs --robot.mode=control")
        if self.config.use_base:
            self._halt_base()
        ok, err = goto_pose(self._cmd_robot, pose, gripper_conversion(self._cmd_robot),
                            lambda: pose_from_joints(self._history.sample_at(time.monotonic())),
                            priority=self.config.priority, timeout=timeout,
                            gripper=self.config.use_gripper, cancel=cancel)
        self._arm.last = None
        return ok, err

    def _base_watchdog(self):
        halted = True
        while not self._stop.wait(0.05):
            idle = time.monotonic() - self._last_action_t > self.config.base_timeout_s
            if idle and not halted:
                self._halt_base()
                halted = True
            elif not idle:
                halted = False

    def _halt_base(self):
        try:
            self._tracker.halt()
            self._cmd_robot.push_command(ignore_control_lock=self.config.ignore_control_lock,
                                         priority=self.config.priority)
        except Exception as e:
            log.warning("could not halt the base: %s", e)

    # -------------------------------------------------------------- disconnect ---

    def disconnect(self) -> None:
        from lerobot_camera_oak.capture import close_quietly
        if self._cmd_robot is not None and self.config.use_base and self._tracker is not None:
            self._halt_base()
        self._stop.set()
        for t in self._threads:
            t.join(timeout=2.0)
        self._threads = []
        if self._decoder is not None:
            self._decoder.shutdown(wait=True)
            self._decoder = None
        for key, cam in self.cameras.items():
            close_quietly(f"camera {key}", cam.disconnect)
        for client in (self._cmd_robot, self._robot):
            if client is not None:
                close_quietly("robot client", client.stop)
        self._robot = self._cmd_robot = None
        if _LATEST["robot"] is self:
            _LATEST["robot"] = None
