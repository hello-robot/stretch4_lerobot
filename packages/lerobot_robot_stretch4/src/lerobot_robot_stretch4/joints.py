"""Joint capture: the robot's status polled into a per-joint history, interpolated to a camera
frame's capture time, so every observation's state is the robot at the moment its images were
taken.

Clocks: joint `timestamp_pc` is time.time() on the robot, converted to time.monotonic() (the
cameras' clock) with one measured offset.
"""

import statistics
import threading
import time
from bisect import bisect_left
from collections import deque

# Steppers report 'force', Feetech servos report 'effort'; both stored as effort.
WRIST_JOINTS = ("wrist_yaw", "wrist_pitch", "wrist_roll")
BASE_FIELDS = ("x", "y", "theta", "x_vel", "y_vel", "theta_vel")


def measure_mono_to_wall_offset(n=51):
    """wall = mono + offset. Median of a sandwich read, to reject scheduling noise."""
    offsets = []
    for _ in range(n):
        m0 = time.monotonic()
        w = time.time()
        m1 = time.monotonic()
        offsets.append(w - (m0 + m1) / 2.0)
    return statistics.median(offsets)


class JointHistory:
    """Per-joint ring buffer of (t_mono, values), linearly interpolated to a query time."""

    def __init__(self, maxlen=600):
        self._lock = threading.Lock()
        self._t = {}
        self._v = {}
        self._maxlen = maxlen

    def append(self, joint, t_mono, values):
        if not t_mono or t_mono <= 0:
            return  # omnibase never gets a timestamp_pc; see joint_thread
        with self._lock:
            if joint not in self._t:
                self._t[joint] = deque(maxlen=self._maxlen)
                self._v[joint] = deque(maxlen=self._maxlen)
            if self._t[joint] and t_mono <= self._t[joint][-1]:
                return
            self._t[joint].append(t_mono)
            self._v[joint].append(values)

    def sample_at(self, t_query):
        out = {}
        with self._lock:
            for joint, ts in self._t.items():
                if not ts:
                    continue
                times, vals = list(ts), list(self._v[joint])
                i = bisect_left(times, t_query)
                if i == 0:
                    sample, residual, extrap = dict(vals[0]), times[0] - t_query, True
                elif i >= len(times):
                    sample, residual, extrap = dict(vals[-1]), times[-1] - t_query, True
                else:
                    t0, t1 = times[i - 1], times[i]
                    v0, v1 = vals[i - 1], vals[i]
                    a = (t_query - t0) / (t1 - t0) if t1 > t0 else 0.0
                    sample = {k: v0[k] + a * (v1[k] - v0[k]) for k in v0 if k in v1}
                    residual, extrap = 0.0, False
                sample["_residual_ms"] = residual * 1e3
                sample["_extrapolated"] = extrap
                out[joint] = sample
        return out


def joint_thread(robot, history, offset, stop_event, rate_hz, ages):
    """Pulls status flat out. The socket is ZMQ CONFLATE=1, so polling faster lowers sample
    age without building a backlog. Never pushes, so it takes no lease and coexists with teleop."""
    from .pose import gripper_conversion
    grip = gripper_conversion(robot)
    period = 1.0 / rate_hz
    failures = 0
    while not stop_event.is_set():
        t0 = time.monotonic()
        try:
            if robot.pull_status(blocking=False):
                _record_status(robot, history, offset, grip, ages)
        except Exception as e:
            # A status missing a field (or a gripper the tool config does not name) must not
            # end the thread: the samples would just stop, with nothing else saying why.
            failures += 1
            if failures in (1, 10) or failures % 1000 == 0:
                print(f"joint thread: status read failed ({type(e).__name__}: {e}); "
                      f"{failures} so far")
        time.sleep(max(0.0, period - (time.monotonic() - t0)))


def _record_status(robot, history, offset, grip, ages):
    """One pulled status into the history."""
    for joint in ("arm", "lift"):
        s = robot.status[joint]
        history.append(joint, s["timestamp_pc"] - offset,
                       {"pos": float(s["pos"]), "vel": float(s["vel"]),
                        "effort": float(s.get("force", 0.0))})
        if joint == "arm":
            ages.append((time.time() - s["timestamp_pc"]) * 1e3)

    for joint in WRIST_JOINTS:
        s = robot.status["end_of_arm"][joint]
        history.append(joint, s["timestamp_pc"] - offset,
                       {"pos": float(s["pos"]), "vel": float(s["vel"]),
                        "effort": float(s["effort"])})

    # Under "gripper" whichever tool it is; pos/vel are servo rad, rad/s.
    g = robot.status["end_of_arm"][grip.name]
    history.append("gripper", g["timestamp_pc"] - offset,
                   {"pos": float(g["pos"]),
                    "aperture_mm": float(grip.aperture_mm(g)),
                    "vel": float(g["vel"]),
                    "effort": float(g["effort"]),
                    "current_mA": float(g.get("current_mA", 0.0))})

    # The omnibase never gets a timestamp_pc (the server leaves it at 0); the stamp
    # that tracks its odometry integration is pose_time_s.
    s = robot.status["omnibase"]
    history.append("omnibase", (s.get("pose_time_s") or 0.0) - offset,
                   {f: float(s[f]) for f in BASE_FIELDS if f in s})
