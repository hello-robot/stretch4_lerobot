"""Frame buffering and device lifecycle for the OAK cameras: timestamped ring buffers the robot
pairs frames from by capture time, and retrying open / quiet close of a depthai device.

Clocks: image `timestamp` is depthai getTimestamp() == time.monotonic(), stamped on-device
at capture.
"""

import threading
import time
from collections import deque


class FrameBuffer:
    """Ring buffer of recent frames, searchable by capture timestamp."""

    def __init__(self, maxlen=32):
        self._lock = threading.Lock()
        self._frames = deque(maxlen=maxlen)
        self._seq = 0

    def put(self, t, frame):
        with self._lock:
            self._frames.append((t, frame))
            self._seq += 1

    def latest(self):
        with self._lock:
            if not self._frames:
                return None, None, self._seq
            t, f = self._frames[-1]
            return t, f, self._seq

    def newest_timestamp(self):
        with self._lock:
            return self._frames[-1][0] if self._frames else None

    def nearest(self, t_query):
        with self._lock:
            if not self._frames:
                return None, None, None
            best = min(self._frames, key=lambda tf: abs(tf[0] - t_query))
        return best[0], best[1], best[0] - t_query


def match_nearest(buf, t_ref, tol, timeout, stop_event=None):
    """The frame in buf nearest t_ref, after waiting up to timeout s for one stamped at or
    after t_ref - tol to arrive: (t, frame, skew) as FrameBuffer.nearest, or Nones. """
    wait_start = time.monotonic()
    while stop_event is None or not stop_event.is_set():
        newest = buf.newest_timestamp()
        if newest is not None and newest >= t_ref - tol:
            break
        if time.monotonic() - wait_start > timeout:
            break
        time.sleep(0.001)
    return buf.nearest(t_ref)


def open_with_retry(label, factory, attempts=6, delay=3.0):
    """A depthai device re-enumerates for a few seconds after a close, so retry."""
    for i in range(attempts):
        try:
            return factory()
        except Exception as e:
            if i == attempts - 1:
                raise
            print(f"{label}: not ready yet ({e}); retrying {i + 1}/{attempts - 1}")
            time.sleep(delay)


def close_quietly(label, fn):
    """Run a teardown step; report failure but never propagate it."""
    try:
        fn()
    except Exception as e:
        print(f"shutdown: {label} failed: {e}")
