"""Record-time lookahead: hand lerobot-record observations N ticks late.

lerobot-record writes one frame per tick from robot.get_observation() and then
teleop.get_action(), and stretch4_observer's action is the robot's newest pose. Returning the
observation from N ticks ago while the newest pose stays the action makes every frame
(observation[t], action = pose[t+N]): the action leads the state, as a teleop command does on
other LeRobot robots, and the dataset needs no shift at training time.

N counts ticks, i.e. dataset rows, which is also what training offsets count, so the relation
holds in rows even when a tick overruns.
"""

from collections import deque


class ObservationDelay:
    """push(obs, t) -> the observation pushed n calls ago, or None while it fills.

    A gap longer than max_gap_s between pushes (saving an episode, a paused loop) empties it,
    so an observation from before the gap is never paired with a pose from after it."""

    def __init__(self, n: int, max_gap_s: float = 0.5):
        if n < 0:
            raise ValueError(f"lookahead must be >= 0, got {n}")
        self.n = n
        self.max_gap_s = max_gap_s
        self._buf = deque(maxlen=n + 1)
        self._last_t = None

    def push(self, obs, t: float):
        if self._last_t is not None and t - self._last_t > self.max_gap_s:
            self._buf.clear()
        self._last_t = t
        self._buf.append(obs)
        return self._buf[0] if len(self._buf) == self.n + 1 else None

    def clear(self) -> None:
        self._buf.clear()
        self._last_t = None
