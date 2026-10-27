"""Robot-side commanding in control mode: rate-limited joint targets and capped base velocity.

The action's base dims are a body-frame velocity, which the base takes directly: no odometry
pose tracking is needed.
"""

import math

from .columns import POSE_JOINTS

WRIST_JOINTS = ("wrist_yaw", "wrist_pitch", "wrist_roll")

# Per-control-step command deltas: a policy sample that jumps further than a teleoperator ever
# did is a bad sample, not a motion, and rate-limiting it is cheaper than discovering it with
# the hardware.
MAX_STEP = {"lift": 0.05, "arm": 0.05, "wrist_yaw": 0.6, "wrist_pitch": 0.6,
            "wrist_roll": 0.6, "grip_mm": 40.0}

# A base stopped by the deadband starts again only once the command clears the deadband
# by this factor.
DEADBAND_HYSTERESIS = 1.5


class ArmCommander:
    """Rate-limited joint commands. Clamps each step against the last *commanded*
    target so a single bad policy sample cannot produce a full-range slam."""

    def __init__(self, robot, conv, limits, gripper=True, bounds=None):
        self.robot, self.conv, self.limits = robot, conv, limits
        self.gripper = gripper
        self.bounds = bounds          # {joint: (lo, hi)} from the dataset, or None
        self.last = None

    def reset(self, pose):
        self.last = dict(pose)

    def _limit(self, pose):
        out = {}
        for j in POSE_JOINTS:
            v = float(pose[j])
            if self.bounds and j in self.bounds:
                lo, hi = self.bounds[j]
                v = min(max(v, lo), hi)
            if self.last is not None:
                d = MAX_STEP[j]
                v = min(max(v, self.last[j] - d), self.last[j] + d)
            out[j] = v
        return out

    def send(self, pose):
        pose = self._limit(pose)
        r, L = self.robot, self.limits
        r.lift.move_to(pose["lift"], v_m=L["v_m"], a_m=L["a_m"])
        r.arm.move_to(pose["arm"], v_m=L["v_m"], a_m=L["a_m"])
        for j in WRIST_JOINTS:
            r.end_of_arm.move_to(j, pose[j], v_r=L["v_r"], a_r=L["a_r"])
        if self.gripper:
            # Stored and predicted as aperture mm.
            r.end_of_arm.move_to(self.conv.name, self.conv.command(pose["grip_mm"]),
                                 v_r=L["v_r"], a_r=L["a_r"])
        self.last = pose
        return pose


class BaseTracker:
    """Body-frame base velocity, capped at max_lin / max_ang.

    deadband_lin/deadband_ang: a predicted twist whose linear speed and |omega| are both under
    these stops the base, and the zero is queued once rather than every tick. It applies to
    the whole twist, never per axis. Once stopped, the base starts again only above
    DEADBAND_HYSTERESIS x the deadband, so a prediction hovering at the threshold cannot toggle
    it every tick."""

    def __init__(self, robot, max_lin=0.15, max_ang=0.6, deadband_lin=0.0, deadband_ang=0.0):
        self.robot = robot
        self.max_lin, self.max_ang = max_lin, max_ang
        self.deadband_lin, self.deadband_ang = deadband_lin, deadband_ang
        self.zero_sent = False    # the last velocity queued was (0, 0, 0)
        self.moving = False       # deadband state: the base was last commanded to move
        self.last_cmd = (0.0, 0.0, 0.0)    # body-frame velocity last queued, for logs

    def command_velocity(self, velocity):
        """A predicted body-frame (vx, vy, omega), under the caps and the deadband."""
        vx, vy, w = (float(v) for v in velocity)
        clamp = lambda v, m: max(-m, min(m, v))
        cmd = (clamp(vx, self.max_lin), clamp(vy, self.max_lin), clamp(w, self.max_ang))
        band = 1.0 if self.moving else DEADBAND_HYSTERESIS
        if (math.hypot(cmd[0], cmd[1]) <= band * self.deadband_lin
                and abs(cmd[2]) <= band * self.deadband_ang):
            if not self.zero_sent:
                self.halt()
            return (0.0, 0.0, 0.0)
        self.robot.omnibase.set_velocity(*cmd)
        self.zero_sent, self.moving = False, True
        self.last_cmd = cmd
        return cmd

    def halt(self):
        self.robot.omnibase.set_velocity(0.0, 0.0, 0.0)
        self.zero_sent, self.moving = True, False
        self.last_cmd = (0.0, 0.0, 0.0)
