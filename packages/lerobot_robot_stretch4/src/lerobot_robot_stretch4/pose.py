"""Save the robot's current pose to a file, and move the robot back to it later.

    stretch4_pose save start.json     # drive the robot to the pose first
    stretch4_pose goto start.json     # e.g. in the reset between lerobot-record episodes
    stretch4_pose show start.json

A bare file name like start.json is in the repo's start_poses/, wherever you run this from; a
path with a folder in it (./start.json, ~/poses/start.json) is used as given. A pose is lift, arm, wrist yaw/pitch/roll and the gripper aperture; the base is not included,
so drive it back yourself. goto moves slowly and asks first (--yes to skip). It runs at
priority 2, above gamepad teleop (1): let go of the sticks until it prints "arrived".

"""

import argparse
import functools
import json
import sys
import time
from pathlib import Path

from .columns import POSE_JOINTS

WRIST_JOINTS = ("wrist_yaw", "wrist_pitch", "wrist_roll")

ARRIVAL_TOL = {"lift": 0.006, "arm": 0.006, "wrist_yaw": 0.03, "wrist_pitch": 0.03,
               "wrist_roll": 0.03, "grip_mm": 3.0}
SLOW = {"v_m": 0.04, "a_m": 0.10, "v_r": 0.6, "a_r": 2.0}

# Gamepad teleop pushes at 1, the homing routines at 3.
DEFAULT_PRIORITY = 2


# SG4, PG4. grip_mm is fingertip aperture on both; only the command units differ.
GRIPPER_JOINTS = ("stretch_gripper", "parallel_gripper")


_warned_mismatch = False


def gripper_joint(robot):
    global _warned_mismatch
    joints = getattr(robot.end_of_arm, "joints", [])
    name = next((j for j in GRIPPER_JOINTS if j in joints), None)
    if name is None:
        raise SystemExit(f"Tool {getattr(robot, 'eoa_name', '?')} has no "
                         f"{' or '.join(GRIPPER_JOINTS)}.")
    live = (robot.status or {}).get("end_of_arm") or {}
    server = next((j for j in GRIPPER_JOINTS if j in live), None)
    if server is not None and server != name and not _warned_mismatch:
        _warned_mismatch = True
        print(f"WARNING: gripper mismatch. This machine's tool config has {name}, the robot "
              f"server reports {server}. Set robot.tool in stretch_user_params.yaml to match "
              f"the installed tool.")
    return name


class Gripper:
    """The installed gripper, in fingertip aperture mm both ways."""

    def __init__(self, name):
        from stretch4_body.utils.tool_metadata import get_tool_metadata
        self.name = name
        self.meta = get_tool_metadata(name)

    def aperture_mm(self, status):
        """From status["end_of_arm"][name]. The server only fills gripper_conversion for
        the SG4, so derive it here; for the PG4 this is just its pos_mm."""
        return self.meta.status_to_metadata(status)["aperture_m"] * 1000.0

    def command(self, aperture_mm):
        """move_to() units: pct for the SG4, metres for the PG4."""
        return self.meta.aperture_to_command(aperture_mm / 1000.0)


@functools.cache
def _gripper(name):
    return Gripper(name)


def gripper_conversion(robot):
    return _gripper(gripper_joint(robot))


def pose_from_status(robot, conv=None):
    """Only safe from the thread that owns the status socket."""
    conv = conv or gripper_conversion(robot)
    s = robot.status
    return {
        "lift": float(s["lift"]["pos"]),
        "arm": float(s["arm"]["pos"]),
        "wrist_yaw": float(s["end_of_arm"]["wrist_yaw"]["pos"]),
        "wrist_pitch": float(s["end_of_arm"]["wrist_pitch"]["pos"]),
        "wrist_roll": float(s["end_of_arm"]["wrist_roll"]["pos"]),
        "grip_mm": float(conv.aperture_mm(s["end_of_arm"][conv.name])),
    }


def pose_from_joints(joints):
    """From a JointHistory.sample_at() result."""
    needed = ("lift", "arm", "gripper") + WRIST_JOINTS
    if not all(k in joints for k in needed):
        return None
    pose = {"lift": float(joints["lift"]["pos"]), "arm": float(joints["arm"]["pos"])}
    for j in WRIST_JOINTS:
        pose[j] = float(joints[j]["pos"])
    pose["grip_mm"] = float(joints["gripper"]["aperture_mm"])
    return pose


def command_pose(robot, pose, conv, v_m=None, a_m=None, v_r=None, a_r=None, gripper=True):
    robot.lift.move_to(float(pose["lift"]), v_m=v_m, a_m=a_m)
    robot.arm.move_to(float(pose["arm"]), v_m=v_m, a_m=a_m)
    for joint in WRIST_JOINTS:
        robot.end_of_arm.move_to(joint, float(pose[joint]), v_r=v_r, a_r=a_r)
    if gripper:
        robot.end_of_arm.move_to(conv.name, conv.command(float(pose["grip_mm"])),
                                 v_r=v_r, a_r=a_r)


def pose_error(pose, current, gripper=True):
    joints = POSE_JOINTS if gripper else tuple(j for j in POSE_JOINTS if j != "grip_mm")
    return {j: abs(current[j] - pose[j]) for j in joints if j in current}


def arrived(pose, current, gripper=True, tol=None):
    tol = tol or ARRIVAL_TOL
    err = pose_error(pose, current, gripper=gripper)
    if len(err) < (len(POSE_JOINTS) if gripper else len(POSE_JOINTS) - 1):
        return False
    return all(v <= tol[j] for j, v in err.items())


def goto_pose(robot, pose, conv, read_current, *, priority=DEFAULT_PRIORITY, timeout=40.0,
              rate_hz=10.0, gripper=True, cancel=None, on_progress=None, limits=None):
    """Drive to `pose`, re-pushing to hold the lease. Returns (arrived, last_error).

    The lease expires 1.1 s after the last push and idle gamepad teleop keeps pushing
    zeroed velocities, so a single move_to gets cancelled out from under us.

    `read_current` is injected so a caller that does not own the status socket (another
    thread polls it) can still supply the current pose.
    """
    limits = SLOW if limits is None else limits
    deadline = time.monotonic() + timeout
    err = {}
    settled = 0

    while time.monotonic() < deadline:
        if cancel is not None and cancel.is_set():
            return False, err

        command_pose(robot, pose, conv, gripper=gripper, **limits)
        # ignore_control_lock: push_command's own lock check calls sys.exit(1) when it
        # loses, which would take the caller down. The server lease arbitrates.
        robot.push_command(ignore_control_lock=True, priority=priority)

        time.sleep(1.0 / rate_hz)

        current = read_current()
        if current is None:
            continue
        err = pose_error(pose, current, gripper=gripper)
        if on_progress is not None:
            on_progress(err)
        if arrived(pose, current, gripper=gripper):
            settled += 1
            if settled >= 2:
                return True, err
        else:
            settled = 0

    return False, err


def start_poses_dir() -> Path:
    """The repo's start_poses/ (this package is installed editable from the repo)."""
    here = Path(__file__).resolve().parents[4] / "start_poses"
    return here if here.is_dir() else Path("~/repos/stretch4_lerobot/start_poses").expanduser()


def pose_path(name: str) -> Path:
    """A bare file name (no folder in it) is in the repo's start_poses/, wherever this runs
    from; a path with a folder (./x.json, ~/x.json, /abs/x.json) is used as given."""
    return start_poses_dir() / name if "/" not in name else Path(name).expanduser()


def save_pose(path: Path, pose, source=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"joints": {j: round(float(pose[j]), 6) for j in POSE_JOINTS},
               "columns": list(POSE_JOINTS),
               "units": "lift/arm m, wrist rad, grip_mm mm aperture",
               "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    if source:
        payload["source"] = source
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def load_pose(path: Path):
    """Accepts a start_pose.json or an episode meta.json."""
    data = json.loads(Path(path).read_text())
    joints = data.get("joints") or (data.get("start_pose") or {}).get("joints")
    if not joints:
        raise ValueError(f"{path} has no start pose")
    missing = [j for j in POSE_JOINTS if j not in joints]
    if missing:
        raise ValueError(f"{path} is missing {', '.join(missing)}")
    return {j: float(joints[j]) for j in POSE_JOINTS}


def format_pose(pose):
    return "  ".join(f"{j}={pose[j]:.3f}" for j in POSE_JOINTS)


def connect(client_id):
    from stretch4_body.robot.robot_client import RobotClient
    robot = RobotClient(client_id=client_id)
    if not robot.startup(verbose=False):
        raise SystemExit("Could not connect to the robot server. Try: stretch_body_server --launch")
    robot.pull_status()
    if not robot.is_homed():
        robot.stop()
        raise SystemExit("Robot is not homed. Run stretch_robot_home first.")
    return robot


def cmd_save(robot, args):
    if args.file.exists() and not args.force:
        raise SystemExit(f"{args.file} exists; pass --force to overwrite it")
    pose = pose_from_status(robot)
    save_pose(args.file, pose, source="stretch4_pose save")
    print(f"saved {args.file}\n  {format_pose(pose)}")
    return 0


def cmd_goto(robot, args, pose):
    print(f"  now:    {format_pose(pose_from_status(robot))}")
    if not args.yes:
        try:
            if input("Move there? [y/N] ").strip().lower() not in ("y", "yes"):
                print("not moved")
                return 1
        except (EOFError, KeyboardInterrupt):
            print("\nnot moved")
            return 1

    def read_current():
        robot.pull_status(blocking=False)
        return pose_from_status(robot)

    ok, err = goto_pose(robot, pose, gripper_conversion(robot), read_current,
                        priority=args.priority, timeout=args.timeout, gripper=not args.no_gripper)
    worst = max(err.items(), key=lambda kv: kv[1]) if err else ("-", 0.0)
    if ok:
        print(f"arrived (worst residual {worst[0]}={worst[1]:.4f})")
        return 0
    print(f"did NOT arrive within {args.timeout:.0f}s (worst {worst[0]}={worst[1]:.4f})")
    return 1


def main(argv=None):
    summary, _, epilog = __doc__.partition("\n\n")
    p = argparse.ArgumentParser(prog="stretch4_pose", description=summary, epilog=epilog,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True, metavar="{save,goto,show}")

    s = sub.add_parser("save", help="save the robot's current pose")
    s.add_argument("file", type=pose_path, help="pose file to write (a bare name: in start_poses/)")
    s.add_argument("--force", action="store_true", help="overwrite an existing file")

    g = sub.add_parser("goto", help="move the robot to a saved pose")
    g.add_argument("file", type=pose_path, help="pose file to read (a bare name: in start_poses/)")
    g.add_argument("--yes", action="store_true", help="do not ask before moving")
    g.add_argument("--no-gripper", action="store_true", help="leave the gripper where it is")
    g.add_argument("--timeout", type=float, default=40.0, metavar="S",
                   help="give up after this many seconds (default 40)")
    g.add_argument("--priority", type=int, default=DEFAULT_PRIORITY, metavar="N",
                   help=f"command priority (default {DEFAULT_PRIORITY}; gamepad teleop is 1)")

    sh = sub.add_parser("show", help="print a saved pose (no robot needed)")
    sh.add_argument("file", type=pose_path, help="pose file to read (a bare name: in start_poses/)")
    args = p.parse_args(argv)

    pose = None
    if args.command in ("goto", "show"):
        if not args.file.exists():
            raise SystemExit(f"Pose file {args.file.name} not found in {args.file.parent}/\n"
                             f"Make one with: stretch4_pose save {args.file}")
        pose = load_pose(args.file)
        print(f"pose {args.file}\n  target: {format_pose(pose)}")
        if args.command == "show":
            return 0

    robot = connect(f"stretch4_pose_{args.command}")
    try:
        return cmd_save(robot, args) if args.command == "save" else cmd_goto(robot, args, pose)
    finally:
        try:
            robot.stop()
        except Exception:
            pass


if __name__ == "__main__":
    sys.exit(main())
