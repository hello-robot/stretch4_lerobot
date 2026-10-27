"""Run a policy for several episodes in one async-inference session.

    stretch4_rollout <LeRobot robot_client flags> --start_pose=<name>.json \\
        --num_episodes=10 --episode_time_s=60 --reset_time_s=20

The cameras open and the policy server loads the policy once, for the whole session. Before each
episode the arm moves slowly to the start pose (the base is stopped, not moved) once you press
Enter, then waits reset_time_s for you to reset the scene. LeRobot's async robot_client control
loop then runs the policy for up to episode_time_s. After the last episode the arm returns to the
start pose (again after Enter).

Keys, in this terminal (or on the robot's own keyboard and screen); the arrows and Esc work as in
lerobot-record:

    Enter        move the arm to the start pose (drive the base back first, if it moved)
    Right arrow  end the current episode, or the wait before the next one
    Left arrow   discard the last episode and run it again
    Esc          stop the session
"""

import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from queue import Queue

import draccus

from lerobot.async_inference.configs import RobotClientConfig
from lerobot.async_inference.robot_client import RobotClient
from lerobot.transport import services_pb2  # type: ignore
from lerobot.utils.import_utils import register_third_party_plugins

# Timesteps jump by this at each episode start, so the actions of a chunk the server was still
# computing when the previous episode ended are all older than the new episode's and dropped.
EPISODE_GAP = 100_000

# How an episode ended -> how the log says it.
END_REASONS = {"time_limit": "time limit reached", "next": "ended with Right arrow",
               "rerun": "discarded with Left arrow", "stopped": "stopped with Esc"}

RUNSTOPPED = "The robot is run-stopped and will not move. Release the run-stop{}."


@dataclass
class RolloutConfig(RobotClientConfig):
    num_episodes: int = 10
    # Longest an episode runs; Right arrow ends it sooner.
    episode_time_s: float = 60.0
    # Wait at the start pose before each episode, to reset the scene; Right arrow skips it.
    reset_time_s: float = 20.0
    # The pose file each episode starts from (stretch4_pose save). Required. A bare name is in
    # the repo's start_poses/; a path with a folder in it is used as given.
    start_pose: str | None = None
    # Give up moving to the start pose after this long (the episode starts anyway).
    goto_timeout_s: float = 40.0
    # Wait for Enter before each move to the start pose, so the base can be driven back first.
    confirm_goto: bool = True
    # Announce "Episode 3" etc. out loud (the robot's voice with --robot.voice=stretch).
    play_sounds: bool = True

    def __post_init__(self):
        super().__post_init__()
        if self.num_episodes <= 0:
            raise ValueError(f"num_episodes must be positive, got {self.num_episodes}")
        if self.episode_time_s <= 0:
            raise ValueError(f"episode_time_s must be positive, got {self.episode_time_s}")
        if self.reset_time_s < 0:
            raise ValueError(f"reset_time_s must not be negative, got {self.reset_time_s}")
        if self.goto_timeout_s <= 0:
            raise ValueError(f"goto_timeout_s must be positive, got {self.goto_timeout_s}")


class RolloutClient(RobotClient):
    """RobotClient whose control loop runs one episode at a time; between episodes no
    observation is sent and no action is executed or queued."""

    def __init__(self, config: RolloutConfig):
        super().__init__(config)
        self.active = threading.Event()
        self._episode_lock = threading.Lock()

    def _aggregate_action_queues(self, incoming_actions, aggregate_fn=None):
        with self._episode_lock:
            if self.active.is_set():
                super()._aggregate_action_queues(incoming_actions, aggregate_fn)

    def begin_episode(self):
        # Server side: empty observation queue, no timestep predicted yet. The policy stays loaded.
        self.stub.Ready(services_pb2.Empty())
        with self._episode_lock:
            with self.action_queue_lock:
                self.action_queue = Queue()
            with self.latest_action_lock:
                self.latest_action = max(self.latest_action, 0) + EPISODE_GAP
            self.must_go.set()
            self.active.set()

    def end_episode(self):
        with self._episode_lock:
            self.active.clear()
            with self.action_queue_lock:
                self.action_queue = Queue()

    def run_episode(self, task: str, duration_s: float, events: dict) -> float:
        """robot_client's control loop, until duration_s, a key or shutdown. -> seconds run."""
        self.begin_episode()
        t0 = time.perf_counter()
        try:
            while (self.running and not events["exit_early"]
                   and time.perf_counter() - t0 < duration_s):
                tick = time.perf_counter()
                if self.actions_available():
                    self.control_loop_action()
                if self._ready_to_send_observation():
                    self.control_loop_observation(task)
                time.sleep(max(0.0, self.config.environment_dt - (time.perf_counter() - tick)))
        finally:
            self.end_episode()
        return time.perf_counter() - t0


def red(text: str) -> str:
    """text in red when the terminal shows colors (LeRobot's log format has none)."""
    return f"\033[91m{text}\033[0m" if sys.stderr.isatty() else text


def resolve_start_pose(cfg: RolloutConfig):
    """-> (pose, its file). A bare name is in the repo's start_poses/ (as for stretch4_pose).
    Exits when --start_pose is missing or its file does not exist."""
    from .pose import load_pose, pose_path
    if not cfg.start_pose:
        raise SystemExit(red("--start_pose is required: the pose file each episode starts from, "
                             "e.g. --start_pose=<name>.json for start_poses/<name>.json. Save "
                             "one with `stretch4_pose save <name>.json`."))
    path = pose_path(cfg.start_pose)
    if not path.exists():
        raise SystemExit(red(f"Start pose file {path.name} not found in {path.parent}/\n"
                             f"Save one with `stretch4_pose save {path}`."))
    return load_pose(path), str(path)


def key_listener(log):
    """lerobot-record's keys and events (exit_early, rerecord_episode, stop_recording), plus
    Enter (events["goto_start"]). -> (listener or None, events)."""
    from lerobot.utils.keyboard_input import create_key_listener
    events = {"exit_early": False, "rerecord_episode": False, "stop_recording": False,
              "goto_start": False}

    def on_key(key):
        if key == "enter":
            events["goto_start"] = True
        elif key == "right":
            events["exit_early"] = True
        elif key == "left":
            log.info("Left arrow: discarding the last episode")
            events.update(rerecord_episode=True, exit_early=True)
        elif key == "esc":
            log.info("Esc: stopping the session")
            events.update(stop_recording=True, exit_early=True)

    listener = create_key_listener(
        on_key, controls_help="Enter: move to start pose, Right: next, Left: discard and rerun, "
                              "Esc: stop")
    return listener, events


def wait(seconds: float, events: dict, running) -> None:
    t0 = time.perf_counter()
    while running() and not events["exit_early"] and time.perf_counter() - t0 < seconds:
        time.sleep(0.05)


def run_episodes(client, cfg: RolloutConfig, events: dict, start_pose, say, log) -> list[dict]:
    """The episode loop. -> one record per episode run (counted or discarded)."""
    robot = client.robot
    running = lambda: client.running
    results, done = [], 0

    def cancelled(final):
        return not running() or (events["exit_early"] if final else events["stop_recording"])

    def to_start(final=False):
        """-> False when the arm was not moved: Esc, or Right arrow at the end."""
        if final:
            events["exit_early"] = False
        while True:
            if cfg.confirm_goto:
                events["goto_start"] = False
                log.info("Press Enter to move the arm to the start pose (%s).",
                         "Right arrow to leave it where it is" if final else "Esc to stop")
                while not events["goto_start"]:
                    if cancelled(final):
                        return False
                    time.sleep(0.05)
                events["goto_start"] = False
            if not robot.is_runstopped():
                break
            log.error(red(RUNSTOPPED.format(", then press Enter again" if cfg.confirm_goto
                                            else " to continue")))
            while not cfg.confirm_goto and robot.is_runstopped():
                if cancelled(final):
                    return False
                time.sleep(0.05)
        log.info("Moving the arm to the start pose")
        ok, err = robot.goto_pose(start_pose, timeout=cfg.goto_timeout_s)
        if not ok:
            worst = max(err.items(), key=lambda kv: kv[1]) if err else ("-", 0.0)
            log.warning(red("Start pose not reached within %.0f s (largest error: %s=%.4f)%s"
                            % (cfg.goto_timeout_s, *worst,
                               "; the robot is run-stopped" if robot.is_runstopped() else "")))
        return True

    def take(key):
        hit = events[key]
        events[key] = False
        return hit

    while done < cfg.num_episodes and not events["stop_recording"] and running():
        say("Reset the scene")
        if not to_start():
            return results   # Esc at the prompt: stop without moving
        events["exit_early"] = False
        log.info("Episode %d/%d starts in %.0f s (Right arrow to start now)",
                 done + 1, cfg.num_episodes, cfg.reset_time_s)
        wait(cfg.reset_time_s, events, running)
        if robot.is_runstopped():
            log.error(red(RUNSTOPPED.format(" to start the episode")))
            while robot.is_runstopped() and running() and not events["stop_recording"]:
                time.sleep(0.05)
        take("exit_early")
        if events["stop_recording"]:
            break
        if take("rerecord_episode"):
            # pressed while waiting: the episode before is discarded
            counted = [r for r in results if r["counted"]]
            if counted:
                done -= 1
                counted[-1]["counted"] = False
                log.info("Episode %d discarded; it will run again", done + 1)
            continue

        say(f"Episode {done + 1}")
        log.info("Episode %d/%d running (up to %.0f s)", done + 1, cfg.num_episodes,
                 cfg.episode_time_s)
        seconds = client.run_episode(cfg.task, cfg.episode_time_s, events)
        take("exit_early")
        rerun = take("rerecord_episode")
        ended = ("stopped" if events["stop_recording"] else "rerun" if rerun
                 else "time_limit" if seconds >= cfg.episode_time_s else "next")
        results.append({"episode": done + 1, "seconds": round(seconds, 1), "ended": ended,
                        "counted": not rerun})
        log.info("Episode %d/%d finished after %.1f s: %s%s", done + 1, cfg.num_episodes,
                 seconds, END_REASONS[ended], "; it will run again" if rerun else "")
        if not rerun:
            done += 1

    if running():
        say("Done")
        to_start(final=True)
    return results


def summary(results: list[dict]) -> str:
    counted = [r for r in results if r["counted"]]
    lines = [f"  Episode {r['episode']}: {r['seconds']:.1f} s, {END_REASONS[r['ended']]}"
             for r in counted]
    discarded = len(results) - len(counted)
    return (f"Rollout complete: {len(counted)} episode(s) counted, {discarded} discarded"
            + ("\n" + "\n".join(lines) if lines else ""))


@draccus.wrap()
def _run(cfg: RolloutConfig):
    from lerobot.utils.utils import log_say
    from .stretch4 import Stretch4

    start_pose, source = resolve_start_pose(cfg)   # before the cameras come up, to fail fast
    client = RolloutClient(cfg)   # connects the robot: the cameras come up here, once
    log = client.logger
    listener = None
    try:
        if not isinstance(client.robot, Stretch4) or client.robot.config.mode != "control":
            raise SystemExit("stretch4_rollout requires --robot.type=stretch4 --robot.mode=control")
        log.info("Start pose: %s", source)
        if not client.start():   # the server loads the policy here, once
            raise SystemExit(f"Could not reach the policy server at {cfg.server_address}")
        listener, events = key_listener(log)
        if listener is None and cfg.confirm_goto:
            log.warning("No interactive keyboard: moving to the start pose without waiting for Enter")
            cfg.confirm_goto = False
        receiver = threading.Thread(target=client.receive_actions, daemon=True, name="actions")
        receiver.start()
        client.start_barrier.wait()   # receive_actions waits for the control loop here
        results = run_episodes(client, cfg, events, start_pose,
                               lambda text: log_say(text, cfg.play_sounds), log)
        log.info(summary(results))
    finally:
        if listener is not None:
            listener.stop()
        client.end_episode()
        client.stop()


def main():
    register_third_party_plugins()
    _run()


if __name__ == "__main__":
    main()
