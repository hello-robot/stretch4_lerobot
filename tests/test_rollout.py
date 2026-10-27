"""stretch4_rollout: one robot_client session, many episodes. Between episodes nothing is
queued or executed, a chunk from the previous episode is dropped, and the keys end, discard or
stop episodes."""

import logging
import threading
import time
from queue import Queue
from types import SimpleNamespace

import pytest
import torch

from lerobot.async_inference.helpers import TimedAction
from lerobot_robot_stretch4 import rollout
from lerobot_robot_stretch4.rollout import EPISODE_GAP, RolloutClient, run_episodes, summary

log = logging.getLogger("test")


def bare_client():
    """An RolloutClient without a robot or a server: only what the episode logic touches."""
    c = object.__new__(RolloutClient)
    c.config = SimpleNamespace(aggregate_fn=None, environment_dt=0.001)
    c.stub = SimpleNamespace(readies=0)
    c.stub.Ready = lambda _: setattr(c.stub, "readies", c.stub.readies + 1)
    c.action_queue, c.action_queue_lock = Queue(), threading.Lock()
    c.latest_action, c.latest_action_lock = -1, threading.Lock()
    c.must_go, c.active, c._episode_lock = threading.Event(), threading.Event(), threading.Lock()
    return c


def chunk(first, n=5):
    return [TimedAction(timestamp=0.0, timestep=first + i, action=torch.zeros(9)) for i in range(n)]


def queued(c):
    return [a.get_timestep() for a in c.action_queue.queue]


def test_no_actions_are_queued_between_episodes():
    c = bare_client()
    c._aggregate_action_queues(chunk(0))
    assert queued(c) == []


def test_a_chunk_from_the_previous_episode_is_dropped():
    c = bare_client()
    c.begin_episode()
    start = c.latest_action
    assert start == EPISODE_GAP and c.must_go.is_set() and c.stub.readies == 1
    c._aggregate_action_queues(chunk(start))
    assert queued(c) == [start + 1, start + 2, start + 3, start + 4]
    c.latest_action = start + 40              # 40 actions executed
    c.end_episode()
    assert queued(c) == []

    c.begin_episode()
    # the server was still computing a chunk for the last observation of episode 1
    c._aggregate_action_queues(chunk(start + 40, n=48))
    assert queued(c) == []
    c._aggregate_action_queues(chunk(c.latest_action))
    assert len(queued(c)) == 4 and c.stub.readies == 2


class FakeRobot:
    def __init__(self, on_goto=None, runstopped=False):
        self.gotos, self.on_goto, self.runstopped = 0, on_goto, runstopped

    def is_runstopped(self):
        return self.runstopped

    def goto_pose(self, pose, timeout):
        self.gotos += 1
        if self.on_goto:
            self.on_goto(self.gotos)
        return True, {}


class FakeClient:
    """run_episode plays a script: each entry is the key pressed during that episode
    (None: it runs out its time)."""

    def __init__(self, keys, robot, events):
        self.keys, self.robot, self.events, self.running = list(keys), robot, events, True

    def run_episode(self, task, duration_s, events):
        key = self.keys.pop(0)
        if key is None:
            return duration_s
        events["exit_early"] = True
        if key == "left":
            events["rerecord_episode"] = True
        if key == "esc":
            events["stop_recording"] = True
        return 1.0


def events():
    return {"exit_early": False, "rerecord_episode": False, "stop_recording": False,
            "goto_start": False}


def cfg(n, confirm_goto=False):
    return SimpleNamespace(num_episodes=n, episode_time_s=60.0, reset_time_s=0.0,
                           goto_timeout_s=1.0, task="t", confirm_goto=confirm_goto)


def test_left_during_an_episode_runs_it_again():
    ev, robot = events(), FakeRobot()
    client = FakeClient([None, "left", "right"], robot, ev)
    results = run_episodes(client, cfg(2), ev, {}, lambda _: None, log)
    assert [(r["episode"], r["ended"], r["counted"]) for r in results] == [
        (1, "time_limit", True), (2, "rerun", False), (2, "next", True)]
    assert robot.gotos == 4                   # before each of the 3 runs, and at the end


def test_left_while_waiting_uncounts_the_episode_before():
    ev = events()
    # Left pressed while the arm goes to the start pose after episode 1
    robot = FakeRobot(on_goto=lambda n: ev.update(exit_early=True, rerecord_episode=True)
                      if n == 2 else None)
    client = FakeClient([None, None, None], robot, ev)
    results = run_episodes(client, cfg(2), ev, {}, lambda _: None, log)
    assert [(r["episode"], r["counted"]) for r in results] == [(1, False), (1, True), (2, True)]


def test_esc_stops_and_returns_to_the_start_pose():
    ev, robot = events(), FakeRobot()
    client = FakeClient([None, "esc"], robot, ev)
    results = run_episodes(client, cfg(5), ev, {}, lambda _: None, log)
    assert [r["ended"] for r in results] == ["time_limit", "stopped"]
    assert robot.gotos == 3


def test_the_arm_moves_to_the_start_pose_only_after_enter():
    ev = events()
    robot = FakeRobot(on_goto=lambda n: moves.append(presses[0]))
    moves, presses = [], [0]

    def press_enter():                        # Enter, twice, 50 ms apart
        for _ in range(2):
            time.sleep(0.05)
            presses[0] += 1
            ev["goto_start"] = True

    threading.Thread(target=press_enter, daemon=True).start()
    results = run_episodes(FakeClient([None], robot, ev), cfg(1, confirm_goto=True), ev, {},
                           lambda _: None, log)
    assert len(results) == 1
    assert moves == [1, 2]                    # before the episode, and at the end: each after Enter


def test_enter_while_run_stopped_does_not_move_and_says_why(caplog):
    ev = events()
    robot = FakeRobot(runstopped=True)

    def operator():                           # Enter while run-stopped, release, Enter again
        time.sleep(0.05)
        ev["goto_start"] = True
        time.sleep(0.1)
        assert robot.gotos == 0
        robot.runstopped = False
        for _ in range(2):                    # before the episode, and at the end
            time.sleep(0.1)
            ev["goto_start"] = True

    threading.Thread(target=operator, daemon=True).start()
    with caplog.at_level(logging.ERROR, logger="test"):
        results = run_episodes(FakeClient([None], robot, ev), cfg(1, confirm_goto=True), ev, {},
                               lambda _: None, log)
    assert len(results) == 1 and robot.gotos == 2
    assert any("run-stopped" in r.getMessage() for r in caplog.records)


def test_esc_at_the_prompt_stops_without_moving():
    ev = events()
    robot = FakeRobot()
    # Esc pressed at the first prompt, instead of Enter
    threading.Timer(0.05, lambda: ev.update(exit_early=True, stop_recording=True)).start()
    results = run_episodes(FakeClient([None], robot, ev), cfg(3, confirm_goto=True), ev, {},
                           lambda _: None, log)
    assert results == [] and robot.gotos == 0


def test_the_start_pose_is_read_from_the_given_file(tmp_path):
    import json
    from lerobot_robot_stretch4.columns import POSE_JOINTS
    path = tmp_path / "any_name.json"
    path.write_text(json.dumps({"joints": {j: 0.2 for j in POSE_JOINTS}}))
    pose, source = rollout.resolve_start_pose(SimpleNamespace(start_pose=str(path)))
    assert source == str(path) and pose["lift"] == 0.2
    with pytest.raises(SystemExit, match="missing.json"):
        rollout.resolve_start_pose(SimpleNamespace(start_pose=str(tmp_path / "missing.json")))
    with pytest.raises(SystemExit, match="--start_pose is required"):
        rollout.resolve_start_pose(SimpleNamespace(start_pose=None))


def test_the_summary_lists_only_counted_episodes():
    results = [{"episode": 1, "seconds": 60.0, "ended": "time_limit", "counted": True},
               {"episode": 2, "seconds": 3.2, "ended": "rerun", "counted": False},
               {"episode": 2, "seconds": 12.5, "ended": "next", "counted": True}]
    assert summary(results) == ("Rollout complete: 2 episode(s) counted, 1 discarded\n"
                                "  Episode 1: 60.0 s, time limit reached\n"
                                "  Episode 2: 12.5 s, ended with Right arrow")
    assert summary([]) == "Rollout complete: 0 episode(s) counted, 0 discarded"
