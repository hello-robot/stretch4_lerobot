"""stretch4_fix_stats (prepare_dataset.py) on a small dataset: held joints written as constants,
stats floored, the "prepared" record, staleness, repeatability and --undo."""

import json

import numpy as np
import pyarrow.parquet as pq
import pytest

from lerobot_robot_stretch4 import prepare_dataset, sidecar
from lerobot_robot_stretch4.columns import POLICY_STATE_COLUMNS, VELOCITY_ACTION_COLUMNS
from lerobot_robot_stretch4.stats_floor import STATS_FLOOR

NAMES = list(VELOCITY_ACTION_COLUMNS)
EPISODES, FRAMES = 12, 40


@pytest.fixture
def dataset(tmp_path):
    """wrist_roll moves in one episode of 12 (too rare to learn), lift never moves, the rest move."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    ds = LeRobotDataset.create("local/fix_stats", fps=30, root=tmp_path / "ds", use_videos=False,
                               features={
                                   "observation.state": {"dtype": "float32", "shape": (9,),
                                                         "names": list(POLICY_STATE_COLUMNS)},
                                   "action": {"dtype": "float32", "shape": (9,), "names": NAMES}})
    rng = np.random.default_rng(0)
    for ep in range(EPISODES):
        for t in range(FRAMES):
            x = rng.normal(0, 0.1, 9).astype(np.float32)
            x[NAMES.index("grip_mm")] = 30 + 20 * rng.random()
            x[NAMES.index("wrist_roll")] = 0.3 + (0.5 * t / FRAMES if ep == 0 else 1e-4 * rng.normal())
            x[NAMES.index("lift")] = 0.5 + 1e-5 * rng.normal()
            ds.add_frame({"observation.state": x, "action": x, "task": "t"})
        ds.save_episode()
    ds.finalize()
    sidecar.save(ds.root, sidecar.make("lerobot-record", {}, POLICY_STATE_COLUMNS, NAMES, 3, {}))
    return ds.root


def column(root, feature):
    return np.concatenate([np.stack(pq.read_table(f)[feature].to_numpy(zero_copy_only=False))
                           for f in sorted((root / "data").glob("*/*.parquet"))])


def stats(root):
    return json.loads((root / "meta" / "stats.json").read_text())


def test_holds_rare_joints_and_floors_stats(dataset):
    record = prepare_dataset.prepare(dataset, verbose=False)
    assert set(record["held_joints"]) == {"lift", "wrist_roll"}
    for feature in ("observation.state", "action"):
        x = column(dataset, feature)
        for joint in ("lift", "wrist_roll"):
            i = NAMES.index(joint)
            assert np.ptp(x[:, i]) == 0                              # written as one constant
            st = stats(dataset)[feature]
            assert st["max"][i] - st["min"][i] == pytest.approx(2 * STATS_FLOOR[joint])
        assert np.ptp(x[:, NAMES.index("arm")]) > 0                   # learned joints untouched
    assert sidecar.read(dataset)["prepared"] == record
    # what the floor fixed, spread before -> after: lift never moved (held, spread 0 -> floor)
    before, after = record["floored"]["action"]["lift"]["min/max"]
    assert before == pytest.approx(0) and after == pytest.approx(2 * STATS_FLOOR["lift"])
    assert prepare_dataset.is_stale(dataset) is None


def test_rerun_starts_from_the_recorded_data(dataset):
    original = column(dataset, "action")
    prepare_dataset.prepare(dataset, verbose=False)
    record = prepare_dataset.prepare(dataset, keep=["wrist_roll"], verbose=False)
    assert set(record["held_joints"]) == {"lift"}
    i = NAMES.index("wrist_roll")
    assert np.array_equal(column(dataset, "action")[:, i], original[:, i])


def test_undo_restores_the_recorded_data(dataset):
    original = column(dataset, "action")
    prepare_dataset.prepare(dataset, verbose=False)
    prepare_dataset.undo(dataset)
    assert np.array_equal(column(dataset, "action"), original)
    assert "prepared" not in sidecar.read(dataset)
    assert not (dataset / prepare_dataset.ORIGINALS).exists()


def test_stale_after_more_episodes(dataset):
    prepare_dataset.prepare(dataset, verbose=False)
    info_path = dataset / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    info["total_episodes"] += 1
    info_path.write_text(json.dumps(info))
    assert "episodes" in prepare_dataset.is_stale(dataset)


def test_relative_action_stats(dataset):
    record = prepare_dataset.prepare(dataset, relative=True, chunk_size=10, verbose=False)
    assert record["relative"] and record["tail"] == 0.1
    assert record["relative_exclude"] == list(prepare_dataset.RELATIVE_EXCLUDE)


@pytest.mark.parametrize("relative", [False, True])
def test_joint_bounds_are_absolute(dataset, relative):
    # --relative leaves action - state in stats.json; the robot's joint clamp must still be the
    # absolute range, or every target would be clamped to ~0
    from lerobot_robot_stretch4.columns import POSE_JOINTS
    from lerobot_robot_stretch4.config_stretch4 import bounds_from_dataset
    prepare_dataset.prepare(dataset, relative=relative, chunk_size=10, verbose=False)
    bounds = bounds_from_dataset(dataset, POSE_JOINTS)
    x = column(dataset, "action")
    for j in POSE_JOINTS:
        i = NAMES.index(j)
        lo, hi = bounds[j]
        assert lo < x[:, i].min() and x[:, i].max() < hi, j
    assert bounds["grip_mm"][0] > 25                       # 30..50 mm, not a relative range


def test_options_from_redo_the_same_preparation(dataset):
    first = prepare_dataset.prepare(dataset, relative=True, chunk_size=10, keep=["wrist_roll"],
                                    verbose=False)
    again = prepare_dataset.prepare(dataset, verbose=False,
                                    **prepare_dataset.options_from(first))
    assert again == first
