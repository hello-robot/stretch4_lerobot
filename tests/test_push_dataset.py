"""stretch4_push_dataset: list, pick, rename, confirm, push (the Hub is faked)."""

import json

import pytest

from lerobot_robot_stretch4 import push_dataset as cli


def make_dataset(home, repo_id, episodes=3, prepared=True):
    """A v3 dataset's meta/; prepared as stretch4_fix_stats leaves it unless prepared=False."""
    from lerobot_robot_stretch4 import sidecar
    root = home / repo_id
    (root / "meta").mkdir(parents=True)
    feats = {f"observation.images.{c}": {"dtype": "video", "shape": [240, 384, 3]}
             for c in ("head_left", "head_right", "gripper")}
    (root / "meta" / "info.json").write_text(json.dumps(
        {"codebase_version": "v3.0", "fps": 30, "total_episodes": episodes,
         "total_frames": 90 * episodes, "features": feats}))
    side = sidecar.make("lerobot-record", {}, [], [], 3, {})
    if prepared:
        side["prepared"] = {"frames": 90 * episodes, "episodes": episodes, "held_joints": {}}
    sidecar.save(root, side)
    return root.resolve()


@pytest.fixture
def hub(tmp_path, monkeypatch):
    home = tmp_path / "lerobot"
    home.mkdir()
    pushed, existing = [], set()
    monkeypatch.setattr(cli, "lerobot_home", lambda: home)
    monkeypatch.setattr(cli, "whoami", lambda: "olagh")
    monkeypatch.setattr(cli, "hub_exists", lambda repo_id: repo_id in existing)
    monkeypatch.setattr(cli, "push", lambda repo_id, root, private: pushed.append((repo_id, root, private)))
    return home, pushed, existing


def answers(monkeypatch, *replies):
    """Feed input() these replies in order; fail if it asks more."""
    it = iter(replies)
    monkeypatch.setattr("builtins.input", lambda prompt: next(it))


def test_list(hub, capsys):
    home, _, _ = hub
    make_dataset(home, "olagh/brush_nail_20260930_155157")
    assert cli.main(["--list"]) == 0
    out = capsys.readouterr().out
    assert "1. olagh/brush_nail_20260930_155157" in out and "3 episodes" in out


def test_interactive_pick_by_number_and_accept_the_suggested_name(hub, monkeypatch):
    home, pushed, _ = hub
    make_dataset(home, "olagh/a_mouse_20260928_112927")
    root = make_dataset(home, "olagh/brush_nail_20260930_155157")
    answers(monkeypatch, "2", "", "", "y")       # dataset 2, suggested name, private, push
    assert cli.main([]) == 0
    assert pushed == [("olagh/brush_nail", root, True)]   # the timestamp stamp dropped


def test_interactive_rename_and_public(hub, monkeypatch):
    home, pushed, _ = hub
    root = make_dataset(home, "olagh/brush_nail_20260930_155157")
    answers(monkeypatch, "olagh/brush_nail_20260930_155157", "brush_nails_v1", "n", "y")
    assert cli.main([]) == 0
    assert pushed == [("olagh/brush_nails_v1", root, False)]   # bare name: same user


def test_interactive_reasks_bad_answers(hub, monkeypatch):
    home, pushed, _ = hub
    root = make_dataset(home, "olagh/brush_nail_x")
    answers(monkeypatch, "7", "nope", "1", "bad name!", "team/brush", "", "y")
    assert cli.main([]) == 0
    assert pushed == [("team/brush", root, True)]


def test_warns_when_the_hub_name_exists_and_no_means_no(hub, monkeypatch, capsys):
    home, pushed, existing = hub
    make_dataset(home, "olagh/brush_nail_20260930_155157")
    existing.add("olagh/brush_nail")
    answers(monkeypatch, "1", "", "", "n")
    assert cli.main([]) == 1
    assert "already exists" in capsys.readouterr().out
    assert not pushed


def test_scripted_with_yes(hub):
    home, pushed, _ = hub
    root = make_dataset(home, "olagh/brush_nail_20260930_155157")
    assert cli.main([str(root), "--repo-id", "olagh/brush_nail", "--public", "--yes"]) == 0
    assert cli.main(["olagh/brush_nail_20260930_155157", "--yes"]) == 0
    assert pushed == [("olagh/brush_nail", root, False),
                      ("olagh/brush_nail_20260930_155157", root, True)]


def test_refuses_bad_input(hub):
    home, pushed, _ = hub
    with pytest.raises(SystemExit, match="no LeRobot dataset"):
        cli.main(["olagh/missing", "--yes"])
    make_dataset(home, "olagh/empty", episodes=0)
    with pytest.raises(SystemExit, match="no episodes"):
        cli.main(["olagh/empty", "--yes"])
    make_dataset(home, "olagh/ok")
    with pytest.raises(SystemExit, match="USER/NAME"):
        cli.main(["olagh/ok", "--repo-id", "brush_nail", "--yes"])
    with pytest.raises(SystemExit):
        cli.main(["--yes"])                      # --yes needs a dataset
    assert not pushed


def test_unprepared_dataset_is_prepared_before_pushing(hub, monkeypatch):
    home, pushed, _ = hub
    root = make_dataset(home, "olagh/brush_nail_20260930_155157", prepared=False)
    prepared = []
    from lerobot_robot_stretch4 import prepare_dataset
    monkeypatch.setattr(prepare_dataset, "prepare", lambda r, **kw: prepared.append((r, kw)))
    assert cli.main([str(root), "--repo-id", "olagh/brush_nail", "--yes"]) == 0
    assert prepared == [(root, {})] and pushed


def test_no_fix_stats_pushes_untouched(hub, monkeypatch):
    home, pushed, _ = hub
    root = make_dataset(home, "olagh/brush_nail_20260930_155157", prepared=False)
    from lerobot_robot_stretch4 import prepare_dataset
    monkeypatch.setattr(prepare_dataset, "prepare", lambda r, **kw: pytest.fail("prepared"))
    assert cli.main([str(root), "--repo-id", "olagh/brush_nail", "--yes", "--no-fix-stats"]) == 0
    assert pushed


def test_changed_relative_dataset_is_prepared_relative_again(hub, monkeypatch):
    # stretch4_fix_stats --relative, then more episodes recorded: the re-prepare keeps --relative
    home, pushed, _ = hub
    root = make_dataset(home, "olagh/brush_nail_20260930_155157")
    from lerobot_robot_stretch4 import prepare_dataset, sidecar
    side = sidecar.read(root)
    side["prepared"].update(frames=1, relative=True, chunk_size=50, tail=0.1,
                            relative_exclude=["grip", "base_v", "base_omega"])
    sidecar.save(root, side)
    prepared = []
    monkeypatch.setattr(prepare_dataset, "prepare", lambda r, **kw: prepared.append((r, kw)))
    assert cli.main([str(root), "--repo-id", "olagh/brush_nail", "--yes"]) == 0
    assert prepared == [(root, {"relative": True, "chunk_size": 50, "tail": 0.1,
                                "exclude": ("grip", "base_v", "base_omega")})] and pushed
