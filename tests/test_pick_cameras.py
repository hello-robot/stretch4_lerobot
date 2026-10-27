"""stretch4_pick_cameras: the tkinter window, the cameras file it writes, and lerobot-record reading it (--robot.cameras_file)."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from lerobot_robot_stretch4 import Stretch4, Stretch4Config
from lerobot_robot_stretch4.config_stretch4 import CAMERAS_FILE_SCHEMA, cameras_from_file
from lerobot_robot_stretch4.pick_cameras import PickerApp, save

EYES = {"head_right": "head_right", "gripper": "gripper_right"}
SIZE = [240, 384]   # 1.6:1


class FakeFrames:
    """Upright native frames of the default cameras (head 1200x1920, gripper 640x400)."""

    def frames(self):
        return {"head_right": np.zeros((1920, 1200, 3), np.uint8),
                "gripper": np.zeros((400, 640, 3), np.uint8)}

    def label(self):
        return "fake"

    def close(self):
        pass


@pytest.fixture
def app(tmp_path):
    tk = pytest.importorskip("tkinter")
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("no display")
    root.withdraw()                      # never shown
    source = FakeFrames()
    a = PickerApp(source, EYES, {}, source.frames(), tmp_path / "cameras.json", SIZE,
                  max_height=800, root=root)
    a.redraw()
    yield a
    root.destroy()


def drag(app, name, start, end):
    """Press, move, release through the handlers bound to the canvas (display coordinates).
    Tk does not deliver generated mouse events to a window that is never shown, so the
    handlers are called directly; test_mouse_handlers_are_bound checks they are the bound ones."""
    for seq, (x, y) in (("<ButtonPress-1>", start), ("<B1-Motion>", end), ("<ButtonRelease-1>", end)):
        app.handlers[name][seq](SimpleNamespace(x=x, y=y))


def test_mouse_handlers_are_bound(app):
    for name, canvas in app.canvases.items():
        for seq in ("<ButtonPress-1>", "<B1-Motion>", "<ButtonRelease-1>", "<Button-4>",
                    "<Button-5>", "<MouseWheel>"):
            assert canvas.bind(seq), f"{name}: nothing bound to {seq}"


def test_boxes_are_native_pixels_at_the_recorded_aspect(app):
    head = app.pickers["head_right"]
    assert head.scale == pytest.approx(800 / 1920)
    drag(app, "head_right", (0, 100), (500, 700))       # a taller drag than 1.6:1
    x, y, w, h = head.box
    assert (x, y) == (0, 240) and w == 1200 and h == 750   # held to 1.6:1, in native pixels
    drag(app, "gripper", (10, 10), (330, 60))
    gw, gh = app.pickers["gripper"].box[2:]
    assert abs(gw / gh - 1.6) < 0.02
    app.redraw()


def test_the_starting_centred_crop_can_be_moved(app):
    head = app.pickers["head_right"]
    y, w, h = 585, 1200, 750                               # centred crop of 1200x1920 at 1.6:1
    cx, cy = 600 * head.scale, (y + h / 2) * head.scale
    drag(app, "head_right", (cx, cy), (cx, cy + 40))       # inside: moved down, same size
    assert head.box[2:] == (w, h) and head.box[1] > y
    app.clear()
    app.handlers["head_right"]["<Button-5>"](SimpleNamespace(x=0, y=0))     # wheel: smaller
    assert head.box[2] < w


def test_a_box_can_be_moved_resized_and_scrolled(app):
    head = app.pickers["head_right"]
    drag(app, "head_right", (100, 100), (300, 200))      # draw
    x, y, w, h = head.box
    cx, cy = (x + w / 2) * head.scale, (y + h / 2) * head.scale
    drag(app, "head_right", (cx, cy), (cx, cy + 40))       # inside: moved down, same size
    assert head.box[2:] == (w, h) and head.box[1] > y and head.box[0] == x
    x, y, w, h = head.box
    right = (x + w) * head.scale
    drag(app, "head_right", (right, cy + 40), (right + 50, cy + 40))   # right edge: wider
    assert head.box[2] > w and abs(head.box[2] / head.box[3] - 1.6) < 0.02
    w = head.box[2]
    app.handlers["head_right"]["<Button-4>"](SimpleNamespace(x=0, y=0))     # wheel up: bigger
    assert head.box[2] > w
    app.handlers["head_right"]["<MouseWheel>"](SimpleNamespace(x=0, y=0, delta=-120))
    app.handlers["head_right"]["<Button-5>"](SimpleNamespace(x=0, y=0))
    assert head.box[2] < w * 1.05
    assert abs(head.box[2] / head.box[3] - 1.6) < 0.02


def test_save_writes_what_lerobot_record_reads(app):
    drag(app, "head_right", (0, 100), (500, 612))
    app.save()
    assert app.result == 0
    data = json.loads(app.out.read_text())
    assert data["schema"] == CAMERAS_FILE_SCHEMA
    assert data["cameras"]["gripper"] == {"camera": "gripper_right", "crop": None, "size": SIZE}
    cfg = Stretch4Config(cameras_file=str(app.out))
    assert cfg.cameras["head_right"].crop == list(app.pickers["head_right"].box)
    assert cfg.cameras["gripper"].crop == [0, 0, 640, 400]              # centred crop = whole frame
    assert {(c.height, c.width) for c in cfg.cameras.values()} == {(240, 384)}


def test_clear_and_quit(app):
    drag(app, "head_right", (0, 100), (500, 612))
    app.clear()
    assert app.boxes() == {}
    app.quit()
    assert app.result == 1 and not app.out.exists()


def test_cameras_file_round_trip_and_errors(tmp_path):
    save(tmp_path / "c.json", EYES, {"head_right": (0, 720, 1200, 750)}, SIZE)
    robot = Stretch4(Stretch4Config(cameras_file=str(tmp_path / "c.json")))
    shapes = {k: v for k, v in robot.observation_features.items() if isinstance(v, tuple)}
    assert shapes == {"head_right": (240, 384, 3), "gripper": (240, 384, 3)}
    (tmp_path / "bad.json").write_text('{"cameras": {}}')
    with pytest.raises(ValueError, match="stretch4_pick_cameras"):
        cameras_from_file(tmp_path / "bad.json")
    with pytest.raises(ValueError, match="not both"):
        Stretch4Config(cameras_file=str(tmp_path / "c.json"), preprocessing_from=str(tmp_path))


def test_match_copies_the_last_drawn_head_eye_onto_the_other(tmp_path):
    tk = pytest.importorskip("tkinter")
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("no display")
    root.withdraw()
    head = np.zeros((1920, 1200, 3), np.uint8)
    source = SimpleNamespace(frames=lambda: {"head_left": head, "head_right": head},
                             label=lambda: "fake", close=lambda: None)
    eyes = {"head_left": "head_left", "head_right": "head_right"}
    app = PickerApp(source, eyes, {}, source.frames(), tmp_path / "cameras.json", SIZE,
                    max_height=800, root=root)
    try:
        app.match_heads()                                   # nothing drawn yet: no-op
        assert not app.boxes()
        drag(app, "head_right", (0, 100), (500, 612))
        drag(app, "head_left", (10, 100), (300, 300))
        app.match_heads()                                   # head_left was drawn last
        assert app.pickers["head_right"].box == app.pickers["head_left"].box
        drag(app, "head_right", (0, 100), (500, 612))
        app.match_heads()
        assert app.pickers["head_left"].box == app.pickers["head_right"].box
        app.save()
        cams = cameras_from_file(app.out)
        assert cams["head_left"].crop == cams["head_right"].crop
    finally:
        root.destroy()


def test_cameras_by_name():
    from lerobot_robot_stretch4.pick_cameras import parse_cameras
    assert parse_cameras(["head_left", "head_right", "gripper"]) == {
        "head_left": "head_left", "head_right": "head_right", "gripper": "gripper_right"}
    assert parse_cameras(["gripper_left"]) == {"gripper": "gripper_left"}
    assert parse_cameras(["wrist=gripper_left"]) == {"wrist": "gripper_left"}
    with pytest.raises(SystemExit, match="not one of"):
        parse_cameras(["head"])
    with pytest.raises(SystemExit, match="given twice"):
        parse_cameras(["gripper", "gripper_left"])
