"""Pick what each camera records and save a cameras file for lerobot-record.

    stretch4_pick_cameras --out cameras.json
    lerobot-record --robot.type=stretch4 --robot.cameras_file=cameras.json ...

--cameras (default head_left head_right gripper): head_left, head_right, gripper (the right
gripper eye) or gripper_left / gripper_right;
each is recorded as observation.images.<name>, the gripper eyes as observation.images.gripper.
NAME=EYE records EYE under another name.

Drag a box on each camera: that part is recorded, cropped then resized to --size. The window
shows each camera live with its box and what will be recorded. Boxes keep the aspect of --size,
so every camera has one shape (diffusion needs that). Each camera starts at the largest
centred crop: drag inside it to move it.

mouse and keys:
    drag outside a box       draw a new box on that camera
    drag inside the box      move it
    drag a corner / an edge  resize it
    wheel                    grow / shrink it about its centre
    m                        copy the last-drawn head eye's box to the other head eye
    r                        clear all boxes
    Enter / Space            save and exit
    q / Esc                  quit without saving
"""

import argparse
import json
import sys
from pathlib import Path

import cv2

from lerobot_camera_oak import CAMERAS
from lerobot_camera_oak.configuration_oak import center_crop

from .config_stretch4 import CAMERAS_FILE_SCHEMA, DEFAULT_IMAGE_SIZE, cameras_from_file
from .crop_ui import Picker, draw_overlay

DEFAULT_CAMERAS = ("head_left", "head_right", "gripper")


def parse_cameras(items):
    """--cameras values -> {dataset column: physical camera}."""
    out = {}
    for item in items:
        name, sep, eye = item.partition("=")
        if not sep:
            eye = "gripper_right" if item == "gripper" else item
            name = "gripper" if eye.startswith("gripper") else eye
        if eye not in CAMERAS or not name:
            raise SystemExit(f"--cameras: {item!r} is not one of head_left, head_right, gripper, "
                             f"gripper_left, gripper_right (or NAME=EYE)")
        if name in out:
            raise SystemExit(f"--cameras: {name!r} given twice")
        out[name] = eye
    return out


# ------------------------------------------------------------------- frame sources ---

class LiveFrames:
    """Full-resolution upright frames from the robot's OAK cameras (no crop, no resize)."""

    def __init__(self, eyes):
        from lerobot_camera_oak import OAKCamera, OAKCameraConfig
        self.cams = {}
        try:
            for name, eye in eyes.items():
                cam = OAKCamera(OAKCameraConfig(camera=eye))
                cam.connect()
                self.cams[name] = cam
        except BaseException:
            self.close()
            raise

    def frames(self):
        return {name: cv2.cvtColor(cam.read_latest(), cv2.COLOR_RGB2BGR)
                for name, cam in self.cams.items()}

    def label(self):
        return "live"

    def close(self):
        for cam in self.cams.values():
            try:
                cam.disconnect()
            except Exception:
                pass


# ---------------------------------------------------------------------- picking ---

def effective_box(box, img, size):
    """What the camera will record: the box, or the centred crop of size's aspect."""
    return tuple(box) if box else tuple(center_crop((img.shape[1], img.shape[0]), size))


def readout(name, eye, box, img, size):
    x, y, w, h = effective_box(box, img, size)
    return [f"{name} ({eye})  upright {img.shape[1]}x{img.shape[0]}",
            f"{'crop' if box else 'centred crop'} {x},{y},{w},{h}  ->  recorded {size[1]}x{size[0]}"]


def save(path, eyes, boxes, size):
    data = {"schema": CAMERAS_FILE_SCHEMA,
            "cameras": {name: {"camera": eye, "crop": list(boxes[name]) if boxes.get(name) else None,
                               "size": list(size)} for name, eye in eyes.items()}}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n")
    cameras = cameras_from_file(path)   # the same loader lerobot-record uses
    print(f"wrote {path}")
    for name, cfg in cameras.items():
        print(f"  observation.images.{name}: {cfg.camera}, crop {cfg.crop}, recorded {cfg.width}x{cfg.height}")
    print(f"\n  lerobot-record --robot.type=stretch4 --robot.cameras_file={path} ...")


# ---------------------------------------------------------------------- window ---

class PickerApp:
    """One tkinter window, a canvas per camera. OpenCV draws the overlay (crop_ui); tkinter
    only shows it: LeRobot's venv has an OpenCV built without a GUI (opencv-python-headless)."""

    REFRESH_MS = 60

    def __init__(self, source, eyes, boxes, frames, out, size, max_height, root=None):
        import tkinter as tk

        self.source, self.eyes, self.frames = source, eyes, frames
        self.out, self.size = out, list(size)
        self.result = None           # 0 saved, 1 quit
        try:
            self.root = root or tk.Tk()
        except tk.TclError:
            raise SystemExit("no display: stretch4_pick_cameras needs a screen (run it at the robot)")
        self.root.title("stretch4_pick_cameras")
        self.pickers, self.canvases, self.handlers, self._images = {}, {}, {}, {}
        self.last_drawn = None       # the camera whose box was drawn last (for Match)
        for col, (name, img) in enumerate(frames.items()):
            size_wh = (img.shape[1], img.shape[0])
            # Every box keeps the recorded aspect, so the resize never stretches it.
            picker = Picker(min(1.0, max_height / float(img.shape[0])), size_wh,
                            aspect=self.size[1] / self.size[0])
            picker.box = boxes.get(name)
            canvas = tk.Canvas(self.root, width=int(size_wh[0] * picker.scale),
                               height=int(size_wh[1] * picker.scale), highlightthickness=0)
            canvas.grid(row=0, column=col, padx=4, pady=4, sticky="n")
            handlers = {seq: (lambda e, pk=picker, ev=ev: pk.on_mouse(ev, e.x, e.y, 0, None))
                        for seq, ev in (("<ButtonPress-1>", cv2.EVENT_LBUTTONDOWN),
                                        ("<B1-Motion>", cv2.EVENT_MOUSEMOVE),
                                        ("<ButtonRelease-1>", cv2.EVENT_LBUTTONUP))}
            press = handlers["<ButtonPress-1>"]
            handlers["<ButtonPress-1>"] = lambda e, n=name, press=press: (
                setattr(self, "last_drawn", n), self.seed(n), press(e))
            # The wheel grows / shrinks the box: Button-4/5 on X11, MouseWheel elsewhere.
            handlers["<Button-4>"] = lambda e, n=name: self.grow(n, 1.05)
            handlers["<Button-5>"] = lambda e, n=name: self.grow(n, 1 / 1.05)
            handlers["<MouseWheel>"] = lambda e, n=name: self.grow(n, 1.05 if e.delta > 0 else 1 / 1.05)
            for seq, handler in handlers.items():
                canvas.bind(seq, handler)
            self.pickers[name], self.canvases[name] = picker, canvas
            self.handlers[name] = handlers   # the functions bound above (tests call them)
        bar = tk.Frame(self.root)
        bar.grid(row=1, column=0, columnspan=len(frames), sticky="w", padx=4, pady=(0, 4))
        tk.Button(bar, text="Save (Enter)", command=self.save).pack(side="left")
        tk.Button(bar, text="Clear (r)", command=self.clear).pack(side="left", padx=4)
        if self.head_pair():
            tk.Button(bar, text="Match head L/R (m)", command=self.match_heads).pack(side="left")
            self.root.bind("m", lambda e: self.match_heads())
        tk.Button(bar, text="Quit (q)", command=self.quit).pack(side="left")
        self.status = tk.Label(bar, text=f"drag a box on each camera; recorded at "
                                         f"{self.size[1]}x{self.size[0]}")
        self.status.pack(side="left", padx=12)
        for key in ("<Return>", "<space>"):
            self.root.bind(key, lambda e: self.save())
        self.root.bind("r", lambda e: self.clear())
        for key in ("q", "<Escape>"):
            self.root.bind(key, lambda e: self.quit())
        self.root.protocol("WM_DELETE_WINDOW", self.quit)

    def seed(self, name):
        """Make the shown centred crop the camera's box, so it can be moved and resized."""
        pk = self.pickers[name]
        if not pk.box:
            pk.box = effective_box(None, self.frames[name], self.size)

    def grow(self, name, factor):
        self.seed(name)
        self.pickers[name].grow(factor)

    def boxes(self):
        return {name: pk.box for name, pk in self.pickers.items() if pk.box}

    def redraw(self):
        from PIL import Image, ImageTk

        for name, pk in self.pickers.items():
            img = self.frames[name]
            lines = [self.source.label()] + readout(name, self.eyes[name], pk.box, img, self.size)
            shown = effective_box(pk.box, img, self.size)
            rgb = cv2.cvtColor(draw_overlay(img, shown, pk.scale, lines), cv2.COLOR_BGR2RGB)
            photo = ImageTk.PhotoImage(Image.fromarray(rgb))
            canvas = self.canvases[name]
            canvas.delete("all")
            canvas.create_image(0, 0, anchor="nw", image=photo)
            self._images[name] = photo       # tkinter shows it only while a reference is held

    def tick(self):
        if self.result is not None:
            return
        self.frames = self.source.frames()
        self.redraw()
        self.root.after(self.REFRESH_MS, self.tick)

    def head_pair(self):
        """(left name, right name) of the head_left and head_right cameras, if both are open."""
        by_eye = {eye: name for name, eye in self.eyes.items()}
        if "head_left" in by_eye and "head_right" in by_eye:
            return by_eye["head_left"], by_eye["head_right"]
        return None

    def match_heads(self):
        """Copy the box of the head eye drawn last (else the one that has a box) onto the other.
        Both eyes have the same upright resolution, so the same pixels are the same framing
        (less the stereo offset between the eyes)."""
        pair = self.head_pair()
        if not pair:
            return
        if self.last_drawn in pair and self.pickers[self.last_drawn].box:
            src = self.last_drawn
        else:
            src = next((n for n in pair if self.pickers[n].box), None)
        if src is None:
            self.status.config(text="draw a box on a head eye first, then Match")
            return
        dst = pair[1] if src == pair[0] else pair[0]
        self.pickers[dst].box = self.pickers[src].box
        self.pickers[dst].dirty = True
        self.status.config(text=f"{dst} box set to {src}'s {self.pickers[src].box}")

    def clear(self):
        for pk in self.pickers.values():
            pk.box = None

    def save(self):
        save(self.out, self.eyes, self.boxes(), self.size)
        self.status.config(text=f"saved {self.out}")
        self.result = 0
        self.root.quit()

    def quit(self):
        print("not saved")
        self.result = 1
        self.root.quit()

    def run(self):
        self.tick()
        self.root.mainloop()
        self.root.destroy()
        return self.result


def main(argv=None):
    summary, _, epilog = __doc__.partition("\n\n")
    p = argparse.ArgumentParser(description=summary, epilog=epilog,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, required=True, metavar="FILE", help="cameras file to write")
    p.add_argument("--cameras", nargs="+", metavar="CAMERA",
                   help=f"the cameras to record (default: {' '.join(DEFAULT_CAMERAS)})")
    p.add_argument("--size", type=int, nargs=2, metavar=("H", "W"), default=None,
                   help=f"recorded size of every camera (default {DEFAULT_IMAGE_SIZE[0]} {DEFAULT_IMAGE_SIZE[1]})")
    p.add_argument("--max-height", type=int, default=800, metavar="PX",
                   help="window height per camera (default 800); crops stay in native pixels")
    args = p.parse_args(argv)

    eyes = parse_cameras(args.cameras or DEFAULT_CAMERAS)
    size = list(args.size) if args.size else list(DEFAULT_IMAGE_SIZE)
    source = LiveFrames(eyes)
    try:
        return PickerApp(source, eyes, {}, source.frames(), args.out, size, args.max_height).run()
    finally:
        source.close()


if __name__ == "__main__":
    sys.exit(main())
