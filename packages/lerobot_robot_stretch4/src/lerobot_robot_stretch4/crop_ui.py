"""Mouse-drawn crop boxes on an upright camera frame, for stretch4_pick_cameras.

Boxes are kept in upright (post-rotation) native pixels - the space every crop flag and camera
config uses - while the window shows the frame scaled to fit.

Mouse, per box:
    drag outside the box     draw a new box
    drag inside the box      move it
    drag a corner / an edge  resize it (the handles drawn on the box)
    wheel                    grow / shrink it about its centre (where the GUI backend reports it)
"""

import cv2
import numpy as np

GREEN = (0, 255, 0)
BLACK = (0, 0, 0)
HANDLE_PX = 10          # grab distance for a corner / edge, in display pixels


class Picker:
    """Mouse state for one window. Box is kept in native pixels; the display is scaled.

    aspect: optional width/height the box is held to, so that crops of several cameras
    resize to one output shape. Drawing and corner / left-right edge drags let the width
    drive the height; top-bottom edge drags let the height drive the width."""

    def __init__(self, scale, size, aspect=None):
        self.scale = scale              # display px per native px
        self.w, self.h = size           # native image size
        self.aspect = aspect
        self.box = None                 # (x, y, w, h) native, normalized
        self.drag = None                # (mode, x0, y0[, box at press]) while the button is down
        self.dirty = True

    def to_native(self, dx, dy):
        x = int(round(dx / self.scale))
        y = int(round(dy / self.scale))
        return max(0, min(self.w, x)), max(0, min(self.h, y))

    # -------------------------------------------------------------------- mouse ---

    def hit(self, x, y):
        """What a press at native (x, y) grabs: 'new', 'move', or the sides being resized as
        a string of l/r/t/b (e.g. 'rb' for the bottom-right corner)."""
        if not self.box:
            return "new"
        bx, by, bw, bh = self.box
        tol = HANDLE_PX / self.scale
        if not (bx - tol <= x <= bx + bw + tol and by - tol <= y <= by + bh + tol):
            return "new"
        sides = ""
        if abs(x - bx) <= tol:
            sides += "l"
        elif abs(x - (bx + bw)) <= tol:
            sides += "r"
        if abs(y - by) <= tol:
            sides += "t"
        elif abs(y - (by + bh)) <= tol:
            sides += "b"
        return sides or "move"

    def on_mouse(self, event, dx, dy, flags, _):
        if event == cv2.EVENT_LBUTTONDOWN:
            x, y = self.to_native(dx, dy)
            mode = self.hit(x, y)
            if mode == "new":
                self.box = None
                self.drag = ("new", x, y)
            else:
                self.drag = (mode, x, y, self.box)
            self.dirty = True
        elif event in (cv2.EVENT_MOUSEMOVE, cv2.EVENT_LBUTTONUP) and self.drag:
            self._drag_to(*self.to_native(dx, dy))
            if event == cv2.EVENT_LBUTTONUP:
                self.drag = None
        elif event == cv2.EVENT_MOUSEWHEEL and self.box:
            self.grow(1.05 if cv2.getMouseWheelDelta(flags) > 0 else 1 / 1.05)

    def _drag_to(self, x, y):
        mode, x0, y0 = self.drag[:3]
        if mode == "new":
            self._set((x0, y0), (x, y))
            return
        bx, by, bw, bh = self.drag[3]
        if mode == "move":
            nx = max(0, min(self.w - bw, bx + x - x0))
            ny = max(0, min(self.h - bh, by + y - y0))
            self.box = (nx, ny, bw, bh)
        else:
            self._resize(mode, x, y, (bx, by, bw, bh))
        self.dirty = True

    # ---------------------------------------------------------------- geometry ---

    def _fit(self, ax, ay, sx, sy, w, h, by_height=False):
        """Box anchored at corner (ax, ay), extending sx/sy (+1 right/down, -1 left/up),
        held to self.aspect (from w, or from h with by_height) and shrunk to fit the image."""
        if self.aspect:
            if by_height:
                w = int(round(h * self.aspect))
            else:
                h = int(round(w / self.aspect))
        room_w = self.w - ax if sx > 0 else ax
        room_h = self.h - ay if sy > 0 else ay
        if w > room_w or h > room_h:
            if self.aspect:
                k = min(room_w / max(w, 1), room_h / max(h, 1))
                w = int(w * k)
                h = min(room_h, int(round(w / self.aspect)))
            w, h = min(w, room_w), min(h, room_h)
        x = ax if sx > 0 else ax - w
        y = ay if sy > 0 else ay - h
        return (x, y, w, h) if w > 1 and h > 1 else None

    def _set(self, a, b):
        """A new box dragged from a to b."""
        (x0, y0), (x1, y1) = a, b
        self.box = self._fit(x0, y0, 1 if x1 >= x0 else -1, 1 if y1 >= y0 else -1,
                             abs(x1 - x0), abs(y1 - y0))
        self.dirty = True

    def _resize(self, sides, x, y, box):
        """Drag `sides` of `box` to (x, y); the opposite sides stay put. With an aspect, an
        edge drag keeps the box's other edge (top for l/r, left for t/b) fixed."""
        bx, by, bw, bh = box
        x1, y1 = bx + bw, by + bh
        if "l" in sides or "r" in sides:
            ax = x1 if "l" in sides else bx
            sx, w = (1 if x >= ax else -1), abs(x - ax)   # past the opposite edge: it flips
        else:
            ax, sx, w = bx, 1, bw
        if "t" in sides or "b" in sides:
            ay = y1 if "t" in sides else by
            sy, h = (1 if y >= ay else -1), abs(y - ay)
        else:
            ay, sy, h = by, 1, bh
        new = self._fit(ax, ay, sx, sy, w, h, by_height=sides in ("t", "b"))
        if new:
            self.box = new

    def grow(self, factor):
        """Scale the box about its centre, keeping it inside the image (and at the aspect)."""
        if not self.box:
            return
        bx, by, bw, bh = self.box
        cx, cy = bx + bw / 2, by + bh / 2
        w = max(2, int(round(bw * factor)))
        h = int(round(w / self.aspect)) if self.aspect else max(2, int(round(bh * factor)))
        if w > self.w or h > self.h:
            k = min(self.w / w, self.h / h)
            w = int(w * k)
            h = min(self.h, int(round(w / self.aspect))) if self.aspect else int(h * k)
        x = int(round(max(0, min(self.w - w, cx - w / 2))))
        y = int(round(max(0, min(self.h - h, cy - h / 2))))
        self.box = (x, y, w, h)
        self.dirty = True

    def fill(self):
        """The largest box at self.aspect that fits, centred: a starting box to move."""
        if not self.aspect:
            self.box = None
            return
        w = min(self.w, int(self.h * self.aspect))
        h = min(self.h, int(round(w / self.aspect)))
        self.box = ((self.w - w) // 2, (self.h - h) // 2, w, h)
        self.dirty = True

    def conform(self):
        """Reshape the current box to self.aspect, keeping its centre and width (shrunk to
        fit). For a box that predates the aspect lock."""
        if self.box and self.aspect:
            self.grow(1.0)


def draw_overlay(img, box, scale, label_lines, color=GREEN):
    """Scaled BGR copy with the box (and its resize handles) and a readout drawn on it."""
    disp = cv2.resize(img, (int(img.shape[1] * scale), int(img.shape[0] * scale)),
                      interpolation=cv2.INTER_AREA)
    if box:
        x, y, w, h = (int(round(v * scale)) for v in box)
        # Dim everything outside the box so the framing is easy to judge.
        shade = disp.copy()
        cv2.rectangle(shade, (0, 0), (disp.shape[1], disp.shape[0]), BLACK, -1)
        cv2.rectangle(shade, (x, y), (x + w, y + h), (255, 255, 255), -1)
        disp = np.where(shade == 0, (disp * 0.45).astype(np.uint8), disp)
        cv2.rectangle(disp, (x, y), (x + w, y + h), color, 2)
        r = 5
        for hx in (x, x + w // 2, x + w):
            for hy in (y, y + h // 2, y + h):
                if (hx, hy) != (x + w // 2, y + h // 2):
                    cv2.rectangle(disp, (hx - r, hy - r), (hx + r, hy + r), color, -1)
    for i, line in enumerate(label_lines):
        if not line:
            continue
        x, y = 10, 26 + i * 24
        # A backing box, not a thick black outline: OpenCV 5 advances thick text wider than
        # thin text, so an outline drawn under the label drifts off it.
        (tw, th), base = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(disp, (x - 3, y - th - 4), (x + tw + 3, y + base + 1), BLACK, -1)
        cv2.putText(disp, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1, cv2.LINE_AA)
    return disp
