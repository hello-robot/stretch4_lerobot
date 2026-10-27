"""Camera frame -> policy pixels: the one implementation recording and deployment both use, so
training and inference cannot disagree about rotation, crop or resize.

    rot90 x rotate_number_of_times -> crop (upright native pixels) -> resize to width
    (INTER_AREA when shrinking) -> BGR2RGB
"""

import cv2
import numpy as np


def apply_crop(img, crop):
    """Crop in upright pixels. Out-of-bounds raises rather than clamping: a silently
    shrunk crop would change the image shape mid-dataset."""
    x, y, w, h = crop
    ih, iw = img.shape[:2]
    if x + w > iw or y + h > ih:
        raise ValueError(f"crop {crop} falls outside the {iw}x{ih} upright image")
    return img[y:y + h, x:x + w]


def preprocess(bgr, rot=0, crop=None, out_width=None):
    """BGR frame as decoded from the camera JPEG -> RGB uint8 HWC.

    rot: the camera's rotate_number_of_times (lossless 90-degree turns).
    crop: (x, y, w, h) in upright native pixels, applied after rotation and before the
    resize, so it stays valid when out_width changes.
    out_width: resize to this width, keeping the aspect ratio. None keeps native size.
    """
    img = bgr
    if rot:
        img = np.ascontiguousarray(np.rot90(img, k=rot))
    if crop:
        img = apply_crop(img, crop)
    if out_width and img.shape[1] != out_width:
        h, w = img.shape[:2]
        out_h = max(1, int(round(h * out_width / float(w))))
        interp = cv2.INTER_AREA if out_width < w else cv2.INTER_LINEAR
        img = cv2.resize(img, (out_width, out_h), interpolation=interp)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
