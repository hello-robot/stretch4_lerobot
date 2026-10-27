"""The OAK camera table matches stretch4_body, which is what the robot actually uses."""

import pytest

from lerobot_camera_oak import CAMERAS, OAKCameraConfig


def _rgb_cameras():
    try:
        from stretch4_body.subsystem.cameras.enums.rgb_camera import RGBCameras
    except (ImportError, KeyError) as e:   # KeyError: HELLO_FLEET_PATH/ID unset, i.e. off-robot
        pytest.skip(f"stretch4_body does not import here ({e!r}); run on the robot")
    return RGBCameras


@pytest.mark.parametrize("camera", sorted(CAMERAS))
def test_table_matches_stretch4_body(camera):
    cfg = _rgb_cameras()[camera].config
    h, w = cfg.image_size
    _, _, native_wh, rot = CAMERAS[camera]
    assert native_wh == (w, h), f"{camera}: CAMERAS says {native_wh}, stretch4_body {(w, h)}"
    assert rot == cfg.rotate_number_of_times, (
        f"{camera}: CAMERAS says {rot} quarter turns, stretch4_body {cfg.rotate_number_of_times}")


@pytest.mark.parametrize("camera,crop", [("head_left", [0, 585, 1200, 750]),
                                         ("head_right", [0, 585, 1200, 750]),
                                         ("gripper_right", [0, 0, 640, 400])])
def test_size_sets_the_shape_with_a_centred_crop(camera, crop):
    # Every camera comes out at the size; with no crop of its own, the largest centred crop of
    # that aspect (the whole gripper frame, which is already 1.6:1).
    cfg = OAKCameraConfig(camera=camera, size=[240, 384])
    assert (cfg.height, cfg.width) == (240, 384) and cfg.crop == crop


def test_no_size_keeps_the_native_upright_frame():
    cfg = OAKCameraConfig(camera="head_right")
    assert (cfg.height, cfg.width) == (1920, 1200) and cfg.crop is None


def test_a_crop_that_would_stretch_is_refused():
    with pytest.raises(ValueError, match="stretch"):
        OAKCameraConfig(camera="head_right", crop=[0, 0, 600, 600], size=[240, 384])


def test_decode_crops_before_it_resizes():
    # Left half red, right half blue. Cropping the left quarter first, in native pixels, and
    # then resizing gives pure red; resizing first would move the crop onto other pixels.
    import cv2
    import numpy as np
    from lerobot_camera_oak import OAKCamera

    native = np.zeros((400, 640, 3), np.uint8)
    native[:, :320] = (0, 0, 255)          # BGR red
    native[:, 320:] = (255, 0, 0)          # BGR blue
    jpeg = cv2.imencode(".jpg", native, [cv2.IMWRITE_JPEG_QUALITY, 100])[1]
    cam = OAKCamera(OAKCameraConfig(camera="gripper_right", crop=[0, 0, 160, 100], size=[50, 80]))
    cam._rot = 0
    rgb = cam._decode(jpeg)
    assert rgb.shape == (50, 80, 3)
    assert rgb[..., 0].min() > 200 and rgb[..., 2].max() < 60     # all red, no blue
