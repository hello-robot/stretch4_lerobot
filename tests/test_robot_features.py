"""The features lerobot-record writes for --robot.type=stretch4: the shared column names, and one
image shape for every camera."""

from lerobot.datasets.pipeline_features import aggregate_pipeline_dataset_features, create_initial_features
from lerobot.processor import make_default_processors

from lerobot_robot_stretch4 import Stretch4, Stretch4Config
from lerobot_robot_stretch4.columns import VELOCITY_ACTION_COLUMNS


def record_features(robot):
    teleop_proc, _, obs_proc = make_default_processors()
    return {
        **aggregate_pipeline_dataset_features(
            pipeline=teleop_proc, initial_features=create_initial_features(action=robot.action_features),
            use_videos=True),
        **aggregate_pipeline_dataset_features(
            pipeline=obs_proc, initial_features=create_initial_features(observation=robot.observation_features),
            use_videos=True),
    }


def test_record_features_are_the_shared_columns():
    features = record_features(Stretch4(Stretch4Config()))
    assert features["action"]["names"] == VELOCITY_ACTION_COLUMNS
    assert features["observation.state"]["names"] == VELOCITY_ACTION_COLUMNS
    assert features["action"]["shape"] == features["observation.state"]["shape"] == (9,)


def test_every_camera_has_the_image_size():
    features = record_features(Stretch4(Stretch4Config()))
    shapes = {k: tuple(v["shape"]) for k, v in features.items() if k.startswith("observation.images.")}
    assert shapes == {"observation.images.head_right": (240, 384, 3),
                      "observation.images.gripper": (240, 384, 3)}
    shapes = {k: tuple(v["shape"]) for k, v in
              record_features(Stretch4(Stretch4Config(image_size=[224, 224]))).items()
              if k.startswith("observation.images.")}
    assert set(shapes.values()) == {(224, 224, 3)}


def test_grip_effort_is_state_only():
    """--robot.grip_effort=true: a 10-wide state ending in grip_effort, and an unchanged 9-wide
    action."""
    features = record_features(Stretch4(Stretch4Config(grip_effort=True)))
    assert features["observation.state"]["names"] == [*VELOCITY_ACTION_COLUMNS, "grip_effort"]
    assert features["action"]["names"] == VELOCITY_ACTION_COLUMNS
