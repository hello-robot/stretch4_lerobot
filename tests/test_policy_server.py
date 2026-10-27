"""stretch4_policy_server with a small diffusion policy (n_obs_steps=2) on the CPU: stock
LeRobot 0.6 fails an assert on it and returns no chunk; this server returns one, built from the
observation 1/fps earlier, and does not filter observations as "too similar"."""

import pickle
import types

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("lerobot.async_inference.policy_server")

from lerobot_robot_stretch4.columns import POLICY_STATE_COLUMNS

from lerobot_robot_stretch4 import policy_server

CAMS = ["head_left", "gripper"]
H, W = 32, 48


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory):
    from lerobot.configs.types import FeatureType, PolicyFeature
    from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
    from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
    from lerobot.policies.factory import make_pre_post_processors

    cfg = DiffusionConfig(
        device="cpu", n_obs_steps=2, horizon=16, n_action_steps=8, down_dims=(32, 64),
        diffusion_step_embed_dim=32, pretrained_backbone_weights=None, crop_shape=None,
        use_group_norm=True, spatial_softmax_num_keypoints=8, num_inference_steps=2,
        noise_scheduler_type="DDIM", push_to_hub=False)
    cfg.input_features = {"observation.state": PolicyFeature(type=FeatureType.STATE, shape=(9,)),
                          **{f"observation.images.{c}": PolicyFeature(type=FeatureType.VISUAL,
                                                                       shape=(3, H, W)) for c in CAMS}}
    cfg.output_features = {"action": PolicyFeature(type=FeatureType.ACTION, shape=(9,))}
    stats = {k: {"min": np.full(9, -1.0), "max": np.full(9, 1.0)} for k in ("observation.state", "action")}
    for c in CAMS:
        stats[f"observation.images.{c}"] = {"mean": np.full((3, 1, 1), 0.5), "std": np.full((3, 1, 1), 0.2)}
    path = tmp_path_factory.mktemp("ckpt")
    DiffusionPolicy(cfg).save_pretrained(path)
    pre, post = make_pre_post_processors(cfg, dataset_stats=stats)
    pre.save_pretrained(path)
    post.save_pretrained(path)
    return path


def make_server(checkpoint, skip_similar=False):
    from lerobot.async_inference.configs import PolicyServerConfig
    from lerobot.async_inference.helpers import RemotePolicyConfig

    ps = policy_server._patch(skip_similar)
    server = ps.PolicyServer(PolicyServerConfig(host="127.0.0.1", port=8099))   # never bound
    features = {"observation.state": {"dtype": "float32", "shape": (9,), "names": POLICY_STATE_COLUMNS},
                **{f"observation.images.{c}": {"dtype": "image", "shape": (H, W, 3),
                                               "names": ["height", "width", "channels"]} for c in CAMS}}
    request = types.SimpleNamespace(data=pickle.dumps(
        RemotePolicyConfig("diffusion", str(checkpoint), features, 8, "cpu")))
    server.SendPolicyInstructions(request, types.SimpleNamespace(peer=lambda: "test"))
    return server


def observation(i, t, must_go=False, timestep=None):
    from lerobot.async_inference.helpers import TimedObservation
    raw = {n: 0.01 * i for n in POLICY_STATE_COLUMNS}
    raw.update({c: np.full((H, W, 3), i % 255, np.uint8) for c in CAMS})
    raw["task"] = "test"
    return TimedObservation(timestamp=t, timestep=i if timestep is None else timestep,
                            observation=raw, must_go=must_go)


def test_n_obs_steps_2_returns_a_chunk_from_two_frames(checkpoint, caplog):
    server = make_server(checkpoint)
    t0 = 1000.0
    for i in range(4):
        server._enqueue_observation(observation(i, t0 + i / 30, must_go=(i == 3), timestep=0))
    obs = None
    while not server.observation_queue.empty():
        obs = server.observation_queue.get_nowait()
    with caplog.at_level("INFO"):
        chunk = server._predict_action_chunk(obs)
    assert len(chunk) == 8 and chunk[0].get_action().shape == (9,)
    assert "the earlier ones 33 ms before" in caplog.text


def test_similar_observations_are_not_filtered(checkpoint):
    server = make_server(checkpoint)
    server.last_processed_obs = observation(0, 1000.0, timestep=0)
    # same joints, a new timestep: the stock server drops it (L2 < 1.0), this one runs it
    assert server._obs_sanity_checks(observation(0, 1000.5, timestep=5), server.last_processed_obs)
    stock = make_server(checkpoint, skip_similar=True)
    stock.last_processed_obs = observation(0, 1000.0, timestep=0)
    assert not stock._obs_sanity_checks(observation(0, 1000.5, timestep=5), stock.last_processed_obs)
