"""stretch4_policy_server: LeRobot's async policy server, with three fixes Stretch 4 needs.

    stretch4_policy_server --host=0.0.0.0 --port=8080       # on the GPU machine / Jetson

Same flags, protocol and robot side as `python -m lerobot.async_inference.policy_server` (the
robot runs the stock `python -m lerobot.async_inference.robot_client --robot.type=stretch4`).

1. Observation history. LeRobot 0.6's server hands the policy one observation per chunk.
   Policies that look at several past frames (n_obs_steps > 1: diffusion 2, multi_task_dit 2,
   VQ-BeT 5) fail an assert on it, logged as an empty "Error in StreamActions:", and the robot
   never moves. This server keeps the observations it receives and, for such a policy, feeds the
   ones 1/fps apart (as in training) through the policy's own observation queues, the same way
   its select_action does. Policies with n_obs_steps == 1 run exactly as stock.

2. The similarity filter. The stock server skips an observation whose joint state is within L2
   distance 1.0 of the last one it ran, which suits joints in degrees. Stretch reports metres and
   radians, so nearly every early request was dropped and the robot emptied its action queue
   before each chunk (stop-and-go). Off by default (--skip_similar_observations=true restores it):
   the robot client only sends once its queue is below --chunk_size_threshold anyway.

3. LoRA checkpoints. A checkpoint trained with --policy.use_peft=true holds only the adapter
   (adapter_model.safetensors, a few MB). The stock server's from_pretrained finds no
   model.safetensors there, logs "Returning model without loading pretrained weights" and runs a
   randomly initialised policy: the joints move at random. This server loads the base policy the
   adapter names (adapter_config.json), with the checkpoint's own config, and merges the adapter in.

Errors during inference are logged with their traceback (stock logs only the message).
"""

import collections
import sys
import threading
import traceback

STOCK_SKIP_FLAG = "--skip_similar_observations"


def _adapter_loading(policy_class):
    """policy_class, with a from_pretrained that also loads LoRA (PEFT) checkpoints."""
    from pathlib import Path

    class Loader:
        @staticmethod
        def from_pretrained(path, **kwargs):
            if not (Path(path) / "adapter_config.json").is_file():
                return policy_class.from_pretrained(path, **kwargs)
            from lerobot.configs.policies import PreTrainedConfig
            from peft import PeftConfig, PeftModel

            base_path = PeftConfig.from_pretrained(path).base_model_name_or_path
            config = PreTrainedConfig.from_pretrained(path)
            base = policy_class.from_pretrained(base_path, config=config, **kwargs)
            return PeftModel.from_pretrained(base, path).merge_and_unload()

    return Loader


def _patch(skip_similar):
    import torch
    from lerobot.async_inference import policy_server as ps
    from lerobot.async_inference.helpers import raw_observation_to_observation
    from lerobot.policies.utils import populate_queues
    from lerobot.utils.constants import ACTION, OBS_IMAGES

    server = ps.PolicyServer
    # The stock methods, kept from the first patch so patching again never wraps a wrapper.
    if not hasattr(server, "_stretch4_stock"):
        server._stretch4_stock = (server._enqueue_observation, server._obs_sanity_checks,
                                  server._predict_action_chunk, server._get_action_chunk)
    stock_enqueue, stock_sanity, stock_predict, stock_get_chunk = server._stretch4_stock
    if not hasattr(ps, "_stretch4_stock_get_policy_class"):
        ps._stretch4_stock_get_policy_class = ps.get_policy_class
    ps.get_policy_class = lambda name: _adapter_loading(ps._stretch4_stock_get_policy_class(name))
    lock = threading.Lock()

    def _enqueue_observation(self, obs):
        with lock:
            if not hasattr(self, "_obs_history"):
                self._obs_history = collections.deque(maxlen=150)   # ~5 s at 30 fps
            self._obs_history.append(obs)
        return stock_enqueue(self, obs)

    def _obs_sanity_checks(self, obs, previous_obs):
        if skip_similar:
            return stock_sanity(self, obs, previous_obs)
        with self._predicted_timesteps_lock:
            if obs.get_timestep() in self._predicted_timesteps:
                self.logger.debug(f"Skipping observation #{obs.get_timestep()} - Timestep predicted already!")
                return False
        return True

    def _predict_action_chunk(self, observation_t):
        self._current_timed_obs = observation_t
        try:
            return stock_predict(self, observation_t)
        except Exception:
            self.logger.error("Inference failed:\n" + traceback.format_exc())
            raise

    def past_observations(self, current, n):
        """The n-1 received observations closest to current's timestamp - k/fps (k = n-1..1);
        current itself when none is within 3 frames (the start of a run), as select_action
        pads its queues."""
        dt = 1.0 / self.config.fps
        t = current.get_timestamp()
        with lock:
            older = [h for h in getattr(self, "_obs_history", []) if h.get_timestamp() < t - 1e-4]
        past = []
        for k in range(n - 1, 0, -1):
            target = t - k * dt
            best = min(older, key=lambda h: abs(h.get_timestamp() - target), default=None)
            past.append(best if best is not None and abs(best.get_timestamp() - target) <= 3 * dt
                        else current)
        return past

    def _get_action_chunk(self, observation):
        policy = self.policy
        n = int(getattr(policy.config, "n_obs_steps", 1) or 1)
        if n <= 1 or not hasattr(policy, "_queues"):
            return stock_get_chunk(self, observation)

        current = self._current_timed_obs
        past = past_observations(self, current, n)
        batches = [self.preprocessor(raw_observation_to_observation(
            p.get_observation(), self.lerobot_features, self.policy_image_features)) for p in past]
        batches.append(observation)

        # select_action's own steps, for n observations at once: images stacked into one key,
        # every key through the policy's queues, then predict_action_chunk stacks the queues.
        policy.reset()
        image_keys = list(getattr(policy.config, "image_features", None) or [])
        for b in batches:
            b = {k: v for k, v in b.items() if k != ACTION}
            if image_keys:
                b[OBS_IMAGES] = torch.stack([b[k] for k in image_keys], dim=-4)
            policy._queues = populate_queues(policy._queues, b)
        gaps = ", ".join(f"{(current.get_timestamp() - p.get_timestamp()) * 1000:.0f} ms" for p in past)
        self.logger.info(f"observation history: {n} frames, the earlier ones {gaps} before")
        chunk = policy.predict_action_chunk(b)
        if chunk.ndim != 3:
            chunk = chunk.unsqueeze(0)
        return chunk[:, : self.actions_per_chunk, :]

    server._enqueue_observation = _enqueue_observation
    server._obs_sanity_checks = _obs_sanity_checks
    server._predict_action_chunk = _predict_action_chunk
    server._get_action_chunk = _get_action_chunk
    return ps


def main():
    skip_similar = False
    argv = []
    for arg in sys.argv[1:]:
        if arg.startswith(STOCK_SKIP_FLAG):
            skip_similar = arg.partition("=")[2].lower() in ("", "1", "true", "yes")
        else:
            argv.append(arg)
    sys.argv = [sys.argv[0], *argv]
    _patch(skip_similar).serve()


if __name__ == "__main__":
    main()
