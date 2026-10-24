"""Deterministic environment preserving TStarBot1 observation semantics."""

from typing import Literal

import gymnasium as gym
import numpy as np


ACTION_NAMES = (
    "do_nothing",
    "build_spawning_pool",
    "produce_zergling",
    "attack",
)
FEATURE_NAMES = (
    "minerals",
    "gas",
    "food_used",
    "game_progress",
    "mask_do_nothing",
    "mask_build_spawning_pool",
    "mask_produce_zergling",
    "mask_attack",
)


class SyntheticMacroEnv(gym.Env):
    """Synthetic macro-action env for interface tests, not RL benchmarks."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        *,
        masked: bool = True,
        api: Literal["legacy", "modern"] = "modern",
        ending: Literal["terminated", "truncated"] = "terminated",
        horizon: int = 3,
    ) -> None:
        if horizon < 1:
            raise ValueError("horizon must be positive")
        self.masked = masked
        self.api = api
        self.ending = ending
        self.horizon = horizon
        self.action_names = ACTION_NAMES
        self.action_space = gym.spaces.Discrete(len(ACTION_NAMES))
        feature_space = gym.spaces.Box(
            0.0, np.inf, shape=(len(FEATURE_NAMES),), dtype=np.float32
        )
        if masked:
            self.observation_space = gym.spaces.Tuple(
                (
                    feature_space,
                    gym.spaces.Box(
                        0.0,
                        1.0,
                        shape=(len(ACTION_NAMES),),
                        dtype=np.float32,
                    ),
                )
            )
        else:
            self.observation_space = feature_space
        self._step_id = 0

    def _mask(self) -> np.ndarray:
        masks = (
            (1.0, 0.0, 1.0, 0.0),
            (1.0, 1.0, 1.0, 0.0),
            (1.0, 1.0, 1.0, 1.0),
        )
        return np.asarray(masks[min(self._step_id, 2)], dtype=np.float32)

    def _observation(self):
        mask = self._mask()
        prefix = np.asarray(
            [50.0 + self._step_id, 0.0, self._step_id, self._step_id / self.horizon],
            dtype=np.float32,
        )
        features = np.concatenate((prefix, mask)).astype(np.float32, copy=False)
        return (features, mask.copy()) if self.masked else features

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._step_id = 0
        observation = self._observation()
        if self.api == "legacy":
            return observation
        return observation, {"seed": seed}

    def step(self, action):
        if not self.action_space.contains(action):
            raise ValueError(f"action {action!r} is outside the action space")
        if self.masked and self._mask()[int(action)] == 0:
            raise ValueError(f"action {action} is masked")

        self._step_id += 1
        at_horizon = self._step_id >= self.horizon
        terminated = at_horizon and self.ending == "terminated"
        truncated = at_horizon and self.ending == "truncated"
        reward = 1.0 if terminated else 0.0
        info = {"step_id": self._step_id}
        if self.api == "legacy":
            if truncated:
                info["TimeLimit.truncated"] = True
            return self._observation(), reward, terminated or truncated, info
        return self._observation(), reward, terminated, truncated, info
