"""Synthetic macro-action environment for local PPO development."""

from __future__ import annotations

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
    """Deterministic, SC2-free environment with TStarBot1 tuple semantics."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        *,
        masked: bool = True,
        api: Literal["legacy", "modern"] = "modern",
        ending: Literal["terminated", "truncated"] = "terminated",
        horizon: int = 8,
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

    def _target_action(self) -> int:
        return 2 if self._step_id % 2 == 0 else 1

    def _mask(self) -> np.ndarray:
        mask = np.zeros(len(ACTION_NAMES), dtype=np.float32)
        mask[0] = 1.0
        mask[self._target_action()] = 1.0
        if self._step_id >= self.horizon // 2:
            mask[3] = 1.0
        return mask

    def _observation(self):
        mask = self._mask()
        prefix = np.asarray(
            [50.0 + self._step_id, 25.0, self._step_id, self._step_id / self.horizon],
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
        reward = 1.0 if int(action) == self._target_action() else 0.0
        self._step_id += 1
        at_horizon = self._step_id >= self.horizon
        terminated = at_horizon and self.ending == "terminated"
        truncated = at_horizon and self.ending == "truncated"
        info = {"step_id": self._step_id, "target_action": self._target_action()}
        if self.api == "legacy":
            if truncated:
                info["TimeLimit.truncated"] = True
            return self._observation(), reward, terminated or truncated, info
        return self._observation(), reward, terminated, truncated, info
