"""SC2-free local rollout, training, and evaluation for PyTorch PPO."""

from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np

from sc2learner.agents.ppo_agent import Model, _stack_observations, compute_gae
from sc2learner.envs.api_compat import normalize_reset, normalize_step


@dataclass(frozen=True)
class RolloutBatch:
    observations: object
    returns: np.ndarray
    dones: np.ndarray
    actions: np.ndarray
    values: np.ndarray
    negative_log_probabilities: np.ndarray
    advantages: np.ndarray
    rewards: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray
    episode_returns: tuple[float, ...]
    initial_state: np.ndarray | None = None


class RolloutCollector:
    """Stateful collector so distributed unrolls do not reset the environment."""

    def __init__(self, env, model: Model, *, gamma, gae_lambda, seed=None):
        self.env = env
        self.model = model
        self.gamma = float(gamma)
        self.gae_lambda = float(gae_lambda)
        self.observation = normalize_reset(env.reset(seed=seed)).observation
        self.state = (
            model.policy.initial_state(1, model.device).cpu().numpy()
            if model.is_recurrent else None
        )
        self.episode_start = True
        self.current_episode_return = 0.0

    def collect(self, n_steps: int) -> RolloutBatch:
        if n_steps < 1:
            raise ValueError("n_steps must be positive")
        observation = self.observation
        state = self.state
        episode_start = self.episode_start
        current_episode_return = self.current_episode_return
        observations = []
        actions = []
        rewards = []
        values = []
        next_values = []
        negative_log_probabilities = []
        terminated = []
        truncated = []
        episode_returns = []
        observation_starts = []
        initial_state = None if state is None else state.copy()

        for _ in range(n_steps):
            observation_starts.append(episode_start)
            action, value, next_state, negative_log_probability = self.model.step(
                observation, state, np.asarray([episode_start])
            )
            observations.append(observation)
            actions.append(int(action[0]))
            values.append(float(value[0]))
            negative_log_probabilities.append(float(negative_log_probability[0]))

            transition = normalize_step(self.env.step(int(action[0])))
            current_episode_return += transition.reward
            rewards.append(transition.reward)
            terminated.append(transition.terminated)
            truncated.append(transition.truncated)
            next_values.append(float(self.model.value(
                transition.observation, next_state, np.asarray([False])
            )[0]))
            state = next_state
            observation = transition.observation
            episode_start = transition.done

            if transition.done:
                episode_returns.append(current_episode_return)
                current_episode_return = 0.0
                observation = normalize_reset(self.env.reset()).observation
                state = (
                    self.model.policy.initial_state(1, self.model.device).cpu().numpy()
                    if self.model.is_recurrent else None
                )

        advantages, returns = compute_gae(
            rewards,
            values,
            next_values,
            terminated,
            truncated,
            self.gamma,
            self.gae_lambda,
        )
        self.observation = observation
        self.state = state
        self.episode_start = episode_start
        self.current_episode_return = current_episode_return
        return RolloutBatch(
            observations=_stack_observations(observations),
            returns=returns,
            dones=np.asarray(observation_starts, dtype=np.bool_),
            actions=np.asarray(actions, dtype=np.int64),
            values=np.asarray(values, dtype=np.float32),
            negative_log_probabilities=np.asarray(
                negative_log_probabilities, dtype=np.float32
            ),
            advantages=advantages,
            rewards=np.asarray(rewards, dtype=np.float32),
            terminated=np.asarray(terminated, dtype=np.bool_),
            truncated=np.asarray(truncated, dtype=np.bool_),
            episode_returns=tuple(episode_returns),
            initial_state=initial_state,
        )


def collect_rollout(env, model: Model, n_steps: int, *, gamma: float,
                    gae_lambda: float, seed: int | None = None) -> RolloutBatch:
    return RolloutCollector(
        env, model, gamma=gamma, gae_lambda=gae_lambda, seed=seed
    ).collect(n_steps)


def evaluate_policy(env, model: Model, episodes: int, *, seed: int = 0):
    if episodes < 1:
        raise ValueError("episodes must be positive")
    returns = []
    action_sequences = []
    for episode in range(episodes):
        observation = normalize_reset(env.reset(seed=seed + episode)).observation
        episode_return = 0.0
        actions = []
        state = None
        episode_start = True
        while True:
            action, _, state, _ = model.step(
                observation, state, np.asarray([episode_start]), deterministic=True
            )
            episode_start = False
            selected = int(action[0])
            actions.append(selected)
            transition = normalize_step(env.step(selected))
            episode_return += transition.reward
            observation = transition.observation
            if transition.done:
                break
        returns.append(episode_return)
        action_sequences.append(tuple(actions))
    return {
        "mean_return": float(np.mean(returns)),
        "episode_returns": tuple(returns),
        "action_sequences": tuple(action_sequences),
    }


class LocalPPOTrainer:
    def __init__(
        self,
        env,
        model: Model,
        *,
        rollout_steps=128,
        gamma=0.998,
        gae_lambda=0.95,
        learning_rate=1e-5,
        clip_range=0.1,
    ):
        self.env = env
        self.model = model
        self.rollout_steps = int(rollout_steps)
        self.gamma = float(gamma)
        self.gae_lambda = float(gae_lambda)
        self.learning_rate = float(learning_rate)
        self.clip_range = float(clip_range)

    def update(self, *, seed=None):
        start = time.perf_counter()
        rollout = collect_rollout(
            self.env,
            self.model,
            self.rollout_steps,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            seed=seed,
        )
        metrics = self.model.update(
            self.learning_rate,
            self.clip_range,
            rollout.observations,
            rollout.returns,
            rollout.actions,
            rollout.values,
            rollout.negative_log_probabilities,
            rollout.dones,
            rollout.initial_state,
        )
        elapsed = time.perf_counter() - start
        metrics.update(
            {
                "rollout_mean_reward": float(rollout.rewards.mean()),
                "episode_return": (
                    float(np.mean(rollout.episode_returns))
                    if rollout.episode_returns
                    else float("nan")
                ),
                "steps_per_second": self.rollout_steps / elapsed,
            }
        )
        return metrics, rollout
