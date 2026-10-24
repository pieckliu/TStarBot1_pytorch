"""PyTorch PPO core and inference agent shared by all execution modes."""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from sc2learner.agents.ppo_policies import LstmPolicy, MlpPolicy


CHECKPOINT_FORMAT_VERSION = 1


def _feature_space(observation_space: Any, action_dim: int):
    spaces = getattr(observation_space, "spaces", None)
    if spaces is None:
        return observation_space, False
    spaces = tuple(spaces)
    if len(spaces) == 2 and tuple(spaces[-1].shape) == (action_dim,):
        return spaces[0], True
    raise ValueError(
        "MLP PPO supports a single feature array or (features, action_mask); "
        "spatial tuple observations are not part of Milestones 2-3"
    )


def _stack_observations(observations):
    if isinstance(observations[0], tuple):
        return tuple(np.stack(parts) for parts in zip(*observations))
    return np.stack(observations)


def compute_gae(
    rewards,
    values,
    next_values,
    terminated,
    truncated,
    gamma: float,
    gae_lambda: float,
):
    """Compute GAE without leaking traces across episode boundaries.

    True terminals suppress value bootstrapping. Time-limit truncations may
    bootstrap from their final observation, but both boundary types stop the
    recursive GAE trace before a reset observation.
    """
    rewards = np.asarray(rewards, dtype=np.float32)
    values = np.asarray(values, dtype=np.float32)
    next_values = np.asarray(next_values, dtype=np.float32)
    terminated = np.asarray(terminated, dtype=np.bool_)
    truncated = np.asarray(truncated, dtype=np.bool_)
    expected_shape = rewards.shape
    for name, array in (
        ("values", values),
        ("next_values", next_values),
        ("terminated", terminated),
        ("truncated", truncated),
    ):
        if array.shape != expected_shape:
            raise ValueError(f"{name} shape {array.shape} != rewards shape {expected_shape}")
    advantages = np.zeros_like(rewards)
    next_advantage = 0.0
    for index in range(len(rewards) - 1, -1, -1):
        bootstrap = 0.0 if terminated[index] else 1.0
        continuation = 0.0 if terminated[index] or truncated[index] else 1.0
        delta = rewards[index] + gamma * next_values[index] * bootstrap - values[index]
        next_advantage = delta + gamma * gae_lambda * continuation * next_advantage
        advantages[index] = next_advantage
    return advantages, advantages + values


def ppo_loss(
    policy,
    observations: torch.Tensor,
    actions: torch.Tensor,
    returns: torch.Tensor,
    old_values: torch.Tensor,
    old_negative_log_probabilities: torch.Tensor,
    *,
    action_mask: torch.Tensor | None,
    clip_range: float,
    entropy_coefficient: float,
    value_coefficient: float,
    value_clip: bool,
    states: torch.Tensor | None = None,
    episode_starts: torch.Tensor | None = None,
):
    if getattr(policy, "is_recurrent", False):
        if states is None:
            raise ValueError("recurrent PPO updates require initial states")
        values, negative_log_probabilities, entropy_values, _, _ = policy.evaluate_actions(
            observations, actions, states, episode_starts, action_mask
        )
    else:
        values, negative_log_probabilities, entropy_values, _ = policy.evaluate_actions(
            observations, actions, action_mask
        )
    advantages = returns - old_values
    advantages = (advantages - advantages.mean()) / (
        advantages.std(unbiased=False) + 1e-8
    )
    ratio = torch.exp(old_negative_log_probabilities - negative_log_probabilities)
    policy_losses = -advantages * ratio
    clipped_policy_losses = -advantages * torch.clamp(
        ratio, 1.0 - clip_range, 1.0 + clip_range
    )
    policy_loss_value = torch.maximum(policy_losses, clipped_policy_losses).mean()

    value_losses = (values - returns).square()
    if value_clip:
        clipped_values = old_values + torch.clamp(
            values - old_values, -clip_range, clip_range
        )
        clipped_value_losses = (clipped_values - returns).square()
        value_loss_value = 0.5 * torch.maximum(
            value_losses, clipped_value_losses
        ).mean()
    else:
        value_loss_value = 0.5 * value_losses.mean()

    entropy = entropy_values.mean()
    approximate_kl = 0.5 * (
        negative_log_probabilities - old_negative_log_probabilities
    ).square().mean()
    clip_fraction = ((ratio - 1.0).abs() > clip_range).float().mean()
    total_loss = (
        policy_loss_value
        - entropy_coefficient * entropy
        + value_coefficient * value_loss_value
    )
    return {
        "total_loss": total_loss,
        "policy_loss": policy_loss_value,
        "value_loss": value_loss_value,
        "entropy": entropy,
        "approx_kl": approximate_kl,
        "clip_fraction": clip_fraction,
        "ratio": ratio,
        "advantages": advantages,
    }


class Model:
    """PyTorch PPO model shared by MLP and recurrent execution."""

    loss_names = [
        "policy_loss",
        "value_loss",
        "policy_entropy",
        "approxkl",
        "clipfrac",
    ]

    def __init__(
        self,
        *,
        policy=MlpPolicy,
        ob_space,
        ac_space,
        nbatch_act=None,
        nbatch_train=None,
        unroll_length=1,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        scope_name="model",
        value_clip=False,
        learning_rate=1e-5,
        device="cpu",
    ):
        del nbatch_train, unroll_length, scope_name
        if not isinstance(policy, type) or not issubclass(policy, (MlpPolicy, LstmPolicy)):
            raise TypeError("policy must be MlpPolicy or LstmPolicy")
        self.action_dim = int(ac_space.n)
        feature_space, self.masked = _feature_space(ob_space, self.action_dim)
        self.observation_shape = tuple(feature_space.shape)
        self.observation_dim = int(np.prod(self.observation_shape))
        self.device = torch.device(device)
        self.entropy_coefficient = float(ent_coef)
        self.value_coefficient = float(vf_coef)
        self.max_grad_norm = max_grad_norm
        self.value_clip = bool(value_clip)
        self.policy = policy(self.observation_dim, self.action_dim).to(self.device)
        self.is_recurrent = self.policy.is_recurrent
        self.optimizer = torch.optim.Adam(
            self.policy.parameters(), lr=learning_rate, eps=1e-5
        )
        self.initial_state = (
            self.policy.initial_state(int(nbatch_act or 1), self.device).cpu().numpy()
            if self.is_recurrent else None
        )
        self.last_train_metrics = None

    def _tensor_observation(self, observation, *, require_batch=False):
        action_mask = None
        features = observation
        if isinstance(observation, tuple):
            if len(observation) != 2:
                raise ValueError("MLP masked observations must be (features, action_mask)")
            features, action_mask = observation
        features = torch.as_tensor(features, dtype=torch.float32, device=self.device)
        if features.ndim == len(self.observation_shape):
            features = features.unsqueeze(0)
        if require_batch and features.ndim != len(self.observation_shape) + 1:
            raise ValueError("training observations must contain one batch dimension")
        if action_mask is not None:
            action_mask = torch.as_tensor(
                action_mask, dtype=torch.float32, device=self.device
            )
            if action_mask.ndim == 1:
                action_mask = action_mask.unsqueeze(0)
        elif self.masked:
            raise ValueError("this model requires an action mask")
        return features, action_mask

    def step(self, observation, state=None, done=None, deterministic=False):
        self.policy.eval()
        features, action_mask = self._tensor_observation(observation)
        with torch.no_grad():
            if self.is_recurrent:
                if state is None:
                    state = self.policy.initial_state(features.shape[0], self.device)
                else:
                    state = torch.as_tensor(state, dtype=torch.float32, device=self.device)
                starts = torch.as_tensor(
                    np.zeros(features.shape[0], dtype=np.bool_) if done is None else done,
                    dtype=torch.bool, device=self.device,
                ).reshape(-1)
                actions, values, negative_log_probabilities, next_state = self.policy.act(
                    features, state, starts, action_mask, deterministic=deterministic
                )
            else:
                actions, values, negative_log_probabilities = self.policy.act(
                    features, action_mask, deterministic=deterministic
                )
                next_state = None
        return (
            actions.cpu().numpy(),
            values.cpu().numpy(),
            None if next_state is None else next_state.cpu().numpy(),
            negative_log_probabilities.cpu().numpy(),
        )

    def value(self, observation, state=None, done=None):
        self.policy.eval()
        features, _ = self._tensor_observation(observation)
        with torch.no_grad():
            if self.is_recurrent:
                if state is None:
                    state = self.policy.initial_state(features.shape[0], self.device)
                else:
                    state = torch.as_tensor(state, dtype=torch.float32, device=self.device)
                starts = torch.as_tensor(
                    np.zeros(features.shape[0], dtype=np.bool_) if done is None else done,
                    dtype=torch.bool, device=self.device,
                ).reshape(-1)
                values = self.policy.forward_sequence(features, state, starts)[0].value
            else:
                values = self.policy(features).value
        return values.cpu().numpy()

    def update(
        self,
        learning_rate,
        clip_range,
        observations,
        returns,
        actions,
        old_values,
        old_negative_log_probabilities,
        dones=None,
        states=None,
    ):
        self.policy.train()
        features, action_mask = self._tensor_observation(
            observations, require_batch=True
        )
        returns = torch.as_tensor(returns, dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(actions, dtype=torch.long, device=self.device)
        old_values = torch.as_tensor(
            old_values, dtype=torch.float32, device=self.device
        )
        old_negative_log_probabilities = torch.as_tensor(
            old_negative_log_probabilities, dtype=torch.float32, device=self.device
        )
        recurrent_states = None if states is None else torch.as_tensor(
            states, dtype=torch.float32, device=self.device
        )
        episode_starts = None if dones is None else torch.as_tensor(
            dones, dtype=torch.bool, device=self.device
        )
        for parameter_group in self.optimizer.param_groups:
            parameter_group["lr"] = float(learning_rate)
        losses = ppo_loss(
            self.policy,
            features,
            actions,
            returns,
            old_values,
            old_negative_log_probabilities,
            action_mask=action_mask,
            clip_range=float(clip_range),
            entropy_coefficient=self.entropy_coefficient,
            value_coefficient=self.value_coefficient,
            value_clip=self.value_clip,
            states=recurrent_states,
            episode_starts=episode_starts,
        )
        self.optimizer.zero_grad(set_to_none=True)
        losses["total_loss"].backward()
        if self.max_grad_norm is None:
            squared_norm = sum(
                parameter.grad.detach().square().sum()
                for parameter in self.policy.parameters()
                if parameter.grad is not None
            )
            gradient_norm = squared_norm.sqrt()
        else:
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                self.policy.parameters(), self.max_grad_norm
            )
        self.optimizer.step()
        metrics = {
            name: float(value.detach().cpu())
            for name, value in losses.items()
            if name not in {"ratio", "advantages"}
        }
        metrics["gradient_norm"] = float(gradient_norm.detach().cpu())
        self.last_train_metrics = metrics
        return metrics

    def train(
        self,
        lr,
        cliprange,
        obs,
        returns,
        dones,
        actions,
        values,
        neglogpacs,
        states=None,
    ):
        metrics = self.update(
            lr, cliprange, obs, returns, actions, values, neglogpacs, dones, states
        )
        return [
            metrics["policy_loss"],
            metrics["value_loss"],
            metrics["entropy"],
            metrics["approx_kl"],
            metrics["clip_fraction"],
        ]

    def read_params(self):
        return OrderedDict(
            (name, value.detach().cpu().clone())
            for name, value in self.policy.state_dict().items()
        )

    def load_params(self, loaded_params: Mapping[str, Any]):
        if not isinstance(loaded_params, Mapping):
            raise TypeError("PyTorch parameter snapshots must be named mappings")
        converted = OrderedDict(
            (name, torch.as_tensor(value, device=self.device))
            for name, value in loaded_params.items()
        )
        self.policy.load_state_dict(converted, strict=True)

    def save(self, save_path):
        checkpoint = {
            "format_version": CHECKPOINT_FORMAT_VERSION,
            "policy_class": type(self.policy).__name__,
            "observation_shape": self.observation_shape,
            "action_dim": self.action_dim,
            "masked": self.masked,
            "value_clip": self.value_clip,
            "model_state_dict": self.policy.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }
        torch.save(checkpoint, Path(save_path))

    def load(self, load_path, *, load_optimizer=False):
        checkpoint = torch.load(
            Path(load_path), map_location=self.device, weights_only=False
        )
        if checkpoint.get("format_version") != CHECKPOINT_FORMAT_VERSION:
            raise ValueError("unsupported PPO checkpoint format")
        if checkpoint.get("policy_class") != type(self.policy).__name__:
            raise ValueError("checkpoint policy class does not match the model")
        expected = (
            self.observation_shape,
            self.action_dim,
            self.masked,
            self.value_clip,
        )
        actual = (
            tuple(checkpoint["observation_shape"]),
            int(checkpoint["action_dim"]),
            bool(checkpoint["masked"]),
            bool(checkpoint["value_clip"]),
        )
        if actual != expected:
            raise ValueError(f"checkpoint schema {actual} does not match model {expected}")
        self.policy.load_state_dict(checkpoint["model_state_dict"], strict=True)
        if load_optimizer:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])


class PPOAgent:
    def __init__(
        self,
        env,
        policy=MlpPolicy,
        model_path=None,
        *,
        device="cpu",
        deterministic=False,
    ):
        self._model = Model(
            policy=policy,
            ob_space=env.observation_space,
            ac_space=env.action_space,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            device=device,
        )
        if model_path is not None:
            self._model.load(model_path)
        self._deterministic = deterministic
        self.reset()

    def act(self, observation):
        action, _, self._state, _ = self._model.step(
            observation, self._state, self._episode_start,
            deterministic=self._deterministic
        )
        self._episode_start = np.asarray([False])
        return int(action[0])

    def reset(self):
        self._state = None
        self._episode_start = np.asarray([True])


class PPOActor:
    """Compatibility constructor for the ZeroMQ actor implementation."""

    def __new__(cls, *args, **kwargs):
        from sc2learner.agents.ppo_distributed import PPOActor as Implementation
        return Implementation(*args, **kwargs)


class PPOLearner:
    """Compatibility constructor for the ZeroMQ learner implementation."""

    def __new__(cls, *args, **kwargs):
        from sc2learner.agents.ppo_distributed import PPOLearner as Implementation
        return Implementation(*args, **kwargs)


class PPOSelfplayActor:
    """Compatibility constructor for the PyTorch self-play actor."""

    def __new__(cls, *args, **kwargs):
        from sc2learner.agents.ppo_distributed import (
            PPOSelfplayActor as Implementation,
        )
        return Implementation(*args, **kwargs)


def constfn(value):
    return lambda _: value


def safemean(values):
    return np.nan if len(values) == 0 else np.mean(values)


def transform_tuple(value, transformer):
    if isinstance(value, tuple):
        return tuple(transformer(part) for part in value)
    return transformer(value)
