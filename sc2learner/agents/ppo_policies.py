"""PyTorch policy networks for TStarBot1 PPO."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.distributions import Categorical


@dataclass(frozen=True)
class PolicyOutput:
    logits: torch.Tensor
    value: torch.Tensor


def _orthogonal_linear(linear: nn.Linear, gain: float) -> None:
    nn.init.orthogonal_(linear.weight, gain=gain)
    nn.init.zeros_(linear.bias)


def masked_logits(logits: torch.Tensor, action_mask: torch.Tensor | None) -> torch.Tensor:
    """Apply a valid-action mask and reject ambiguous all-masked rows."""
    if action_mask is None:
        return logits
    raw_mask = action_mask.to(device=logits.device)
    if raw_mask.shape != logits.shape:
        raise ValueError(
            f"action mask shape {tuple(raw_mask.shape)} does not match logits "
            f"shape {tuple(logits.shape)}"
        )
    if raw_mask.dtype != torch.bool:
        if not torch.isfinite(raw_mask).all() or not torch.all(
            (raw_mask == 0) | (raw_mask == 1)
        ):
            raise ValueError("action masks must contain only finite 0/1 values")
    mask = raw_mask.to(dtype=torch.bool)
    if not torch.all(mask.any(dim=-1)):
        raise ValueError("each action-mask row must contain at least one valid action")
    return logits.masked_fill(~mask, -torch.inf)


class MlpPolicy(nn.Module):
    """Faithful PyTorch version of the original independent PPO MLP towers."""

    def __init__(self, observation_dim: int, action_dim: int) -> None:
        super().__init__()
        self.is_recurrent = False
        if observation_dim < 1 or action_dim < 1:
            raise ValueError("observation_dim and action_dim must be positive")
        self.observation_dim = int(observation_dim)
        self.action_dim = int(action_dim)

        self.policy_body = nn.Sequential(
            nn.Linear(observation_dim, 512),
            nn.Tanh(),
            nn.Linear(512, 512),
            nn.Tanh(),
            nn.Linear(512, 512),
            nn.Tanh(),
        )
        self.value_body = nn.Sequential(
            nn.Linear(observation_dim, 512),
            nn.Tanh(),
            nn.Linear(512, 512),
            nn.Tanh(),
            nn.Linear(512, 512),
            nn.Tanh(),
        )
        self.policy_head = nn.Linear(512, action_dim)
        self.value_head = nn.Linear(512, 1)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        hidden_gain = 2.0**0.5
        for body in (self.policy_body, self.value_body):
            for module in body:
                if isinstance(module, nn.Linear):
                    _orthogonal_linear(module, hidden_gain)
        _orthogonal_linear(self.policy_head, 0.01)
        _orthogonal_linear(self.value_head, 1.0)

    def forward(self, observation: torch.Tensor) -> PolicyOutput:
        if observation.ndim < 2:
            raise ValueError("observations must include a batch dimension")
        flat = observation.reshape(observation.shape[0], -1)
        if flat.shape[1] != self.observation_dim:
            raise ValueError(
                f"expected flattened observation size {self.observation_dim}, "
                f"got {flat.shape[1]}"
            )
        logits = self.policy_head(self.policy_body(flat))
        value = self.value_head(self.value_body(flat)).squeeze(-1)
        return PolicyOutput(logits=logits, value=value)

    def distribution(
        self, logits: torch.Tensor, action_mask: torch.Tensor | None = None
    ) -> Categorical:
        return Categorical(logits=masked_logits(logits, action_mask))

    def act(
        self,
        observation: torch.Tensor,
        action_mask: torch.Tensor | None = None,
        *,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        output = self(observation)
        distribution = self.distribution(output.logits, action_mask)
        action = (
            distribution.logits.argmax(dim=-1)
            if deterministic
            else distribution.sample()
        )
        negative_log_probability = -distribution.log_prob(action)
        return action, output.value, negative_log_probability

    def evaluate_actions(
        self,
        observation: torch.Tensor,
        actions: torch.Tensor,
        action_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        output = self(observation)
        distribution = self.distribution(output.logits, action_mask)
        negative_log_probability = -distribution.log_prob(actions)
        return (
            output.value,
            negative_log_probability,
            distribution.entropy(),
            distribution.logits,
        )


class TStarLSTMCell(nn.Module):
    """The original TStarBot LSTM equations with explicit ``[c, h]`` state."""

    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.input_kernel = nn.Parameter(torch.empty(input_size, 4 * hidden_size))
        self.hidden_kernel = nn.Parameter(torch.empty(hidden_size, 4 * hidden_size))
        self.bias = nn.Parameter(torch.zeros(4 * hidden_size))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.orthogonal_(self.input_kernel)
        nn.init.orthogonal_(self.hidden_kernel)
        nn.init.zeros_(self.bias)

    def forward(
        self,
        inputs: torch.Tensor,
        state: torch.Tensor,
        episode_start: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if state.shape != (inputs.shape[0], 2 * self.hidden_size):
            raise ValueError(
                f"expected recurrent state {(inputs.shape[0], 2 * self.hidden_size)}, "
                f"got {tuple(state.shape)}"
            )
        cell, hidden = state.chunk(2, dim=-1)
        if episode_start is not None:
            keep = (~episode_start.to(dtype=torch.bool)).to(inputs.dtype).unsqueeze(-1)
            cell, hidden = cell * keep, hidden * keep
        gates = inputs @ self.input_kernel + hidden @ self.hidden_kernel + self.bias
        input_gate, forget_gate, output_gate, candidate = gates.chunk(4, dim=-1)
        cell = torch.sigmoid(forget_gate) * cell + torch.sigmoid(input_gate) * torch.tanh(candidate)
        hidden = torch.sigmoid(output_gate) * torch.tanh(cell)
        return hidden, torch.cat((cell, hidden), dim=-1)


class LstmPolicy(nn.Module):
    """Recurrent policy preserving the legacy gate order and env-major layout."""

    def __init__(self, observation_dim: int, action_dim: int, hidden_size: int = 512) -> None:
        super().__init__()
        self.is_recurrent = True
        self.observation_dim = int(observation_dim)
        self.action_dim = int(action_dim)
        self.hidden_size = int(hidden_size)
        self.state_size = 2 * self.hidden_size
        self.encoder = nn.Sequential(
            nn.Linear(observation_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
        )
        self.lstm = TStarLSTMCell(512, hidden_size)
        self.policy_head = nn.Linear(hidden_size, action_dim)
        self.value_head = nn.Linear(hidden_size, 1)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.encoder:
            if isinstance(module, nn.Linear):
                _orthogonal_linear(module, 2.0**0.5)
        self.lstm.reset_parameters()
        _orthogonal_linear(self.policy_head, 1.0)
        _orthogonal_linear(self.value_head, 1.0)

    def initial_state(self, batch_size: int, device=None) -> torch.Tensor:
        return torch.zeros(batch_size, self.state_size, device=device)

    def forward_sequence(
        self,
        observation: torch.Tensor,
        state: torch.Tensor,
        episode_starts: torch.Tensor | None = None,
    ) -> tuple[PolicyOutput, torch.Tensor]:
        flat = observation.reshape(observation.shape[0], -1)
        if flat.shape[1] != self.observation_dim:
            raise ValueError(f"expected observation size {self.observation_dim}, got {flat.shape[1]}")
        environments = state.shape[0]
        if flat.shape[0] % environments:
            raise ValueError("env-major sequence length must be divisible by state batch size")
        steps = flat.shape[0] // environments
        encoded = self.encoder(flat).reshape(environments, steps, -1)
        if episode_starts is None:
            episode_starts = torch.zeros(flat.shape[0], dtype=torch.bool, device=flat.device)
        starts = episode_starts.reshape(environments, steps)
        outputs = []
        next_state = state
        for step in range(steps):
            hidden, next_state = self.lstm(encoded[:, step], next_state, starts[:, step])
            outputs.append(hidden)
        hidden = torch.stack(outputs, dim=1).reshape(flat.shape[0], self.hidden_size)
        return PolicyOutput(self.policy_head(hidden), self.value_head(hidden).squeeze(-1)), next_state

    def distribution(self, logits, action_mask=None) -> Categorical:
        return Categorical(logits=masked_logits(logits, action_mask))

    def act(self, observation, state, episode_starts=None, action_mask=None, *, deterministic=False):
        output, next_state = self.forward_sequence(observation, state, episode_starts)
        distribution = self.distribution(output.logits, action_mask)
        action = distribution.logits.argmax(-1) if deterministic else distribution.sample()
        return action, output.value, -distribution.log_prob(action), next_state

    def evaluate_actions(self, observation, actions, state, episode_starts=None, action_mask=None):
        output, next_state = self.forward_sequence(observation, state, episode_starts)
        distribution = self.distribution(output.logits, action_mask)
        return output.value, -distribution.log_prob(actions), distribution.entropy(), distribution.logits, next_state
