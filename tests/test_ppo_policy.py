import math

import pytest
import torch
from torch import nn

from sc2learner.agents.ppo_agent import ppo_loss
from sc2learner.agents.ppo_policies import (
    LstmPolicy, MlpPolicy, TStarLSTMCell, masked_logits,
)


def _orthogonal_gram(weight):
    return weight.T @ weight if weight.shape[0] >= weight.shape[1] else weight @ weight.T


def test_mlp_forward_shapes_and_independent_towers():
    policy = MlpPolicy(observation_dim=8, action_dim=4)
    output = policy(torch.zeros(7, 8))

    assert output.logits.shape == (7, 4)
    assert output.value.shape == (7,)
    assert not {id(parameter) for parameter in policy.policy_body.parameters()} & {
        id(parameter) for parameter in policy.value_body.parameters()
    }


def test_initialization_scales_and_zero_biases():
    torch.manual_seed(3)
    policy = MlpPolicy(observation_dim=8, action_dim=4)
    layers = [module for module in policy.modules() if isinstance(module, nn.Linear)]

    for layer in layers:
        assert torch.count_nonzero(layer.bias) == 0
    checks = [
        (policy.policy_body[0].weight, math.sqrt(2.0)),
        (policy.value_body[0].weight, math.sqrt(2.0)),
        (policy.policy_head.weight, 0.01),
        (policy.value_head.weight, 1.0),
    ]
    for weight, gain in checks:
        gram = _orthogonal_gram(weight.detach())
        expected = torch.eye(gram.shape[0]) * gain**2
        torch.testing.assert_close(gram, expected, atol=2e-5, rtol=2e-5)


def test_sampling_log_probability_and_deterministic_action():
    torch.manual_seed(4)
    policy = MlpPolicy(3, 3)
    observation = torch.tensor([[1.0, 2.0, 3.0]])
    mask = torch.tensor([[True, False, True]])
    output = policy(observation)
    distribution = policy.distribution(output.logits, mask)

    deterministic_action, value, negative_log_probability = policy.act(
        observation, mask, deterministic=True
    )

    assert deterministic_action.item() in {0, 2}
    assert deterministic_action.item() != 1
    assert value.shape == (1,)
    torch.testing.assert_close(
        negative_log_probability,
        -distribution.log_prob(deterministic_action),
    )


def test_mask_never_samples_invalid_action():
    policy = MlpPolicy(2, 4)
    observations = torch.zeros(512, 2)
    mask = torch.tensor([[False, True, False, False]]).expand(512, -1)
    actions, _, _ = policy.act(observations, mask)
    assert torch.all(actions == 1)


def test_mask_validation():
    logits = torch.zeros(2, 3)
    with pytest.raises(ValueError, match="at least one"):
        masked_logits(logits, torch.tensor([[True, False, False], [False] * 3]))
    with pytest.raises(ValueError, match="shape"):
        masked_logits(logits, torch.ones(2, 2))
    with pytest.raises(ValueError, match="finite 0/1"):
        masked_logits(logits, torch.tensor([[1.0, 0.5, 0.0], [1.0, 0.0, 0.0]]))


def test_custom_lstm_cell_matches_reference_gate_equations():
    torch.manual_seed(7)
    cell = TStarLSTMCell(3, 2)
    inputs = torch.randn(2, 3)
    state = torch.randn(2, 4)
    output, next_state = cell(inputs, state)
    c, h = state.chunk(2, -1)
    i, f, o, u = (inputs @ cell.input_kernel + h @ cell.hidden_kernel + cell.bias).chunk(4, -1)
    expected_c = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(u)
    expected_h = torch.sigmoid(o) * torch.tanh(expected_c)
    torch.testing.assert_close(output, expected_h)
    torch.testing.assert_close(next_state, torch.cat((expected_c, expected_h), -1))


def test_lstm_episode_start_reset_and_env_major_sequence():
    torch.manual_seed(8)
    policy = LstmPolicy(3, 2, hidden_size=4)
    observations = torch.randn(6, 3)  # two environments, three steps each
    state = torch.randn(2, 8)
    starts = torch.tensor([True, False, False, False, True, False])
    sequence, next_state = policy.forward_sequence(observations, state, starts)
    env_outputs = []
    env_states = []
    for env in range(2):
        out, final = policy.forward_sequence(
            observations[env * 3:(env + 1) * 3], state[env:env + 1],
            starts[env * 3:(env + 1) * 3],
        )
        env_outputs.append(out.logits)
        env_states.append(final)
    torch.testing.assert_close(sequence.logits, torch.cat(env_outputs))
    torch.testing.assert_close(next_state, torch.cat(env_states))

    reset_output, reset_state = policy.forward_sequence(
        observations[:1], state[:1], torch.tensor([True])
    )
    zero_output, zero_state = policy.forward_sequence(
        observations[:1], torch.zeros_like(state[:1]), torch.tensor([False])
    )
    torch.testing.assert_close(reset_output.logits, zero_output.logits)
    torch.testing.assert_close(reset_state, zero_state)


class _ReferencePolicy:
    def __init__(self, values, negative_log_probabilities, entropies):
        self.values = values
        self.negative_log_probabilities = negative_log_probabilities
        self.entropies = entropies

    def evaluate_actions(self, observations, actions, action_mask):
        del actions, action_mask
        logits = torch.zeros(observations.shape[0], 2)
        return self.values, self.negative_log_probabilities, self.entropies, logits


def test_ppo_loss_matches_independent_reference():
    values = torch.tensor([0.4, 0.1, -0.2, 0.3])
    new_neglogp = torch.tensor([0.6, 0.2, 0.9, 0.5])
    entropy = torch.tensor([0.4, 0.5, 0.6, 0.7])
    old_values = torch.tensor([0.2, 0.0, 0.1, 0.4])
    returns = torch.tensor([1.2, 0.5, -0.5, 0.0])
    old_neglogp = torch.tensor([0.5, 0.4, 0.7, 0.6])
    clip_range = 0.1
    result = ppo_loss(
        _ReferencePolicy(values, new_neglogp, entropy),
        torch.zeros(4, 1),
        torch.tensor([0, 1, 0, 1]),
        returns,
        old_values,
        old_neglogp,
        action_mask=None,
        clip_range=clip_range,
        entropy_coefficient=0.01,
        value_coefficient=0.5,
        value_clip=False,
    )

    raw_advantages = returns - old_values
    advantages = (raw_advantages - raw_advantages.mean()) / (
        raw_advantages.std(unbiased=False) + 1e-8
    )
    ratio = torch.exp(old_neglogp - new_neglogp)
    expected_policy = torch.maximum(
        -advantages * ratio,
        -advantages * torch.clamp(ratio, 0.9, 1.1),
    ).mean()
    expected_value = 0.5 * (values - returns).square().mean()
    expected_kl = 0.5 * (new_neglogp - old_neglogp).square().mean()
    expected_clip_fraction = ((ratio - 1).abs() > clip_range).float().mean()
    expected_total = expected_policy - 0.01 * entropy.mean() + 0.5 * expected_value

    torch.testing.assert_close(result["policy_loss"], expected_policy)
    torch.testing.assert_close(result["value_loss"], expected_value)
    torch.testing.assert_close(result["approx_kl"], expected_kl)
    torch.testing.assert_close(result["clip_fraction"], expected_clip_fraction)
    torch.testing.assert_close(result["total_loss"], expected_total)


def test_value_clipping_uses_larger_squared_error():
    values = torch.tensor([2.0, -2.0])
    old_values = torch.tensor([0.0, 0.0])
    returns = torch.tensor([1.0, -1.0])
    result = ppo_loss(
        _ReferencePolicy(values, torch.zeros(2), torch.ones(2)),
        torch.zeros(2, 1),
        torch.tensor([0, 1]),
        returns,
        old_values,
        torch.zeros(2),
        action_mask=None,
        clip_range=0.1,
        entropy_coefficient=0.0,
        value_coefficient=1.0,
        value_clip=True,
    )
    clipped_values = torch.tensor([0.1, -0.1])
    expected = 0.5 * torch.maximum(
        (values - returns).square(), (clipped_values - returns).square()
    ).mean()
    torch.testing.assert_close(result["value_loss"], expected)
