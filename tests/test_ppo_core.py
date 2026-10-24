import math

import numpy as np
import pytest
import torch

from sc2learner.agents.ppo_agent import Model, compute_gae
from sc2learner.agents.ppo_policies import MlpPolicy
from sc2learner.envs.mock_env import SyntheticMacroEnv


def _model(env, **kwargs):
    return Model(
        policy=MlpPolicy,
        ob_space=env.observation_space,
        ac_space=env.action_space,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        learning_rate=1e-3,
        **kwargs,
    )


def test_gae_terminal_reference_values():
    advantages, returns = compute_gae(
        rewards=[1.0, 2.0],
        values=[0.5, 0.25],
        next_values=[0.25, 9.0],
        terminated=[False, True],
        truncated=[False, False],
        gamma=0.9,
        gae_lambda=0.8,
    )
    np.testing.assert_allclose(advantages, [1.985, 1.75], rtol=1e-6)
    np.testing.assert_allclose(returns, [2.485, 2.0], rtol=1e-6)


def test_gae_truncation_bootstraps_but_stops_trace():
    advantages, _ = compute_gae(
        rewards=[1.0, 2.0],
        values=[0.5, 0.25],
        next_values=[0.25, 3.0],
        terminated=[False, False],
        truncated=[False, True],
        gamma=0.9,
        gae_lambda=0.8,
    )
    np.testing.assert_allclose(advantages, [3.929, 4.45], rtol=1e-6)


def test_optimizer_propagates_gradients_and_updates_parameters():
    torch.manual_seed(11)
    env = SyntheticMacroEnv(masked=True, horizon=4)
    model = _model(env)
    observation, _ = env.reset(seed=0)
    features, mask = observation
    observations = (
        np.repeat(features[None, :], 4, axis=0),
        np.repeat(mask[None, :], 4, axis=0),
    )
    before = {name: value.clone() for name, value in model.read_params().items()}
    metrics = model.update(
        1e-3,
        0.1,
        observations,
        returns=np.asarray([1.0, 0.5, -0.5, 0.25], dtype=np.float32),
        actions=np.asarray([0, 2, 0, 2], dtype=np.int64),
        old_values=np.zeros(4, dtype=np.float32),
        old_negative_log_probabilities=np.ones(4, dtype=np.float32),
    )

    assert metrics["gradient_norm"] > 0
    assert all(math.isfinite(value) for value in metrics.values())
    assert any(
        not torch.equal(before[name], after)
        for name, after in model.read_params().items()
    )


def test_checkpoint_round_trip(tmp_path):
    torch.manual_seed(7)
    env = SyntheticMacroEnv(masked=True)
    model = _model(env)
    observation, _ = env.reset(seed=4)
    expected = model.step(observation, deterministic=True)
    checkpoint = tmp_path / "ppo.pt"
    model.save(checkpoint)

    torch.manual_seed(99)
    restored = _model(env)
    restored.load(checkpoint, load_optimizer=True)
    actual = restored.step(observation, deterministic=True)

    np.testing.assert_array_equal(actual[0], expected[0])
    np.testing.assert_allclose(actual[1], expected[1])
    np.testing.assert_allclose(actual[3], expected[3])


def test_checkpoint_rejects_schema_mismatch(tmp_path):
    env = SyntheticMacroEnv(masked=True)
    model = _model(env)
    checkpoint = tmp_path / "ppo.pt"
    model.save(checkpoint)
    other_env = SyntheticMacroEnv(masked=False)
    with pytest.raises(ValueError, match="schema"):
        _model(other_env).load(checkpoint)
