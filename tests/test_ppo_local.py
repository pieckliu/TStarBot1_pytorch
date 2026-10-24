import math

import numpy as np
import torch

from sc2learner.agents.ppo_agent import Model
from sc2learner.agents.ppo_local import (
    LocalPPOTrainer,
    collect_rollout,
    evaluate_policy,
)
from sc2learner.agents.ppo_policies import LstmPolicy, MlpPolicy
from sc2learner.envs.mock_env import SyntheticMacroEnv


def _model(env):
    return Model(
        policy=MlpPolicy,
        ob_space=env.observation_space,
        ac_space=env.action_space,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        learning_rate=1e-3,
    )


def test_synthetic_rollout_shapes_and_boundaries():
    torch.manual_seed(0)
    env = SyntheticMacroEnv(masked=True, horizon=4)
    rollout = collect_rollout(
        env, _model(env), 10, gamma=0.998, gae_lambda=0.95, seed=0
    )
    features, masks = rollout.observations
    assert features.shape == (10, 8)
    assert masks.shape == (10, 4)
    assert rollout.actions.shape == rollout.returns.shape == (10,)
    assert rollout.dones.sum() == 3
    assert rollout.terminated.sum() + rollout.truncated.sum() == 2
    assert len(rollout.episode_returns) == 2
    assert np.isfinite(rollout.advantages).all()


def test_local_training_smoke_updates_parameters_and_metrics():
    torch.manual_seed(2)
    env = SyntheticMacroEnv(masked=True, horizon=4)
    model = _model(env)
    before = {name: value.clone() for name, value in model.read_params().items()}
    trainer = LocalPPOTrainer(
        env,
        model,
        rollout_steps=16,
        gamma=0.998,
        gae_lambda=0.95,
        learning_rate=1e-3,
        clip_range=0.1,
    )
    metrics, rollout = trainer.update(seed=8)

    assert len(rollout.episode_returns) == 4
    for name in (
        "policy_loss",
        "value_loss",
        "entropy",
        "approx_kl",
        "clip_fraction",
        "gradient_norm",
        "rollout_mean_reward",
        "episode_return",
        "steps_per_second",
    ):
        assert math.isfinite(metrics[name])
    assert any(
        not torch.equal(before[name], after)
        for name, after in model.read_params().items()
    )


def test_deterministic_evaluation_is_repeatable():
    torch.manual_seed(5)
    env = SyntheticMacroEnv(masked=True, horizon=4)
    model = _model(env)

    first = evaluate_policy(env, model, episodes=3, seed=100)
    second = evaluate_policy(env, model, episodes=3, seed=100)

    assert first == second
    assert len(first["action_sequences"]) == 3


def test_recurrent_rollout_and_update_carry_initial_state():
    torch.manual_seed(9)
    env = SyntheticMacroEnv(masked=True, horizon=4)
    model = Model(
        policy=LstmPolicy, ob_space=env.observation_space,
        ac_space=env.action_space, learning_rate=1e-4,
    )
    rollout = collect_rollout(env, model, 8, gamma=0.998, gae_lambda=0.95, seed=2)
    assert rollout.initial_state.shape == (1, 1024)
    metrics = model.update(
        1e-4, 0.1, rollout.observations, rollout.returns, rollout.actions,
        rollout.values, rollout.negative_log_probabilities,
        rollout.dones, rollout.initial_state,
    )
    assert math.isfinite(metrics["total_loss"])
