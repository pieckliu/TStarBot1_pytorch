import numpy as np
import pytest

from sc2learner.envs.api_compat import normalize_reset, normalize_step
from tests.support.mock_env import ACTION_NAMES, FEATURE_NAMES, SyntheticMacroEnv


def test_masked_observation_schema_is_stable():
    env = SyntheticMacroEnv(masked=True)
    features, mask = normalize_reset(env.reset(seed=17)).observation
    assert env.action_names == ACTION_NAMES
    assert FEATURE_NAMES[-4:] == tuple(f"mask_{name}" for name in ACTION_NAMES)
    assert features.shape == (8,)
    assert mask.shape == (4,)
    assert features.dtype == mask.dtype == np.float32
    np.testing.assert_array_equal(features[-4:], mask)
    assert mask[0] == 1.0
    assert env.observation_space.contains((features, mask))


def test_unmasked_observation_is_single_array():
    env = SyntheticMacroEnv(masked=False)
    observation = normalize_reset(env.reset(seed=17)).observation
    assert isinstance(observation, np.ndarray)
    assert observation.shape == (8,)
    assert env.observation_space.contains(observation)


@pytest.mark.parametrize("api", ["legacy", "modern"])
@pytest.mark.parametrize("ending", ["terminated", "truncated"])
def test_legacy_and_modern_episode_boundaries(api, ending):
    env = SyntheticMacroEnv(api=api, ending=ending, horizon=2)
    normalize_reset(env.reset(seed=5))
    first = normalize_step(env.step(0))
    final = normalize_step(env.step(0))
    assert not first.done
    assert final.done
    assert final.terminated is (ending == "terminated")
    assert final.truncated is (ending == "truncated")
    assert final.bootstrap_allowed is (ending == "truncated")


def test_reset_seed_is_deterministic():
    env = SyntheticMacroEnv()
    first = normalize_reset(env.reset(seed=123)).observation
    second = normalize_reset(env.reset(seed=123)).observation
    for first_part, second_part in zip(first, second):
        np.testing.assert_array_equal(first_part, second_part)


def test_masked_action_is_rejected():
    env = SyntheticMacroEnv(masked=True)
    env.reset(seed=0)
    with pytest.raises(ValueError, match="masked"):
        env.step(1)
