import numpy as np
import pytest

from sc2learner.envs.api_compat import as_legacy_step, normalize_reset, normalize_step


def test_normalize_reset_preserves_masked_tuple_observation():
    observation = (np.zeros(3, dtype=np.float32), np.ones(2, dtype=np.float32))
    normalized = normalize_reset(observation)
    assert normalized.observation is observation
    assert normalized.info == {}


def test_normalize_reset_unpacks_modern_result():
    observation = np.zeros(3, dtype=np.float32)
    normalized = normalize_reset((observation, {"seed": 7}))
    assert normalized.observation is observation
    assert normalized.info == {"seed": 7}


@pytest.mark.parametrize(
    ("raw", "terminated", "truncated", "bootstrap_allowed"),
    [
        (("obs", 1, True, {}), True, False, False),
        (("obs", 0, True, {"TimeLimit.truncated": True}), False, True, True),
        (("obs", 1, True, False, {}), True, False, False),
        (("obs", 0, False, True, {}), False, True, True),
    ],
)
def test_normalize_step_boundaries(raw, terminated, truncated, bootstrap_allowed):
    normalized = normalize_step(raw)
    assert normalized.terminated is terminated
    assert normalized.truncated is truncated
    assert normalized.done is True
    assert normalized.bootstrap_allowed is bootstrap_allowed


def test_as_legacy_step_preserves_truncation_marker():
    normalized = normalize_step(("obs", 0.0, False, True, {"source": "test"}))
    assert as_legacy_step(normalized) == (
        "obs", 0.0, True, {"source": "test", "TimeLimit.truncated": True}
    )


@pytest.mark.parametrize(
    "raw", ["not-a-tuple", ("too", "short"), ("obs", 0, False, False, False, {})]
)
def test_normalize_step_rejects_invalid_arity(raw):
    with pytest.raises((TypeError, ValueError)):
        normalize_step(raw)


def test_normalize_step_rejects_double_boundary():
    with pytest.raises(ValueError, match="both terminated and truncated"):
        normalize_step(("obs", 0.0, True, True, {}))
