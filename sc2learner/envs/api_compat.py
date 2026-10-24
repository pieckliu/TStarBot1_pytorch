"""Normalize legacy Gym and modern Gymnasium environment results."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class ResetResult:
    observation: Any
    info: dict[str, Any]


@dataclass(frozen=True)
class StepResult:
    observation: Any
    reward: float
    terminated: bool
    truncated: bool
    info: dict[str, Any]

    @property
    def done(self) -> bool:
        return self.terminated or self.truncated

    @property
    def bootstrap_allowed(self) -> bool:
        return not self.terminated


def normalize_reset(result: Any) -> ResetResult:
    """Normalize reset without confusing ``(features, mask)`` observations."""
    if (
        isinstance(result, tuple)
        and len(result) == 2
        and isinstance(result[1], Mapping)
    ):
        observation, info = result
        return ResetResult(observation, dict(info))
    return ResetResult(result, {})


def normalize_step(result: Any) -> StepResult:
    """Normalize legacy four-value and modern five-value step results."""
    if not isinstance(result, tuple):
        raise TypeError("env.step() must return a tuple")

    if len(result) == 5:
        observation, reward, terminated, truncated, info = result
    elif len(result) == 4:
        observation, reward, done, info = result
        if not isinstance(info, Mapping):
            raise TypeError("env.step() info must be a mapping")
        truncated = bool(done and info.get("TimeLimit.truncated", False))
        terminated = bool(done and not truncated)
    else:
        raise ValueError(
            "env.step() must return 4 values (Gym) or 5 values (Gymnasium)"
        )

    if not isinstance(info, Mapping):
        raise TypeError("env.step() info must be a mapping")
    if bool(terminated) and bool(truncated):
        raise ValueError("a transition cannot be both terminated and truncated")
    return StepResult(
        observation,
        float(reward),
        bool(terminated),
        bool(truncated),
        dict(info),
    )


def as_legacy_step(result: StepResult) -> tuple[Any, float, bool, dict[str, Any]]:
    """Expose a normalized result through TStarBot1's historical step API."""
    info = dict(result.info)
    if result.truncated:
        info.setdefault("TimeLimit.truncated", True)
    return result.observation, result.reward, result.done, info
