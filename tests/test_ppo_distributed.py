import socket
import threading
import errno

import pytest
import zmq
from sc2learner.agents.ppo_distributed import (
    OpponentPool, PPOActor, PPOLearner, PPOSelfplayActor, _validate,
)
from sc2learner.agents.ppo_policies import MlpPolicy
from sc2learner.envs.mock_env import SyntheticMacroEnv


def _free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return str(sock.getsockname()[1])


def test_protocol_rejects_version_and_schema_mismatches():
    with pytest.raises(ValueError, match="protocol version"):
        _validate({"protocol_version": 99, "kind": "parameter_request"}, "parameter_request")
    with pytest.raises(ValueError, match="missing fields"):
        _validate({"protocol_version": 1, "kind": "parameters"}, "parameters")


def test_opponent_pool_evicts_old_snapshots_and_can_select_latest():
    pool = OpponentPool(2)
    pool.add({"weight": [1.0]})
    pool.add({"weight": [2.0]})
    pool.add({"weight": [3.0]})
    historical, was_latest = pool.choose({"weight": [9.0]}, probability_latest=0.0)
    assert not was_latest
    assert historical["weight"][0] in {2.0, 3.0}
    latest, was_latest = pool.choose({"weight": [9.0]}, probability_latest=1.0)
    assert was_latest
    assert latest["weight"][0] == 9.0


class PairedSyntheticEnv:
    def __init__(self, horizon=2):
        self.base = SyntheticMacroEnv(masked=True, horizon=horizon)
        self.observation_space = self.base.observation_space
        self.action_space = self.base.action_space

    def reset(self, *, seed=None):
        self.base.reset(seed=seed)
        observation = self.base._observation()
        return (observation, observation), {"seed": seed}

    def step(self, actions):
        learner_action, opponent_action = map(int, actions)
        mask = self.base._mask()
        if not mask[learner_action] or not mask[opponent_action]:
            raise ValueError("masked self-play action")
        reward = float(learner_action == self.base._target_action())
        self.base._step_id += 1
        terminated = self.base._step_id >= self.base.horizon
        observation = self.base._observation()
        return (observation, observation), reward, terminated, False, {}


def test_one_actor_one_learner_updates_and_resynchronizes(tmp_path):
    parameter_port, rollout_port = _free_port(), _free_port()
    try:
        learner = PPOLearner(
            env=SyntheticMacroEnv(masked=True, horizon=4), policy=MlpPolicy,
            unroll_length=8, lr=1e-4, clip_range=0.1, batch_size=1,
            save_interval=1, save_dir=tmp_path, port_A=parameter_port,
            port_B=rollout_port,
        )
    except zmq.ZMQError as error:
        if error.errno == errno.EACCES:
            pytest.skip("sandbox denies ZeroMQ context creation")
        raise
    actor = PPOActor(
        env=SyntheticMacroEnv(masked=True, horizon=4), policy=MlpPolicy,
        unroll_length=8, gamma=0.998, lam=0.95,
        learner_ip="127.0.0.1", port_A=parameter_port, port_B=rollout_port,
    )
    result = {}
    learner_thread = threading.Thread(
        target=lambda: result.setdefault("metrics", learner.run(max_updates=1)),
        daemon=True,
    )
    learner_thread.start()
    try:
        assert actor.run(max_unrolls=1) == 1
        learner_thread.join(timeout=20)
        assert not learner_thread.is_alive()
        assert result["metrics"]["policy_version"] == 1
        assert (tmp_path / "ppo_1.pt").exists()
        service_thread = threading.Thread(target=lambda: learner._service(2000), daemon=True)
        service_thread.start()
        assert actor.synchronize() == 1
        service_thread.join(timeout=5)
    finally:
        actor.close()
        learner.close()


def test_selfplay_actor_uses_pool_and_feeds_learner(tmp_path):
    parameter_port, rollout_port = _free_port(), _free_port()
    try:
        learner = PPOLearner(
            env=SyntheticMacroEnv(masked=True, horizon=2), policy=MlpPolicy,
            unroll_length=8, lr=1e-4, clip_range=0.1, batch_size=1,
            save_interval=0, save_dir=tmp_path, port_A=parameter_port,
            port_B=rollout_port,
        )
    except zmq.ZMQError as error:
        if error.errno == errno.EACCES:
            pytest.skip("sandbox denies ZeroMQ context creation")
        raise
    actor = PPOSelfplayActor(
        env=PairedSyntheticEnv(), policy=MlpPolicy, unroll_length=8,
        gamma=0.998, lam=0.95, model_cache_size=3, model_cache_prob=1.0,
        learner_ip="127.0.0.1", port_A=parameter_port, port_B=rollout_port,
    )
    result = {}
    thread = threading.Thread(
        target=lambda: result.setdefault("metrics", learner.run(max_updates=1)),
        daemon=True,
    )
    thread.start()
    try:
        assert actor.run(max_unrolls=1) == 1
        thread.join(timeout=20)
        assert not thread.is_alive()
        assert result["metrics"]["policy_version"] == 1
        assert len(actor.opponent_pool) == 2
    finally:
        actor.close()
        learner.close()
