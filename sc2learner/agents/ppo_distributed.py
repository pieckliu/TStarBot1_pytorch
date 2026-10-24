"""Versioned ZeroMQ transport for PyTorch PPO actors and learners.

The wire format is Python/NumPy and is intended only for trusted workers.
TensorFlow variable lists are deliberately not accepted: parameters are named,
CPU NumPy arrays so schema mismatches fail loudly.
"""

from __future__ import annotations

from collections import deque
from pathlib import Path
import random

import numpy as np
import torch
import zmq

from sc2learner.agents.ppo_agent import _stack_observations, compute_gae
from sc2learner.agents.ppo_local import RolloutBatch, RolloutCollector
from sc2learner.envs.api_compat import normalize_reset, normalize_step


PROTOCOL_VERSION = 1


def _parameters(model, version):
    return {
        "protocol_version": PROTOCOL_VERSION,
        "kind": "parameters",
        "policy_version": int(version),
        "parameters": {name: value.numpy() for name, value in model.read_params().items()},
    }


def _rollout_payload(rollout: RolloutBatch, policy_version: int):
    return {
        "protocol_version": PROTOCOL_VERSION,
        "kind": "rollout",
        "policy_version": int(policy_version),
        "observations": rollout.observations,
        "returns": rollout.returns,
        "episode_starts": rollout.dones,
        "actions": rollout.actions,
        "values": rollout.values,
        "negative_log_probabilities": rollout.negative_log_probabilities,
        "initial_state": rollout.initial_state,
        "episode_returns": rollout.episode_returns,
    }


def _validate(message, kind):
    if not isinstance(message, dict):
        raise ValueError("distributed PPO message must be a dictionary")
    if message.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError("unsupported distributed PPO protocol version")
    if message.get("kind") != kind:
        raise ValueError(f"expected {kind!r} message")
    required = {
        "parameter_request": set(),
        "parameters": {"policy_version", "parameters"},
        "rollout": {
            "policy_version", "observations", "returns", "episode_starts",
            "actions", "values", "negative_log_probabilities",
            "initial_state", "episode_returns",
        },
    }[kind]
    missing = required.difference(message)
    if missing:
        raise ValueError(f"{kind} message is missing fields: {sorted(missing)}")


def _concatenate(parts):
    if isinstance(parts[0], tuple):
        return tuple(np.concatenate(items, axis=0) for items in zip(*parts))
    return np.concatenate(parts, axis=0)


def _clone_parameters(parameters):
    return {name: np.asarray(value).copy() for name, value in parameters.items()}


class OpponentPool:
    """Bounded historical policy pool with explicit latest-policy selection."""

    def __init__(self, capacity):
        if capacity < 1:
            raise ValueError("opponent pool capacity must be positive")
        self._items = deque(maxlen=int(capacity))

    def add(self, parameters):
        self._items.append(_clone_parameters(parameters))

    def choose(self, latest, probability_latest=0.0, rng=random):
        if not self._items or rng.random() < probability_latest:
            return _clone_parameters(latest), True
        return _clone_parameters(rng.choice(tuple(self._items))), False

    def __len__(self):
        return len(self._items)


class PPOActor:
    def __init__(
        self, *, env, policy, unroll_length, gamma, lam, queue_size=1,
        enable_push=True, learner_ip="localhost", port_A="5700", port_B="5701",
        device="cpu", context=None,
    ):
        del queue_size
        # Imported lazily to keep ppo_agent's compatibility re-export cycle safe.
        from sc2learner.agents.ppo_agent import Model

        self.env = env
        self.unroll_length = int(unroll_length)
        self.gamma = float(gamma)
        self.lam = float(lam)
        self.model = Model(policy=policy, ob_space=env.observation_space,
                           ac_space=env.action_space, nbatch_act=1, device=device)
        self.context = context or zmq.Context.instance()
        self.parameter_socket = self.context.socket(zmq.REQ)
        self.parameter_socket.setsockopt(zmq.LINGER, 0)
        self.parameter_socket.connect(f"tcp://{learner_ip}:{port_A}")
        self.rollout_socket = self.context.socket(zmq.PUSH)
        self.rollout_socket.setsockopt(zmq.LINGER, 0)
        self.enable_push = bool(enable_push)
        if self.enable_push:
            self.rollout_socket.connect(f"tcp://{learner_ip}:{port_B}")
        self.policy_version = -1
        self.collector = RolloutCollector(
            env, self.model, gamma=self.gamma, gae_lambda=self.lam
        )

    def synchronize(self):
        self.parameter_socket.send_pyobj({
            "protocol_version": PROTOCOL_VERSION, "kind": "parameter_request"
        })
        message = self.parameter_socket.recv_pyobj()
        _validate(message, "parameters")
        self.model.load_params(message["parameters"])
        self.policy_version = int(message["policy_version"])
        return self.policy_version

    def run(self, max_unrolls=None):
        produced = 0
        while max_unrolls is None or produced < max_unrolls:
            self.synchronize()
            rollout = self.collector.collect(self.unroll_length)
            if self.enable_push:
                self.rollout_socket.send_pyobj(
                    _rollout_payload(rollout, self.policy_version))
            produced += 1
        return produced

    def close(self):
        self.parameter_socket.close(linger=0)
        self.rollout_socket.close(linger=0)


class PPOSelfplayActor(PPOActor):
    """Two-player PPO actor with a bounded pool of historical opponents."""

    def __init__(
        self, *, env, policy, unroll_length, gamma, lam, model_cache_size,
        model_cache_prob, queue_size=1, prob_latest_opponent=0.0,
        init_opponent_pool_filelist=None, freeze_opponent_pool=False,
        enable_push=True, learner_ip="localhost", port_A="5700", port_B="5701",
        device="cpu", context=None,
    ):
        del queue_size
        from sc2learner.agents.ppo_agent import Model

        self.env = env
        self.unroll_length = int(unroll_length)
        self.gamma = float(gamma)
        self.lam = float(lam)
        self.enable_push = bool(enable_push)
        self.model_cache_probability = float(model_cache_prob)
        self.probability_latest_opponent = float(prob_latest_opponent)
        self.freeze_opponent_pool = bool(freeze_opponent_pool)
        self.model = Model(policy=policy, ob_space=env.observation_space,
                           ac_space=env.action_space, nbatch_act=1, device=device)
        self.opponent_model = Model(policy=policy, ob_space=env.observation_space,
                                    ac_space=env.action_space, nbatch_act=1,
                                    device=device)
        self.opponent_pool = OpponentPool(model_cache_size)
        self.latest_parameters = {
            name: value.numpy() for name, value in self.model.read_params().items()
        }
        if init_opponent_pool_filelist:
            for line in Path(init_opponent_pool_filelist).read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                checkpoint = torch.load(line.strip(), map_location="cpu", weights_only=False)
                parameters = checkpoint.get("model_state_dict", checkpoint)
                self.opponent_pool.add(parameters)
        if len(self.opponent_pool) == 0:
            self.opponent_pool.add(self.latest_parameters)
        self.context = context or zmq.Context.instance()
        self.parameter_socket = self.context.socket(zmq.REQ)
        self.parameter_socket.setsockopt(zmq.LINGER, 0)
        self.parameter_socket.connect(f"tcp://{learner_ip}:{port_A}")
        self.rollout_socket = self.context.socket(zmq.PUSH)
        self.rollout_socket.setsockopt(zmq.LINGER, 0)
        if self.enable_push:
            self.rollout_socket.connect(f"tcp://{learner_ip}:{port_B}")
        self.policy_version = -1
        observations = normalize_reset(env.reset()).observation
        self.observation, self.opponent_observation = observations
        self.state = (
            self.model.policy.initial_state(1, self.model.device).cpu().numpy()
            if self.model.is_recurrent else None)
        self.opponent_state = (
            self.opponent_model.policy.initial_state(1, self.opponent_model.device).cpu().numpy()
            if self.opponent_model.is_recurrent else None)
        self.episode_start = True
        self.current_episode_return = 0.0
        self.last_opponent_was_latest = False
        self._select_opponent()

    def synchronize(self):
        version = super().synchronize()
        self.latest_parameters = {
            name: value.numpy() for name, value in self.model.read_params().items()
        }
        if (not self.freeze_opponent_pool
                and random.random() < self.model_cache_probability):
            self.opponent_pool.add(self.latest_parameters)
        return version

    def _select_opponent(self):
        parameters, latest = self.opponent_pool.choose(
            self.latest_parameters, self.probability_latest_opponent
        )
        self.opponent_model.load_params(parameters)
        self.last_opponent_was_latest = latest

    def _collect(self):
        observations, actions, rewards, values = [], [], [], []
        next_values, neglogps, starts = [], [], []
        terminated, truncated, episode_returns = [], [], []
        initial_state = None if self.state is None else self.state.copy()
        for _ in range(self.unroll_length):
            starts.append(self.episode_start)
            action, value, next_state, neglogp = self.model.step(
                self.observation, self.state, np.asarray([self.episode_start]))
            opponent_action, _, next_opponent_state, _ = self.opponent_model.step(
                self.opponent_observation, self.opponent_state,
                np.asarray([self.episode_start]))
            observations.append(self.observation)
            actions.append(int(action[0]))
            values.append(float(value[0]))
            neglogps.append(float(neglogp[0]))
            transition = normalize_step(self.env.step(
                [int(action[0]), int(opponent_action[0])]))
            learner_observation, opponent_observation = transition.observation
            next_values.append(float(self.model.value(
                learner_observation, next_state, np.asarray([False]))[0]))
            rewards.append(transition.reward)
            terminated.append(transition.terminated)
            truncated.append(transition.truncated)
            self.current_episode_return += transition.reward
            self.observation, self.opponent_observation = (
                learner_observation, opponent_observation)
            self.state, self.opponent_state = next_state, next_opponent_state
            self.episode_start = transition.done
            if transition.done:
                episode_returns.append(self.current_episode_return)
                self.current_episode_return = 0.0
                pair = normalize_reset(self.env.reset()).observation
                self.observation, self.opponent_observation = pair
                self.state = (
                    self.model.policy.initial_state(1, self.model.device).cpu().numpy()
                    if self.model.is_recurrent else None)
                self.opponent_state = (
                    self.opponent_model.policy.initial_state(
                        1, self.opponent_model.device).cpu().numpy()
                    if self.opponent_model.is_recurrent else None)
                self._select_opponent()
        advantages, returns = compute_gae(
            rewards, values, next_values, terminated, truncated,
            self.gamma, self.lam)
        return RolloutBatch(
            observations=_stack_observations(observations), returns=returns,
            dones=np.asarray(starts, dtype=np.bool_),
            actions=np.asarray(actions, dtype=np.int64),
            values=np.asarray(values, dtype=np.float32),
            negative_log_probabilities=np.asarray(neglogps, dtype=np.float32),
            advantages=advantages, rewards=np.asarray(rewards, dtype=np.float32),
            terminated=np.asarray(terminated, dtype=np.bool_),
            truncated=np.asarray(truncated, dtype=np.bool_),
            episode_returns=tuple(episode_returns), initial_state=initial_state,
        )

    def run(self, max_unrolls=None):
        produced = 0
        while max_unrolls is None or produced < max_unrolls:
            self.synchronize()
            rollout = self._collect()
            if self.enable_push:
                self.rollout_socket.send_pyobj(
                    _rollout_payload(rollout, self.policy_version))
            produced += 1
        return produced


class PPOLearner:
    def __init__(
        self, *, env, policy, unroll_length, lr, clip_range, batch_size,
        ent_coef=0.01, vf_coef=0.5, max_grad_norm=0.5, queue_size=1024,
        print_interval=1000, save_interval=50000, learn_act_speed_ratio=0,
        save_dir="./checkpoints", init_model_path=None, port_A="5700",
        port_B="5701", device="cpu", context=None, unroll_split=8,
    ):
        from sc2learner.agents.ppo_agent import Model

        self.model = Model(policy=policy, ob_space=env.observation_space,
                           ac_space=env.action_space, nbatch_act=1,
                           ent_coef=ent_coef, vf_coef=vf_coef,
                           max_grad_norm=max_grad_norm, device=device)
        if init_model_path:
            self.model.load(init_model_path, load_optimizer=True)
        self.unroll_length = int(unroll_length)
        self.learning_rate = lr if callable(lr) else lambda _: float(lr)
        self.clip_range = clip_range if callable(clip_range) else lambda _: float(clip_range)
        self.batch_size = int(batch_size)
        self.learn_act_speed_ratio = float(learn_act_speed_ratio)
        self.print_interval = int(print_interval)
        self.save_interval = int(save_interval)
        self.save_dir = None if save_dir is None else Path(save_dir)
        self.policy_version = 0
        self.updates = 0
        self.num_unrolls = 0
        self.unroll_split = 1 if self.model.is_recurrent else int(unroll_split)
        if self.unroll_length % self.unroll_split:
            raise ValueError("unroll_length must be divisible by unroll_split")
        self.samples_per_update = self.batch_size * self.unroll_split
        self.pending = deque(maxlen=int(queue_size) * self.unroll_split)
        self.context = context or zmq.Context.instance()
        self.parameter_socket = self.context.socket(zmq.REP)
        self.parameter_socket.setsockopt(zmq.LINGER, 0)
        self.parameter_socket.bind(f"tcp://*:{port_A}")
        self.rollout_socket = self.context.socket(zmq.PULL)
        self.rollout_socket.setsockopt(zmq.LINGER, 0)
        self.rollout_socket.setsockopt(zmq.RCVHWM, int(queue_size))
        self.rollout_socket.bind(f"tcp://*:{port_B}")
        self.poller = zmq.Poller()
        self.poller.register(self.parameter_socket, zmq.POLLIN)
        self.poller.register(self.rollout_socket, zmq.POLLIN)

    def _service(self, timeout_ms=100):
        events = dict(self.poller.poll(timeout_ms))
        if self.parameter_socket in events:
            request = self.parameter_socket.recv_pyobj()
            _validate(request, "parameter_request")
            self.parameter_socket.send_pyobj(_parameters(self.model, self.policy_version))
        if self.rollout_socket in events:
            message = self.rollout_socket.recv_pyobj()
            _validate(message, "rollout")
            self.num_unrolls += 1
            if self.unroll_split == 1:
                self.pending.append(message)
            else:
                arrays = (
                    "returns", "episode_starts", "actions", "values",
                    "negative_log_probabilities",
                )
                observation_parts = (
                    [np.split(part, self.unroll_split) for part in message["observations"]]
                    if isinstance(message["observations"], tuple)
                    else np.split(message["observations"], self.unroll_split)
                )
                split_arrays = {name: np.split(message[name], self.unroll_split) for name in arrays}
                for index in range(self.unroll_split):
                    fragment = dict(message)
                    fragment["observations"] = (
                        tuple(part[index] for part in observation_parts)
                        if isinstance(message["observations"], tuple)
                        else observation_parts[index]
                    )
                    for name in arrays:
                        fragment[name] = split_arrays[name][index]
                    self.pending.append(fragment)

    def _update(self):
        # Preserve the legacy recent-unroll replay behavior: random sampling
        # does not remove fragments from the bounded deque.
        messages = random.sample(list(self.pending), self.samples_per_update)
        observations = _concatenate([item["observations"] for item in messages])
        initial_states = [item["initial_state"] for item in messages]
        states = None if initial_states[0] is None else np.concatenate(initial_states, axis=0)
        metrics = self.model.update(
            self.learning_rate(self.updates + 1), self.clip_range(self.updates + 1), observations,
            np.concatenate([item["returns"] for item in messages]),
            np.concatenate([item["actions"] for item in messages]),
            np.concatenate([item["values"] for item in messages]),
            np.concatenate([item["negative_log_probabilities"] for item in messages]),
            np.concatenate([item["episode_starts"] for item in messages]), states,
        )
        self.updates += 1
        self.policy_version += 1
        metrics["policy_version"] = self.policy_version
        metrics["mean_policy_lag"] = float(np.mean([
            self.policy_version - 1 - item["policy_version"] for item in messages
        ]))
        if self.save_dir is not None and self.save_interval > 0 and self.updates % self.save_interval == 0:
            self.save_dir.mkdir(parents=True, exist_ok=True)
            self.model.save(self.save_dir / f"ppo_{self.updates}.pt")
        return metrics

    def run(self, max_updates=None):
        last_metrics = None
        while max_updates is None or self.updates < max_updates:
            self._service()
            rate_allows_update = (
                self.learn_act_speed_ratio <= 0
                or self.updates * self.batch_size
                < self.num_unrolls * self.learn_act_speed_ratio
            )
            if len(self.pending) >= self.samples_per_update and rate_allows_update:
                last_metrics = self._update()
                if self.print_interval > 0 and self.updates % self.print_interval == 0:
                    print({key: round(value, 6) for key, value in last_metrics.items()})
        return last_metrics

    def close(self):
        self.parameter_socket.close(linger=0)
        self.rollout_socket.close(linger=0)

