from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import random
from copy import deepcopy
import queue
from threading import Thread
import zmq

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from sc2learner.agents.replay_memory import Transition
from sc2learner.agents.replay_memory import RemoteReplayMemory
from sc2learner.utils.utils import tprint


DQN_CHECKPOINT_VERSION = 1
DQN_PROTOCOL_VERSION = 1


def double_dqn_mmc_target(online_next_q, target_next_q, rewards, dones,
                          mc_returns, discount, mmc_beta):
  """Preserve the original DDQN bootstrap/MMC interpolation exactly."""
  next_actions = online_next_q.argmax(dim=1, keepdim=True)
  futures = target_next_q.gather(1, next_actions).squeeze(-1)
  one_step = rewards + discount * futures * (1.0 - dones)
  return one_step * mmc_beta + (1.0 - mmc_beta) * mc_returns


def epsilon_schedule(steps, eps_start, eps_end, eps_decay_steps,
                     eps_decay_steps2):
  if steps < eps_decay_steps:
    return eps_start - (eps_start - eps_end) * steps / eps_decay_steps
  if steps < eps_decay_steps2:
    # The denominator intentionally remains eps_decay_steps2 for parity.
    return eps_end - (eps_end - 0.01) * (
        steps - eps_decay_steps) / eps_decay_steps2
  return 0.01


class DQNAgent(object):

  def __init__(self, network, action_space, init_model_path=None, *, device="cpu"):
    if not hasattr(action_space, "n") or not callable(getattr(action_space, "sample", None)):
      raise TypeError("action_space must provide n and sample()")
    self._action_space = action_space
    self.device = torch.device(device)
    self._network = network.to(self.device)
    self._optimizer = None
    self._target_network = None
    self._num_optim_steps = 0
    if init_model_path is not None:
      self.load(init_model_path)

  def act(self, observation, eps=0):
    self._network.eval()
    if random.uniform(0, 1) >= eps:
      observation = torch.as_tensor(
          np.expand_dims(observation, 0), dtype=torch.float32, device=self.device)
      with torch.no_grad():
        q = self._network(observation)
        action = q.argmax(dim=1).item()
      return action
    else:
      return self._action_space.sample()

  def optimize_step(self,
                    obs_batch,
                    next_obs_batch,
                    action_batch,
                    reward_batch,
                    done_batch,
                    mc_return_batch,
                    discount,
                    mmc_beta,
                    gradient_clipping,
                    adam_eps,
                    learning_rate,
                    target_update_interval):
    # create optimizer
    if self._optimizer is None:
      self._optimizer = optim.Adam(self._network.parameters(),
                                   eps=adam_eps,
                                   lr=learning_rate)
    # create target network
    if self._target_network is None:
      self._target_network = deepcopy(self._network)
      self._target_network.to(self.device)
      self._target_network.eval()

    # update target network
    if self._num_optim_steps % target_update_interval == 0:
      self._target_network.load_state_dict(self._network.state_dict())

    obs_batch = obs_batch.to(self.device)
    next_obs_batch = next_obs_batch.to(self.device)
    action_batch = action_batch.to(self.device)
    reward_batch = reward_batch.to(self.device)
    mc_return_batch = mc_return_batch.to(self.device)
    done_batch = done_batch.to(self.device)

    # compute max-q target
    self._network.eval()
    with torch.no_grad():
      q_next_target = self._target_network(next_obs_batch)
      q_next = self._network(next_obs_batch)
      target_q = double_dqn_mmc_target(
          q_next, q_next_target, reward_batch, done_batch, mc_return_batch,
          discount, mmc_beta)

    # define loss
    self._network.train()
    q = self._network(obs_batch).gather(1, action_batch.view(-1, 1)).squeeze(-1)
    loss = F.mse_loss(q, target_q.detach())

    # compute gradient and update parameters
    self._optimizer.zero_grad()
    loss.backward()
    for param in self._network.parameters():
      if param.grad is not None:
        param.grad.detach().clamp_(-gradient_clipping, gradient_clipping)
    self._optimizer.step()
    self._num_optim_steps += 1
    return loss.detach().item()

  def reset(self):
    pass

  def load_params(self, state_dict):
    self._network.load_state_dict(state_dict)

  def read_params(self):
    return {name: value.detach().cpu().clone()
            for name, value in self._network.state_dict().items()}

  def save(self, path):
    torch.save({
        "format_version": DQN_CHECKPOINT_VERSION,
        "model_state_dict": self._network.state_dict(),
        "target_state_dict": (
            None if self._target_network is None else self._target_network.state_dict()),
        "optimizer_state_dict": (
            None if self._optimizer is None else self._optimizer.state_dict()),
        "num_optim_steps": self._num_optim_steps,
    }, path)

  def load(self, path, *, load_training_state=False):
    checkpoint = torch.load(path, map_location=self.device, weights_only=False)
    # Weight-only checkpoints from the original PyTorch DQN remain loadable.
    if "format_version" not in checkpoint:
      self.load_params(checkpoint)
      return
    if checkpoint["format_version"] != DQN_CHECKPOINT_VERSION:
      raise ValueError("unsupported DQN checkpoint format")
    self.load_params(checkpoint["model_state_dict"])
    if load_training_state:
      if checkpoint["target_state_dict"] is not None:
        self._target_network = deepcopy(self._network)
        self._target_network.load_state_dict(checkpoint["target_state_dict"])
        self._target_network.eval()
      if checkpoint["optimizer_state_dict"] is not None:
        self._optimizer = optim.Adam(self._network.parameters())
        self._optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
      self._num_optim_steps = int(checkpoint["num_optim_steps"])


class DQNActor(object):

  def __init__(self,
               memory_size,
               memory_warmup_size,
               env,
               network,
               discount,
               send_freq=4.0,
               ports=("5700", "5701", "5702"),
               learner_ip="localhost",
               device="cpu"):
    if not hasattr(env.action_space, "n"):
      raise TypeError("environment action space must be discrete")
    assert len(ports) == 3
    self._env = env
    self._discount = discount
    self._epsilon = 1.0

    self._agent = DQNAgent(network, env.action_space, device=device)
    self._replay_memory = RemoteReplayMemory(
        is_server=False,
        memory_size=memory_size,
        memory_warmup_size=memory_warmup_size,
        send_freq=send_freq,
        ports=ports[:2],
        server_ip=learner_ip)

    self._zmq_context = zmq.Context()
    self._model_requestor = self._zmq_context.socket(zmq.REQ)
    self._model_requestor.connect("tcp://%s:%s" % (learner_ip, ports[2]))

  def run(self, max_rollouts=None):
    produced = 0
    while max_rollouts is None or produced < max_rollouts:
      # fetch model
      t = time.time()
      self._update_model()
      tprint("Update model time: %f eps: %f" % (time.time() - t, self._epsilon))
      # rollout
      t = time.time()
      self._rollout()
      tprint("Rollout time: %f" % (time.time() - t))
      produced += 1
    return produced

  def _rollout(self):
    rollout, done = [], False
    observation = self._env.reset()
    while not done:
      action = self._agent.act(observation, eps=self._epsilon)
      next_observation, reward, done, info = self._env.step(action)
      rollout.append(
          (observation, action, reward, next_observation, done))
      observation = next_observation

    discounted_return = 0
    for transition in reversed(rollout):
      reward = transition[2]
      discounted_return = discounted_return * self._discount + reward
      self._replay_memory.push(*transition, discounted_return)

  def _update_model(self):
      self._model_requestor.send_string("request model")
      message = self._model_requestor.recv_pyobj()
      if message.get("protocol_version") != DQN_PROTOCOL_VERSION:
        raise ValueError("unsupported DQN synchronization protocol")
      self._agent.load_params(message["parameters"])
      self._epsilon = float(message["epsilon"])

  def close(self):
    self._model_requestor.close(linger=0)
    self._replay_memory.close()
    self._zmq_context.destroy(linger=0)


class DQNLearner(object):

  def __init__(self,
               network,
               action_space,
               memory_size,
               memory_warmup_size,
               discount,
               eps_start,
               eps_end,
               eps_decay_steps,
               eps_decay_steps2,
               batch_size,
               mmc_beta,
               gradient_clipping,
               adam_eps,
               learning_rate,
               target_update_interval,
               checkpoint_dir,
               checkpoint_interval,
               print_interval,
               ports=("5700", "5701", "5702"),
               init_model_path=None,
               device="cpu"):
    if not hasattr(action_space, "n"):
      raise TypeError("action space must be discrete")
    self._agent = DQNAgent(network, action_space, device=device)
    self._replay_memory = RemoteReplayMemory(
        is_server=True,
        memory_size=memory_size,
        memory_warmup_size=memory_warmup_size,
        ports=ports[:2])
    if init_model_path is not None:
      self._agent.load(init_model_path, load_training_state=True)
    self._model_params = self._agent.read_params()

    self._batch_size = batch_size
    self._mmc_beta = mmc_beta
    self._gradient_clipping = gradient_clipping
    self._adam_eps = adam_eps
    self._learning_rate = learning_rate
    self._target_update_interval = target_update_interval
    self._checkpoint_dir = checkpoint_dir
    self._checkpoint_interval = checkpoint_interval
    self._print_interval = print_interval
    self._discount = discount
    self._eps_start = eps_start
    self._eps_end = eps_end
    self._eps_decay_steps = eps_decay_steps
    self._eps_decay_steps2 = eps_decay_steps2
    self._epsilon = eps_start

    self._zmq_context = zmq.Context()
    self._reply_model_thread = Thread(
        target=self._reply_model, args=(self._zmq_context, ports[2]),
        daemon=True)
    self._reply_model_thread.start()

  def run(self, max_updates=None):
    batch_queue = queue.Queue(8)
    batch_thread = Thread(target=self._prepare_batch,
                          args=(batch_queue, self._batch_size,), daemon=True)
    batch_thread.start()

    updates, loss, total_rollout_frames = 0, [], 0
    time_start = time.time()
    last_loss = None
    while max_updates is None or updates < max_updates:
      updates += 1
      observation, next_observation, action, reward, done, mc_return = \
          batch_queue.get()
      self._epsilon = self._schedule_epsilon(updates)
      last_loss = self._agent.optimize_step(
          obs_batch=observation,
          next_obs_batch=next_observation,
          action_batch=action,
          reward_batch=reward,
          done_batch=done,
          mc_return_batch=mc_return,
          discount=self._discount,
          mmc_beta=self._mmc_beta,
          gradient_clipping=self._gradient_clipping,
          adam_eps=self._adam_eps,
          learning_rate=self._learning_rate,
          target_update_interval=self._target_update_interval)
      loss.append(last_loss)
      self._model_params = self._agent.read_params()
      if updates % self._checkpoint_interval == 0:
        ckpt_path = os.path.join(self._checkpoint_dir,
                                 'checkpoint-%d' % updates)
        self._save_checkpoint(ckpt_path)
      if updates % self._print_interval == 0:
        time_elapsed = time.time() - time_start
        train_fps = self._print_interval * self._batch_size / time_elapsed
        rollout_fps = (self._replay_memory.total - total_rollout_frames) \
            / time_elapsed
        loss_mean = np.mean(loss)
        tprint("Update: %d	Train-fps: %.1f	Rollout-fps: %.1f	"
               "Loss: %.5f	Epsilon: %.5f	Time: %.1f" % (updates, train_fps,
               rollout_fps, loss_mean, self._epsilon, time_elapsed))
        time_start, loss = time.time(), []
        total_rollout_frames = self._replay_memory.total
    return {
        "updates": updates,
        "loss": last_loss,
        "epsilon": self._epsilon,
        "replay_total": self._replay_memory.total,
    }

  def close(self):
    self._replay_memory.close()
    self._zmq_context.destroy(linger=0)
    self._reply_model_thread.join(timeout=1.0)

  def _prepare_batch(self, batch_queue, batch_size):
    while True:
      transitions = self._replay_memory.sample(batch_size)
      batch = self._transitions_to_batch(transitions)
      batch_queue.put(batch)

  def _transitions_to_batch(self, transitions):
    batch = Transition(*zip(*transitions))
    observation = torch.from_numpy(np.stack(batch.observation))
    next_observation = torch.from_numpy(np.stack(batch.next_observation))
    reward = torch.FloatTensor(batch.reward)
    action = torch.LongTensor(batch.action)
    done = torch.Tensor(batch.done)
    mc_return = torch.FloatTensor(batch.mc_return)

    if torch.cuda.is_available():
      observation = observation.pin_memory()
      next_observation = next_observation.pin_memory()
      action = action.pin_memory()
      reward = reward.pin_memory()
      mc_return = mc_return.pin_memory()
      done = done.pin_memory()

    return observation, next_observation, action, reward, done, mc_return

  def _save_checkpoint(self, checkpoint_path):
    self._agent.save(checkpoint_path)

  def _schedule_epsilon(self, steps):
    return epsilon_schedule(
        steps, self._eps_start, self._eps_end,
        self._eps_decay_steps, self._eps_decay_steps2)

  def _reply_model(self, zmq_context, port):
    receiver = zmq_context.socket(zmq.REP)
    receiver.bind("tcp://*:%s" % port)
    try:
      while True:
        assert receiver.recv_string() == "request model"
        receiver.send_pyobj({
            "protocol_version": DQN_PROTOCOL_VERSION,
            "parameters": self._model_params,
            "epsilon": self._epsilon,
        })
    except zmq.ZMQError:
      pass
    finally:
      receiver.close(linger=0)
