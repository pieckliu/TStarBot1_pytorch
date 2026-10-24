# PyTorch PPO and DQN training pipelines

The requested fork replaces the historical TensorFlow Session/Graph PPO with
PyTorch while retaining the actor-learner topology and macro environment:

```text
SC2RawEnv + wrappers
  -> PPOActor collects 8-step rollout with action masks and GAE
  -> ZeroMQ PUSH sends versioned NumPy trajectory
  -> PPOLearner samples rollout fragments
  -> clipped PPO loss + value loss - entropy bonus
  -> Adam update and policy_version increment
  -> ZeroMQ REQ/REP returns named model parameters to the next actor sync
```

The transport is Python-object serialization and must only be used on trusted
worker networks. Checkpoints contain a format version, policy class,
observation shape, action dimension, mask/value-clip flags, model state, and
optimizer state. Loading rejects schema mismatches.

The real smoke configuration used one CPU actor, one CPU learner, rollout
length 8, batch size 1, difficulty 1, one actor unroll, one learner update,
print/save interval 1, and ports 15800/15801. The final scripted run reported
`total_loss=0.0015115011`, `value_loss=0.0137144933`,
`entropy=0.5345796`, `gradient_norm=1.5857598`, and `policy_version=1`.
The same 23,554,647-byte checkpoint schema was previously saved, reloaded, and
used for a complete real SC2 game.

`max_actor_unrolls=0` and `max_learner_updates=0` remain the defaults, so normal
training behavior is unlimited. The finite flags only make automated smoke
tests terminate cleanly. Run `scripts/train_ppo_smoke.ps1` for the bounded test.

## Dueling Double DQN

DQN uses an unmasked `Discrete(62)` action space and a 1623-dimensional
region-enabled observation. Its actor receives the online network and epsilon
over a REQ/REP socket, completes an SC2 episode, computes Monte Carlo returns,
and sends replay blocks through a two-port PUSH/PULL proxy. The learner samples
the remote replay, applies the preserved Double-DQN/MMC target and per-element
gradient clipping, updates the online network, and periodically copies it to
the target network.

The bounded Windows test uses one actor episode, replay warmup 128, batch size
8, and one learner update across ports 15900-15902. The validated run completed
with 512 received replay transitions, loss `0.0026543457`, and epsilon
`0.9999991`. Its 39,498,145-byte checkpoint contains the online network, target
network, Adam state, and `num_optim_steps=1`.

`max_actor_episodes=0` and `max_learner_updates=0` preserve unlimited normal
training. Run `scripts/train_dqn_smoke.ps1` for the finite real-SC2 test.
