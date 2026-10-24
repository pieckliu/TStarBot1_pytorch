# PyTorch PPO training pipeline

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
