# Milestones 6-7 report: self-play, DQN, and integration

Audit date: 2026-10-24

## Completed work

- Migrated self-play to named PyTorch snapshots, bounded historical-pool
  eviction, latest/historical selection, frozen evaluator pools, native
  checkpoint seed lists, and paired-player rollout collection.
- Removed TensorFlow session setup from self-play and LAN PPO entry points.
- Preserved DQN dueling heads, Double-DQN selection, target evaluation, MMC
  interpolation, replay, epsilon schedule and element-wise gradient clipping.
- Added explicit DQN devices, safe batch-size-one shapes, modern autograd use,
  a versioned actor/learner message, and resumable native checkpoints.
- Removed all runtime TensorFlow imports and deleted `utils_tf.py`.
- Removed TensorFlow-era joblib from core dependency metadata.
- Documented installation, local and distributed execution, self-play, DQN,
  checkpoint compatibility, baseline flags and known limitations.

## Intentional compatibility decisions

- TensorFlow PPO checkpoints remain incompatible; no unverified positional
  converter is supplied.
- Original PyTorch DQN weight-only state dictionaries remain loadable. New
  checkpoints add target network, optimizer and update count.
- DQN's unusual second epsilon-decay denominator and MMC weighting remain
  unchanged.
- ZeroMQ pickle transport remains trusted-network-only.
- Real SC2 execution remains opt-in and unverified on this machine.

## Validation result

The final full suite completed with **53 passed and 1 skipped**. The skipped
test is the explicitly opt-in real SC2 runtime test. Both the ordinary PPO and
self-play one-actor/one-learner ZeroMQ loopbacks ran successfully. Python
bytecode compilation, `git diff --check`, and `pip check` also passed.
