# Milestones 4-5 report: distributed PPO and LSTM

Audit date: 2026-10-24

## Completed

- Added a versioned ZeroMQ protocol for named model parameters, rollouts, and
  actor policy-version tracking.
- Restored the REQ/REP parameter channel and PUSH/PULL rollout channel.
- Preserved bounded recent-data reuse, MLP eight-way unroll splitting, and
  complete unsplit recurrent sequences.
- Added a stateful distributed collector so actor unroll boundaries do not
  reset the environment or recurrent state.
- Migrated the custom recurrent cell with explicit `[c,h]` state, `i,f,o,u`
  gates, per-step reset masks, and environment-major sequence packing.
- Generalized acting, value inference, PPO updates, local rollout collection,
  deterministic evaluation, and checkpoints to MLP and LSTM policies.
- Removed TensorFlow setup from `sc2learner.bin.train_ppo` and added explicit
  PyTorch device selection.

## Intentional differences and boundaries

- Named, schema-checked PyTorch/NumPy parameter dictionaries replace positional
  TensorFlow variable lists. Protocol and checkpoint formats are versioned.
- The recurrent value output is standardized to `[batch]`, and the old
  undefined-variable bug in `LstmPolicy.value` is fixed.
- ZeroMQ Python-object transport remains pickle-based for compatibility and is
  documented as trusted-network-only; authentication is not added here.
- Self-play actor/opponent-pool migration remains Milestone 6.
- StarCraft II was not launched because the game assets are not present in the
  validated core environment.

## Validation

The unit suite covers direct gate-equation parity, reset equivalence,
multi-environment sequence layout, recurrent rollout/update flow, protocol
validation, parameter version refresh, checkpoint creation, and a real
one-actor/one-learner loopback update. The loopback test requires permission to
create local ZeroMQ sockets; restricted sandboxes skip it, while the same test
was executed successfully outside that restriction. Final full-suite result:
**44 passed, 1 optional SC2 runtime test skipped**. Bytecode compilation,
`git diff --check`, and `pip check` also passed.
