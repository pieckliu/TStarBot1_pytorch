# Milestones 2-3 report: MLP PPO and local training

Audit date: 2026-10-23

## Completed work

- Replaced `ppo_policies.py` with a native `torch.nn.Module` MLP policy.
- Replaced the TensorFlow PPO model/loss path in `ppo_agent.py` with PyTorch.
- Added categorical sampling, deterministic inference, negative log
  probabilities, entropy, and strict action-mask validation.
- Added independently testable PPO loss and GAE functions.
- Added Adam updates, global-norm clipping, explicit device selection, and
  versioned PyTorch checkpoints with schema validation.
- Added `SyntheticMacroEnv`, local rollout collection, local optimization,
  deterministic evaluation, and JSON metrics.
- Added `python -m sc2learner.bin.train_ppo_local` as the SC2-free CPU entry
  point.
- Updated the PPO branch of the existing SC2 evaluation entry point to remove
  TensorFlow session creation and instantiate the PyTorch agent.

## Explicit boundaries

- `LstmPolicy` is a migration boundary until Milestone 5.
- `PPOActor`, `PPOLearner`, and `PPOSelfplayActor` raise a clear deferred-work
  error until their corresponding distributed/self-play milestones.
- TensorFlow legacy utilities remain in `utils_tf.py` only for potential
  reference/conversion work. Neither the migrated PPO core nor local entry point
  imports TensorFlow.
- DQN was not modified.
- StarCraft II was not launched.

## Behavioral differences

1. All-zero action masks now raise an exception. The old finite `-1e30` logits
   could produce a uniform invalid distribution.
2. Deterministic argmax inference is available but stochastic action sampling
   remains the default.
3. Local GAE explicitly distinguishes truncation from termination and prevents
   trace leakage across either reset boundary.
4. Native checkpoints are versioned named PyTorch state dictionaries and are
   intentionally incompatible with unnamed TensorFlow joblib parameter lists.
5. Orthogonality and gains match the old initializer, but seeded random weights
   and sampled action sequences are not cross-framework identical.

## Validation coverage

Final framework-independent test result: **40 passed, 1 SC2 test skipped** on
Windows with Python 3.11.9 and PyTorch 2.14.1+cpu. `pip check`, Python bytecode
compilation, and `git diff --check` also passed.

The tests cover:

- MLP output shapes and independent policy/value parameter towers;
- orthogonal gains and zero biases;
- stochastic and deterministic actions, negative log probability, invalid
  action exclusion, shape validation, and all-zero masks;
- clipped policy loss, unclipped/clipped value loss, entropy, KL and clip
  fraction against independent tensor calculations;
- terminal and truncation GAE reference values;
- gradients, clipping metrics and optimizer parameter updates;
- checkpoint round trips and schema mismatch rejection;
- synthetic rollout shapes, episode boundaries and finite advantages;
- local update metrics and deterministic evaluation repeatability;
- the existing Milestone 1 dependency, packaging and API compatibility tests.

## Executed smoke run

```text
python -m sc2learner.bin.train_ppo_local \
  --updates 2 --rollout-steps 16 --horizon 4 --seed 7 \
  --checkpoint .tmp-ppo-smoke.pt
```

The run completed two CPU optimizer updates, wrote and subsequently removed the
temporary checkpoint, emitted every required metric, and completed deterministic
evaluation with mean return `4.0`. The observed value loss decreased from about
`1.4545` to `0.5552`; this tiny synthetic run is a data-flow smoke test, not a
learning-quality benchmark.

## Remaining limitations

- No TensorFlow-versus-PyTorch fixed-weight forward comparison was executed
  because TensorFlow is not installed in the validated Python 3.11 environment.
  Independent numerical references and initializer-property tests were used.
- The SC2 evaluation factory was updated but could not be run without game
  assets.
- Distributed policy versioning, transport, shutdown and actor/learner tests
  remain Milestone 4 work.
- Recurrent state equations and resets remain Milestone 5 work.
