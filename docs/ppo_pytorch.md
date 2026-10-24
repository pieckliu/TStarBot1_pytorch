# PyTorch PPO: local, distributed, and recurrent execution

## Supported scope

The migrated path supports the original non-spatial MLP and LSTM PPO policies
on CPU or an explicitly selected CUDA device. It includes stochastic acting,
deterministic argmax evaluation, valid-action masking, PPO optimization, GAE,
checkpointing, SC2-free local rollouts, and ZeroMQ actor/learner execution.

Self-play uses the same versioned parameter protocol and a bounded historical
opponent pool. TensorFlow checkpoint lists are not silently loaded by the
native PyTorch path.

## Architecture and preserved math

`MlpPolicy` has independent actor and critic towers. Each contains three
512-unit `tanh` layers. Hidden weights use orthogonal gain `sqrt(2)`, the
policy head uses gain `0.01`, the value head uses gain `1.0`, and all biases are
zero. Values have shape `[batch]`.

The loss preserves the original conventions:

- stored probabilities are negative log probabilities;
- `ratio = exp(old_neglogp - new_neglogp)`;
- advantages are `returns - old_values`, normalized over the update batch with
  population standard deviation and epsilon `1e-8`;
- policy loss is the maximum of clipped and unclipped negative surrogates;
- value loss is half MSE, with optional value clipping disabled by default;
- entropy is subtracted from total loss;
- approximate KL uses half the mean squared negative-log-probability change;
- Adam epsilon is `1e-5` and gradient clipping is global norm.

Invalid logits are set to negative infinity. An all-zero action-mask row raises
an error instead of reproducing the historical accidental uniform distribution
over invalid actions. The SC2 wrapper's no-op action is expected to remain
valid, so this is an invariant check and documented legacy bug fix.

PyTorch's orthogonal initializer preserves the requested matrix property and
gain but does not reproduce TensorFlow/NumPy RNG samples from the same seed.
Sampling streams likewise differ across frameworks. Numerical validation uses
fixed tensors and distribution probabilities rather than matching RNG streams.

## Recurrent policy

`LstmPolicy` directly implements the legacy cell rather than using
`torch.nn.LSTM`. Its state is `[cell, hidden]`, gate order is `i,f,o,u`, and an
episode-start mask clears both state halves before that time step. Flat training
batches remain environment-major: `[environment, time, ...]` is flattened with
all steps of one environment adjacent. The shared encoder is two 512-unit ReLU
layers, the recurrent width is 512, and both heads use orthogonal gain `1.0`.

Two legacy defects are intentionally fixed: recurrent value inference uses the
provided boundary mask (the TensorFlow code referenced an undefined variable),
and recurrent values have shape `[batch]` rather than `[batch, 1]`.

## Distributed actor/learner

`sc2learner.bin.train_ppo` retains the two-channel topology:

- REQ/REP on port A exchanges named CPU NumPy parameter snapshots;
- PUSH/PULL on port B transports rollout messages.

Every message carries protocol version `1`, a message kind, and rollouts carry
the policy version used to generate them. The learner reports mean policy lag.
MLP unrolls are split into eight fragments by default and randomly sampled from
a bounded recent-fragment deque without removal, preserving the historical
replay-like learner behavior. Recurrent unrolls are never split and retain one
initial `[c,h]` state per sequence. The transport uses Python pickle through
ZeroMQ and therefore must only be exposed to trusted workers/networks.

Use `--device=auto`, `cpu`, or `cuda`; the old TensorFlow session setup has been
removed from this entry point. SC2 dependencies and assets are still required
to launch the real training environment.

## GAE boundary behavior

Local rollouts distinguish true termination from time-limit truncation:

- true terminals suppress value bootstrapping;
- truncations bootstrap from the final observation;
- both stop the recursive GAE trace before the environment is reset.

This removes the ambiguity of the historical single `done` value while the API
adapter continues to accept old four-value environment results.

## Local CPU training

After installing the core/test environment:

```bash
python -m sc2learner.bin.train_ppo_local \
  --updates 10 \
  --rollout-steps 128 \
  --device cpu \
  --checkpoint checkpoints/mock-ppo.pt
```

The command uses `SyntheticMacroEnv`; it does not import or launch StarCraft
II. Each update logs policy loss, value loss, entropy, approximate KL, clip
fraction, total loss, gradient norm, rollout reward, completed-episode return,
and steps per second as JSON. It finishes with deterministic evaluation.

The existing `sc2learner.bin.evaluate` PPO factory creates a PyTorch
`PPOAgent`. Use `--deterministic=true` for argmax evaluation and `--device` to
select the PyTorch device. This entry point still requires the external SC2
runtime and was not game-tested in Milestones 2-5.

## Checkpoints

Native checkpoints contain:

- format version;
- policy class;
- observation shape, action count, and mask contract;
- value-clipping configuration;
- model state dictionary;
- optimizer state dictionary.

Loading validates the observation/action schema. TensorFlow joblib lists are
not accepted as PyTorch checkpoints. No converter is supplied because a safe
conversion would require explicit name/layout mapping and fixed-input parity.
