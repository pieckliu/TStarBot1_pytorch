# TStarBot1 TensorFlow-to-PyTorch Migration Plan

## Scope and audit status

This document is the Milestone 0 repository audit. It describes the repository at commit `6a5c1dc` on branch `dev-open` and proposes a staged migration. No implementation code was changed during this milestone.

The audit covered every tracked source file, all executable entry points, packaging metadata, the PPO and DQN data paths, the environment/wrapper stack, and the upstream project constraints. The local audit machine has Python 3.11.9, but none of PyTorch, TensorFlow, Gym/Gymnasium, PySC2, pyzmq, joblib, or pytest is installed. Consequently, this milestone contains static analysis only; no agent, training loop, or StarCraft II runtime was executed.

External facts used in this plan:

- The paper describes a global RL controller over hand-written macro actions, raw unit control and a Zerg technology tree, available-action masking, non-spatial MLP experiments, PPO with truncated GAE, Dueling Double DQN with an MMC target, and distributed rollout collection: [TStarBots paper](https://arxiv.org/html/1809.07193v3).
- The Tencent PySC2 fork explicitly says TStarBot1/TStarBot2 require commit `4f790218` (2019-05-15), while newer commits target TStarBot-X, and it requires an editable installation because the technology-tree data ships with the source: [Tencent PySC2 extension](https://github.com/Tencent/PySC2TencentExtension).
- Current upstream PySC2 is Python 3.8+ and has breaking changes relative to the old API; Windows and macOS are supported but upstream says Linux is the thoroughly tested platform: [upstream PySC2](https://github.com/google-deepmind/pysc2).
- Modern Gymnasium distinguishes `terminated` from `truncated` and changes `reset()` to return `(observation, info)`: [Gymnasium migration guide](https://gymnasium.farama.org/introduction/migration_guide/).
- The current PyTorch installation page requires Python 3.10 or later for its stable release. The exact PyTorch/CUDA pin should be selected and exercised in Milestone 1 rather than inferred from the legacy package file: [PyTorch installation guide](https://pytorch.org/get-started/locally/).

## Executive summary

The repository is a small research codebase with three distinct layers:

1. A StarCraft II environment layer converts Tencent PySC2 raw observations into non-spatial or spatial NumPy features and converts a flat discrete macro-action ID into raw SC2 protocol actions.
2. RL agents consume those features. PPO is TensorFlow 1.x graph/session code; DQN is already PyTorch 0.4-era code.
3. Actor/learner processes exchange rollouts and parameters over ZeroMQ. PPO sends Python/NumPy objects; DQN serializes a PyTorch state dictionary into bytes. Self-play adds an in-memory pool of historical PPO parameter snapshots.

The faithful path is therefore not a rewrite of the environment or DQN. First isolate a framework-independent PPO contract and synthetic environment, then migrate MLP PPO, prove it locally, restore the existing ZeroMQ topology, migrate the custom recurrent policy, and only then modernize self-play and DQN.

There is no current test suite, lock file, CI configuration, or mock environment. `setup.py` pins `gym==0.10.5` and `torch==0.4.0`, permits any TensorFlow version from 1.4.1 upward, omits several direct runtime dependencies, and lists only top-level packages rather than discovering all subpackages.

## Current architecture

### Runtime entry points

| Entry point | Role | Framework path |
| --- | --- | --- |
| `sc2learner/bin/train_ppo.py` | PPO actor or learner against built-in AI | TensorFlow 1.x + ZeroMQ |
| `sc2learner/bin/train_ppo_selfplay.py` | PPO actor, learner, built-in evaluator, or self-play evaluator | TensorFlow 1.x + ZeroMQ |
| `sc2learner/bin/evaluate.py` | PPO, DQN, random, or keyboard evaluation | TensorFlow for PPO; PyTorch for DQN |
| `sc2learner/bin/play_vs_ppo_agent.py` | LAN human-versus-PPO agent | TensorFlow 1.x |
| `sc2learner/bin/train_dqn.py` | DQN actor or learner | PyTorch + ZeroMQ |

All supplied training configurations use non-spatial observations. PPO can select an MLP or LSTM policy. DQN uses `NonspatialDuelingQNet`; the spatial `DuelingQNet` is present but is not selected by the supplied entry point.

### Environment, observation, and macro-action path

The environment stack is:

```text
Tencent PySC2 SC2Env
  -> SC2RawEnv / SC2SelfplayRawEnv / LanSC2RawEnv
  -> optional reward wrapper
  -> ZergActionWrapper (macro ID -> raw protocol actions; optional mask)
  -> ZergObservationWrapper (raw observation -> NumPy features)
  -> PPO or DQN agent
```

`SC2RawEnv` and its self-play/LAN variants expose the legacy Gym API: `reset() -> observation` and `step() -> (observation, reward, done, info)`. They use the Tencent extension's raw unit objects and protocol actions, not merely standard feature-layer PySC2. The default game map is AbyssalReef, the supplied training step multiplier is 32, the game/tech-tree version flag is `4.6`, and the normal reward is the terminal game outcome. Reward wrappers can add shaping but are off by default.

`ZergActionWrapper` builds a stable ordered list of build, production, upgrade, resource, and combat macro actions. The current default prunes combat to three actions, matching the paper's reported experimental configuration. `use_all_combat_actions` in this repository adds only the ten whole-source-to-target-region actions; the 100 source-region/target-region Cartesian product described in the paper is commented out. This code-to-paper difference must be retained and documented rather than silently expanding the action space.

When masking is enabled, `_get_valid_action_mask()` calls each macro action's `is_valid` predicate. The no-op action is always valid, so the normal SC2 wrapper should never produce an all-zero mask. The mask is returned as the last tuple element **and** copied into the non-spatial feature vector. This duplication is part of the learned input contract.

`ZergObservationWrapper` builds counts and statistics for units/buildings, player and score features, worker features, recent-action one-hot history, optional game progress, optional region features, and the optional action mask. It can also build channel-first spatial maps, although the supplied PPO and DQN entry points do not exercise them. The policy implementations do not generally support arbitrary tuple observations: masked PPO assumes exactly `(features, mask)`, while its unmasked policy assumes a single array.

### PPO model architecture

`sc2learner/agents/ppo_policies.py` contains both TensorFlow policies.

The MLP has independent actor and critic towers. Each tower flattens the input and applies three 512-unit `tanh` layers. All hidden weights use the custom NumPy SVD orthogonal initializer with scale `sqrt(2)` and zero biases. The policy head uses orthogonal scale `0.01`; the scalar value head uses scale `1.0`. Invalid actions have `1e30` subtracted from their logits. Sampling uses the Gumbel-max construction. The distribution API returns negative log-probability, not log-probability.

The LSTM policy first applies two shared 512-unit `ReLU` fully connected layers, then a custom 512-unit recurrent cell. State layout is `[cell, hidden]`. At every time step, both halves are multiplied by `(1 - done)` before the update. Gate order and equations are:

```text
z = x @ Wx + h @ Wh + b
i, f, o, u = split(z)
c = sigmoid(f) * c + sigmoid(i) * tanh(u)
h = sigmoid(o) * tanh(c)
```

The recurrent head initialization scale is `1.0`, including the policy head; this differs from the MLP policy's `0.01` policy head. Batch layout is environment-major sequences: a flat `[nenv * nsteps, ...]` tensor is reshaped to `[nenv, nsteps, ...]`, iterated over time, then flattened back in the same order. This custom cell must be migrated directly; `torch.nn.LSTM` has different parameter layout and default gate ordering.

### PPO loss and optimization

`sc2learner/agents/ppo_agent.py::Model` creates a weight-sharing act model and train model. Its exact training behavior is:

- Advantages are recomputed as `returns - old_values`, then normalized over the entire sampled batch with population standard deviation plus `1e-8`.
- `ratio = exp(old_neglogp - new_neglogp)`.
- Policy loss is the mean maximum of unclipped and clipped negative surrogate losses.
- Approximate KL is `0.5 * mean((new_neglogp - old_neglogp)^2)`.
- Entropy is subtracted with coefficient `ent_coef`.
- Value loss is half mean-squared error. A clipped-value branch exists, but no caller enables it.
- Total loss is `policy_loss - ent_coef * entropy + vf_coef * value_loss`.
- Gradients are clipped by global norm when configured (all entry points use `0.5`).
- Adam uses epsilon `1e-5`; default training learning rate is `1e-5` and clip range is `0.1`.

The implementation does not perform the conventional multiple PPO epochs over one freshly collected rollout. Instead, actors continually send unrolls to a learner-side bounded deque, and each learner update randomly samples unroll fragments from that deque. This can introduce policy lag and data reuse. It is original repository behavior and must be preserved for parity until an explicitly separate algorithm experiment changes it.

### PPO rollout and GAE flow

Each actor repeatedly:

1. Requests the newest model parameter list from the learner.
2. Collects `unroll_length` steps (default 128), storing observation, action, predicted value, negative log-probability, episode-start `done`, and the initial recurrent state.
3. Resets the environment and recurrent state after terminal steps while allowing the rollout to span episode boundaries.
4. Bootstraps the final observation when nonterminal.
5. Computes reverse-time GAE with `gamma=0.998` and `lambda=0.95`, suppressing bootstrap and recursive advantage across terminal boundaries.
6. Sends `(obs, returns, dones, actions, values, neglogpacs, states, episode_infos)` via ZeroMQ/Python pickle.

For MLP, the learner splits each 128-step unroll into eight 16-step fragments, then samples `batch_size * 8` fragments so the model receives `batch_size * unroll_length` transitions. For LSTM, splitting is disabled and the learner samples `batch_size` complete sequences with their initial states.

Checkpoints are joblib dumps of an ordered list of TensorFlow variable arrays. They do not include optimizer state, update count, schedules, RNG state, metadata, observation/action schemas, or TensorFlow variable names. They are not directly compatible with PyTorch checkpoints.

### Distributed PPO and self-play

PPO uses two ZeroMQ channels:

- REQ/REP on port A (default 5700): actor requests the latest parameter snapshot.
- PUSH/PULL on port B (default 5701): actor sends rollout Python objects to the learner.

Threads bridge blocking sockets to small in-process queues. There is no protocol version, schema validation, model version attached to a rollout, authentication, retry policy, shutdown protocol, or atomic learner snapshot lock. `send_pyobj`/`recv_pyobj` uses pickle and is safe only on trusted networks.

Self-play maintains a bounded deque of historical ordered parameter lists. An actor may cache a newly received model with a configured probability and chooses either the latest model or a random cached model at episode boundaries. Initial opponents are loaded from a text file of joblib checkpoint paths.

### Existing PyTorch DQN

The following modules already use PyTorch and should be modernized only when required:

- `sc2learner/agents/dqn_networks.py`: spatial and non-spatial dueling Q networks.
- `sc2learner/agents/dqn_agent.py`: greedy/epsilon-greedy inference, Double DQN target selection, target network, Adam optimization, element-wise gradient clipping, checkpointing, actor, and learner.
- `sc2learner/bin/train_dqn.py` and the DQN branch of `evaluate.py`.

The DQN target mixes a one-step Double DQN bootstrap and a full-episode Monte Carlo return as `mmc_beta * one_step_target + (1 - mmc_beta) * mc_return`. Despite the parameter name, the default `mmc_beta=0.9` therefore emphasizes the one-step target. The dueling aggregation is `value + advantage - mean(advantage)`.

Legacy issues include unused `Variable` and `gym.spaces.prng` imports, `.data` access, implicit global CUDA selection, unguarded `param.grad.data`, scalar-shape-sensitive `squeeze()`, exact-type checks against Gym spaces, unsafe pickle transport, and checkpoints without optimizer/training metadata. DQN does not support action masking in the supplied path. Its per-parameter gradient clamp is not the same as PPO's global-norm clipping and must not be changed under the label of simple API modernization.

## TensorFlow dependency inventory and mapping

| Current dependency/construct | Location | PyTorch target |
| --- | --- | --- |
| TensorFlow import, placeholders, session, graph reuse | `ppo_agent.py`, `ppo_policies.py`, PPO entry points | Eager `nn.Module`, explicit tensors/device, one shared module for acting/training |
| `tf.layers.flatten`, `tf.matmul`, variables | `ppo_policies.py`, `utils_tf.py` | `torch.flatten`, `nn.Linear` or explicit parameters |
| Custom SVD orthogonal initializer | `utils_tf.py` | Faithful initializer with TensorFlow-shaped NumPy reference tests; do not assume `nn.init.orthogonal_` yields identical seeded arrays |
| Custom categorical distribution | `utils_tf.py` | `torch.distributions.Categorical`, with an explicit masked-logit utility and negative-log-probability compatibility surface |
| Gumbel-max sampling | `utils_tf.py` | `Categorical.sample()` for distributional behavior; retain an optional reference Gumbel sampler if seeded parity tests require it |
| Custom LSTM and sequence reshape helpers | `utils_tf.py` | Explicit recurrent cell/module preserving `[c,h]`, `i,f,o,u`, masks, and environment-major layout |
| PPO loss graph and `tf.gradients` | `ppo_agent.py` | Pure loss function + autograd + Adam + `clip_grad_norm_` |
| `tf.trainable_variables(scope=...)` parameter snapshots | `ppo_agent.py` | Named `state_dict`; wire format should use detached CPU tensors/arrays and a protocol version |
| `tf.global_variables_initializer` | `ppo_agent.py` | Constructor-time initialization without global side effects |
| joblib list checkpoints | `ppo_agent.py`, self-play | Versioned PyTorch checkpoint dictionary; optional, separately tested TF-list conversion tool |
| TensorFlow session configuration | PPO entry points | Explicit `--device`, CPU thread configuration, optional CUDA selection |

TensorFlow is imported only by `sc2learner/agents/ppo_agent.py`, `sc2learner/agents/ppo_policies.py`, `sc2learner/agents/utils_tf.py`, and the PPO setup paths in `train_ppo.py`, `train_ppo_selfplay.py`, `evaluate.py`, and `play_vs_ppo_agent.py`. No environment or DQN module needs TensorFlow.

## Known defects and behavior decisions

These findings are not fixes in Milestone 0. Each must receive a regression test and an explicit decision in its implementation milestone.

| Finding | Original behavior | Mathematical/engineering assessment | Proposed handling |
| --- | --- | --- | --- |
| `LstmPolicy.value()` feeds `DONE: mask`, but `mask` is undefined | Recurrent bootstrap/value calls raise `NameError` | Defect | Fix to use the supplied `done`; document as a bug fix and test it |
| LSTM value head returns `[batch, 1]`, while MLP returns `[batch]` and PPO placeholders/rollout logic expect vectors | Recurrent rollout/batching can retain an extra dimension and fail or broadcast unexpectedly | Defect | Standardize to `[batch]`; add shape and GAE integration tests |
| All-masked logits become equal finite values after subtracting `1e30` | Sampling becomes effectively uniform over invalid actions rather than failing | Unsafe edge behavior | Environment invariant says no-op is valid; validate masks and raise a descriptive error for all-zero rows unless a separately documented fallback is chosen |
| PPO evaluation always samples | No deterministic PPO evaluation mode exists | Missing capability, not an algorithm bug | Add opt-in argmax evaluation while keeping stochastic acting as default |
| MLP PPO uses replay-like random sampling from a bounded recent-unroll deque | Data may be reused and generated by stale policies | Diverges from common strictly on-policy PPO, but is deliberate repository behavior | Preserve first; expose metrics/model versioning later without silently changing sampling |
| Value clipping code exists but all callers leave it disabled | Unclipped value MSE is used | Deliberate current behavior | Default to disabled and test both branches |
| TensorFlow checkpoints are unnamed ordered array lists | Ordering is implicit and optimizer state is absent | Incompatible with robust PyTorch restore | New versioned format; optional converter maps by a documented ordered manifest and verifies output parity |
| `_safe_create_env`/`_safe_reset` swallow exceptions until the last retry | Early diagnostic context is lost | Reliability issue outside framework migration | Preserve runtime semantics initially; improve logging in environment work, separately from PPO parity |
| `_first_create` is never set false on the first reset path | The intended recreate-on-reset branch is unreachable | Likely defect | Record an environment compatibility test before changing it |
| DQN second epsilon segment divides by `eps_decay_steps2`, not the segment length | Epsilon does not reach exactly `0.01` at the second boundary | Likely schedule defect | Preserve until DQN milestone; test and document before offering correction |
| DQN clips each gradient element; PPO clips global norm | Algorithms have intentionally different clipping behavior in code | Neither should be conflated | Preserve both independently |
| ZeroMQ Python-object transport unpickles network input | Arbitrary-code risk on untrusted networks | Security risk | Restrict documented use to trusted networks; later use a validated versioned payload without redesigning topology |

## Proposed target architecture

The target should remain compact and recognizable:

```text
sc2learner/
  agents/
    ppo_policies.py     # PyTorch MLP and custom recurrent nn.Module policies
    ppo_agent.py        # rollout, GAE, loss/update, local agent, actor/learner
    ppo_types.py        # only if needed: small typed rollout/checkpoint contracts
    dqn_*.py            # retained, minimally modernized in Milestone 6
  envs/                 # semantics retained; legacy/modern API adapter isolated here
  bin/                  # same user-facing roles; explicit device and deterministic eval flags
tests/
  unit/                 # no SC2 installation required
  integration/          # mock env and loopback ZeroMQ
  sc2/                  # opt-in, asset-dependent tests
```

The policy module should accept NumPy-compatible observations but internally operate on tensors on an explicit device. It should expose logits, value, optional next recurrent state, stochastic sampling, deterministic selection, log/negative-log probability, and entropy. Mask validation should be centralized.

PPO math should be factored into small pure functions (`compute_gae`, normalized advantages, clipped surrogate loss, optional value clipping) so independent NumPy reference tests do not instantiate actors, sockets, or SC2. A local rollout collector/trainer should use the same functions and data contract as distributed training.

The distributed protocol should keep ZeroMQ and the actor/learner responsibilities. Parameter payloads should be CPU state dictionaries plus minimal metadata (`format_version`, model architecture/schema fingerprint, update/model version). Rollouts should retain the current logical fields and add model version without changing the learning algorithm. Serialization changes must have a one-actor/one-learner loopback test.

Checkpoints should contain at least model state, optimizer state, update count, configuration/hyperparameters, observation/action schema, format version, and optional RNG state. Loading weights only for evaluation should remain easy. Old TensorFlow joblib lists must never be accepted as if they were native PyTorch checkpoints.

## Dependency and environment strategy

### Recommended split

Use two clearly separated dependency surfaces:

1. **Core/test environment:** Python 3.11, a current supported PyTorch 2.x release, NumPy, pytest, and only lightweight runtime packages. It must run all PPO/DQN math, policy, checkpoint, mock-environment, and loopback communication tests without SC2 or TensorFlow.
2. **SC2 integration extra/environment:** the exact Tencent PySC2 compatibility source, `s2clientprotocol`, game assets, maps, and platform-specific requirements. Failures here must not block the framework-independent test suite.

Python 3.11 is the proposed baseline because it is installed locally, supported by current upstream PySC2 metadata, and within modern PyTorch's supported Python range. The provisional PyTorch candidate is the current stable 2.x CPU wheel, with an optional matching CUDA wheel. Milestone 1 must resolve and execute a version matrix before writing a durable lock; the old Tencent commit may force patches or a separate compatibility environment.

Do not replace the Tencent fork with upstream PySC2 merely because upstream installs on modern Python. This repository imports extension-specific `pysc2.lib.tech_tree`, `pysc2.lib.unit_controls`, raw protobuf units/actions, and historical enum names. The first SC2 compatibility candidate is Tencent commit `4f790218`; validate its data files and game version 4.6. If it cannot run on Python 3.11, maintain a narrowly patched fork or an isolated SC2 adapter process. Do not downgrade the core PyTorch environment or alter observation/action semantics without evidence.

Adopt Gymnasium only behind a compatibility boundary. Internally distinguish `terminated` from `truncated` so GAE bootstrapping is correct, but retain adapters for existing callers expecting a single `done`. Snapshot observation shapes, dtypes, feature order, action IDs/names, masks, and terminal behavior before changing wrappers.

Replace the static `setup.py` package list with modern metadata and package discovery only in Milestone 1. Use dependency groups/extras such as core, test, and SC2. TensorFlow should remain present only in an optional parity/conversion environment while reference comparisons are needed, and should be absent from the final runtime dependencies.

### Platform notes to validate

- Linux is the reference SC2 training platform and the only sensible target for large headless actor fleets.
- Windows should support the core/test suite and may support local SC2 if the Tencent fork and exact game binary work; this is unverified.
- CUDA is optional. CPU behavior is the correctness baseline; device placement must never depend on `torch.cuda.is_available()` alone.
- SC2 4.6, AbyssalReef assets, Tencent technology-tree data, and compatible protocol packages are a separate reproducibility bundle from Python/PyTorch dependencies.

## File-level migration plan

| File or area | Planned work | Milestone |
| --- | --- | --- |
| Packaging metadata (currently `setup.py`) | Define Python range, discover all packages, split core/test/SC2 dependencies, remove runtime TensorFlow after parity work | 1, finalized in 7 |
| `tests/` (new) | pytest configuration, synthetic spaces/observations, mock legacy and modern-step environments, deterministic seeds | 1 |
| `sc2learner/agents/ppo_policies.py` | Replace TF MLP with faithful PyTorch module; later add explicit custom recurrent cell | 2 (MLP), 5 (LSTM) |
| `sc2learner/agents/ppo_agent.py` | Replace TF model/loss/checkpoint path; first support local MLP, then distributed and recurrent/self-play paths | 2-6 |
| `sc2learner/agents/utils_tf.py` | Retain only while parity tests/conversion need it; replace utilities with PyTorch/framework-neutral code, then remove | 2, remove in 7 |
| `sc2learner/bin/evaluate.py` | Remove TF session setup for PPO; explicit device and optional deterministic policy | 3 |
| `sc2learner/bin/train_ppo.py` | Add local smoke path, then restore actor/learner PyTorch entry points without changing flags unnecessarily | 3-4 |
| `sc2learner/agents/replay_memory.py` | No algorithm rewrite; add loopback fixtures and validate payloads where shared transport requires it | 4/6 |
| `sc2learner/bin/train_ppo_selfplay.py` | Remove TF setup, migrate opponent checkpoint/cache flow | 6 |
| `sc2learner/bin/play_vs_ppo_agent.py` | Remove TF setup and load PyTorch evaluation checkpoint | 6 |
| `sc2learner/agents/dqn_networks.py` | Preserve networks; add shape and state-dict tests; only necessary modern API/device changes | 6 |
| `sc2learner/agents/dqn_agent.py` | Remove obsolete APIs/imports, explicit device, safe shape handling, versioned checkpoints/transport while preserving DDQN/MMC/replay behavior | 6 |
| `sc2learner/envs/**` | Snapshot semantics first; add API adapter and compatibility fixes separately from policy math | 1 and integration milestones |
| README and `docs/` | Setup, training/evaluation, checkpoint compatibility, platform limits, baseline configs, intentional differences | 7 |

## Prioritized implementation sequence

1. **Build a framework-independent safety net.** Establish packaging, pytest, synthetic masked discrete spaces, a mock environment, observation/action schema fixtures, and independent numerical references. Resolve the Tencent extension strategy without importing it in core tests.
2. **Migrate MLP PPO only.** Reproduce initializers, separate 3x512 actor/critic towers, masking, negative log-probabilities, losses, GAE, gradient clipping, and checkpoints. Compare TensorFlow and PyTorch outputs where the old stack can be made runnable; otherwise compare against frozen NumPy references.
3. **Prove local rollout-to-update behavior.** Run CPU smoke training and deterministic evaluation in a mock environment, with metrics and terminal/truncation tests.
4. **Restore distributed PPO.** Keep ZeroMQ topology, add versioned payloads and a loopback actor/learner test, and measure parameter/rollout synchronization.
5. **Migrate the custom LSTM.** Directly reproduce gate order, state `[c,h]`, sequence packing, reset masks, and boundary handling; do not substitute `nn.LSTM` without a proven conversion.
6. **Restore self-play and minimally modernize DQN.** Convert historical model caching/checkpoint management, then update DQN APIs without changing DDQN, MMC, replay, epsilon, or gradient-clipping semantics.
7. **Remove TensorFlow and finish integration documentation.** Only after reference paths and imports are gone.

This sequence deliberately defers distributed complexity and recurrence until the MLP math and local data path are independently testable.

## Validation strategy

### Framework-independent unit tests

- MLP actor/value output shapes and parameter counts for multiple observation/action sizes.
- Initializer properties (orthogonality, scale, zero biases) and, where possible, frozen cross-framework fixtures.
- Masked logits, probability renormalization, no invalid samples, mask dtype/shape checks, and explicit all-zero-mask behavior.
- Stochastic samples, deterministic argmax, log probability, negative log probability, and entropy against independent NumPy calculations.
- PPO policy loss, optional value clipping, total loss, KL estimate, and clip fraction on hand-computed tensors including positive and negative advantages.
- Advantage normalization, GAE, terminal boundaries, truncation bootstrapping, and rollout ending on both terminal and nonterminal states.
- Nonzero gradients, global-norm clipping, optimizer parameter updates, and no unintended critic-to-actor/actor-to-critic coupling in the MLP.
- Versioned checkpoint round-trip on CPU and CPU loading of a checkpoint saved from CUDA.
- Synthetic rollout collection, one or more optimizer updates, and measurable parameter change without SC2.

### Cross-framework parity tests

If a runnable TensorFlow 1.x reference environment can be isolated, export named weights and fixed observation/mask/action batches. Load transposed/mapped weights into PyTorch and compare logits, values, negative log-probabilities, entropy, PPO loss components, GAE, and recurrent transitions within documented tolerances. Sampling streams need not match because TensorFlow and PyTorch RNGs differ; compare distributions or fixed-action probabilities instead.

### Integration tests

- Local mock environment: collect complete and cross-boundary unrolls, train, save/load, and evaluate deterministically on CPU.
- Distributed loopback: one actor and one learner on ephemeral/local ports; exchange model version and one rollout; verify learner update and actor refresh; ensure clean shutdown.
- Recurrent: multiple environments/sequences, hidden-state carry, terminal reset in the middle of a sequence, and bootstrap value shape.
- Self-play: cache insertion/eviction, latest-versus-historical selection, and opponent checkpoint load.
- DQN: dueling outputs, Double DQN action selection/target evaluation, MMC target, replay sampling, target update cadence, epsilon schedule as implemented, and checkpoint round-trip.
- SC2 (opt-in): action IDs/names, mask, observation feature checksum/schema, reset/step signatures, short random/macro-agent episode, self-play reset, and game/version diagnostics.

Tests must report skips for missing SC2 assets rather than implying integration success.

## Key risks and mitigations

| Risk | Impact | Mitigation |
| --- | --- | --- |
| Tencent PySC2 commit is old and extension-specific | Modern Python install may fail; upstream replacement may silently change units/actions/tech tree | Pin the documented TStarBot1 commit, build an API/schema probe, patch narrowly or isolate it, and keep core tests independent |
| SC2/game/map/protocol version coupling | Launch failures or changed game semantics | Record exact game build, SC2 4.6 expectation, map checksum/location, protocol version, and platform in an opt-in diagnostic |
| Action ordering or feature ordering changes | Existing policies/checkpoints become meaningless | Golden schema fixtures for action names/IDs, feature lengths/order, mask location, dtype, and observation normalization |
| TensorFlow/PyTorch initialization and layout differences | Early learning dynamics and checkpoint conversion differ | Explicit initializers, named mapping manifest, fixed-weight parity tests, and documented tolerance |
| Masking numerical edge cases | Invalid actions sampled or NaNs | Boolean validation, stable masked logits, guaranteed-valid invariant, all-zero-mask test |
| Negative-log-probability sign mistakes | Incorrect PPO ratios/KL | Preserve names and formula; test with independently calculated fixed values |
| Terminal versus truncation handling | Biased GAE/value targets | Normalize environment step results into separate flags before computing bootstrap masks |
| Recurrent gate/state/order mismatch | LSTM checkpoint and behavior mismatch | Custom cell, direct gate equations, `[c,h]` state tests, sequence-layout tests, mid-sequence reset tests |
| Asynchronous stale-policy rollouts | Training behavior changes if batching/protocol is redesigned | Preserve deque/splitting/sampling behavior first; add model version metrics before considering policy-lag controls |
| Unsafe/unversioned ZeroMQ serialization | Security and compatibility failures | Trusted-network warning, schema/version fields, validated payload types/shapes, loopback tests |
| Legacy checkpoints lack metadata/optimizer state | Cannot truly resume training | Treat them as weight-only imports; create a richer native checkpoint and optional converter |
| Accidental DQN redesign | Baseline behavior changes during cleanup | Freeze numerical target and schedule tests before modernizing APIs |
| No installed local dependencies/assets | False confidence from static review | Milestone 1 must install a reproducible core environment and execute tests; SC2 results remain explicitly unverified until assets exist |

## Milestone completion criteria

### Milestone 0 (this audit)

- Every source and entry point is inventoried.
- TensorFlow and existing PyTorch ownership is identified.
- PPO observation-to-action, rollout, loss, optimization, checkpoint, evaluation, and distributed flows are documented.
- MLP/LSTM architecture, masking, recurrent state, DQN, dependency, and SC2 risks are recorded.
- A staged file-level plan and validation strategy exist.
- No implementation file was modified.

### Milestone 1 acceptance criteria

Milestone 1 is accepted only when all of the following are true:

1. A clean, documented Python 3.11 core environment installs on CPU with a selected supported PyTorch 2.x version and without TensorFlow or StarCraft II.
2. Dependency metadata discovers every `sc2learner` subpackage and provides separate test and SC2 dependency groups; a reproducible lock/constraints mechanism is committed for at least the core/test environment.
3. `pytest` runs from the repository root and has a CI-suitable command. The baseline includes import/collection tests and clearly marks asset-dependent SC2 tests.
4. A mock environment exercises both the legacy four-value step API and normalized terminated/truncated handling, masked and unmasked discrete actions, terminal reset, and synthetic observations with the tuple structure expected by PPO.
5. Golden fixtures or schema tests record action-mask shape/dtype/location, feature ordering/shape, action ordering/names, and no-op validity without launching SC2.
6. A dependency probe documents whether Tencent PySC2 commit `4f790218` installs and imports on Python 3.11 on the tested platform, including `tech_tree`, `unit_controls`, raw units/actions, enums, `s2clientprotocol`, and packaged data. If it fails, the exact incompatibilities and chosen narrow patch/isolation strategy are documented.
7. SC2 4.6 and AbyssalReef runtime requirements are documented separately from core PyTorch requirements. No SC2 execution is claimed unless an episode was actually launched.
8. No PPO or DQN algorithm or network implementation is changed. Any environment compatibility edits are adapter-only and covered by schema/behavior tests.
9. The selected Python/PyTorch/NumPy/Gym strategy is justified by an executed compatibility matrix, not only by package metadata.
10. The milestone report lists every command executed, its result, skips, and remaining platform limitations.

### Later milestone gates

- **Milestone 2:** all requested MLP PPO numerical, mask, GAE, gradient, checkpoint, and synthetic training tests pass on CPU; TensorFlow is absent from the new runtime path.
- **Milestone 3:** a complete local mock rollout/train/evaluate workflow passes, deterministic evaluation is verified, and required metrics are emitted.
- **Milestone 4:** one actor/one learner loopback integration passes with versioned parameter and rollout payloads and clean process/thread shutdown.
- **Milestone 5:** custom recurrent transition and sequence parity tests pass, including terminal resets and the documented legacy bug fixes.
- **Milestone 6:** self-play cache/opponent tests and DQN frozen-behavior tests pass; evaluation entry points load native PyTorch checkpoints.
- **Milestone 7:** runtime TensorFlow dependency and unused TF utilities are removed, CI passes without SC2, SC2 tests are clearly optional, and user documentation/baseline configurations are complete.

## Milestone 0 conclusion

The initial project description is accurate but not exhaustive. PPO's model, loss, checkpoint, distributed actor/learner, evaluation, LAN play, and self-play paths all still depend on TensorFlow 1.x. DQN is PyTorch already, but it and the packaging/environment stack require compatibility work. The highest-risk dependency is not PPO math; it is preserving the Tencent raw-unit/technology-tree environment contract on a modern Python stack. The highest-risk algorithmic areas are action-mask edge behavior, negative-log-probability signs, asynchronous rollout sampling, TensorFlow-shaped initialization/checkpoint mapping, and the custom recurrent state/gate/sequence conventions.

Milestone 1 should therefore establish a runnable core/test environment and freeze interfaces before any PPO implementation is changed.
