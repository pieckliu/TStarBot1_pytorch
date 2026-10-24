# Debugging log

## Repository replacement and isolation

The previous TensorFlow-oriented checkout was archived to
`C:\linux_project\TStarBot1-dev-open-pre-pytorch-20261024.zip`, then replaced
with the requested PyTorch fork. A new Python 3.11 environment named
`tstarbot1_pytorch_env` was created. The older local virtual environment was
removed only after the replacement passed all tests. TStarBot2 and BurnySC2
were not modified.

## Compatibility findings and fixes

| Symptom | Cause | Minimal fix | Verification |
|---|---|---|---|
| Tencent PySC2 import failed on Python 3.11 | `random.shuffle` no longer accepts a custom RNG argument | Reimplemented the historical deterministic Fisher-Yates shuffle in `pysc2/lib/colors.py` | Compatibility probe imports all required modules |
| NumPy rejected `np.stack(generator)` | NumPy 2 requires a materialized sequence | Converted the two generators to lists in `pysc2/lib/features.py` | Real observation reset/step succeeded |
| `enum34` attempted to install on modern Python | Unconditional historical dependency | Added a Python `<3.4` environment marker in Tencent `setup.py` | Editable install and `pip check` pass |
| `np.int` attribute error | Alias removed from modern NumPy | Replaced it with built-in `int` in `placer.py` | Building placement ran in real SC2 |
| Action-mask dtype mismatch | Historical default float dtype was implicit | Made the mask explicitly `np.float32` | Runtime tuple is `(float32[857], float32[62])` |
| Original actor/learner runs forever | Appropriate for training but unsuitable for a bounded smoke test | Added optional `max_actor_unrolls` and `max_learner_updates`; defaults remain unlimited | Actor produced one rollout and learner performed one update, then both exited |

The reusable Tencent patch is stored in
`patches/pysc2-tencent-py311.patch`. It reverse-applies cleanly against the
validated vendor working tree.

No reward definition, macro-action meaning, PPO loss, policy network, or DDQN
algorithm was changed during Windows compatibility work.

Warnings about legacy Gym and `pkg_resources` are upstream deprecation
warnings. They did not affect the real-game run and were not hidden.
