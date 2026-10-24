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
| PowerShell stopped on Gym's deprecation text | Windows PowerShell turns native `stderr` piped through `2>&1` into error records; the scripts used `ErrorActionPreference=Stop` | Allow native stderr through logging only while Python runs, then decide success from `$LASTEXITCODE` | Random/PPO scripts retain warnings in their logs without treating them as process failures |
| DQN actor/learner could not form a bounded smoke test | Both loops and their ZeroMQ worker threads were designed for permanent cluster jobs | Added optional finite episode/update limits and explicit replay/model socket shutdown; defaults remain unlimited | One real episode delivered 512 replay transitions, learner updated once, saved and cleanly exited |
| DQN smoke script reported an empty actor exit code after successful completion | Windows PowerShell can expose `$null` from an exited `Start-Process` object with redirected streams; `$null -ne 0` caused a false failure | Prefer the numeric exit code, with a fallback requiring fresh completion markers, no traceback, and a checkpoint newer than the run start | The affected run was confirmed successful from both logs and its refreshed checkpoint |

The reusable Tencent patch is stored in
`patches/pysc2-tencent-py311.patch`. It reverse-applies cleanly against the
validated vendor working tree.

No reward definition, macro-action meaning, PPO loss, policy network, or DDQN
algorithm was changed during Windows compatibility work.

Warnings about legacy Gym and `pkg_resources` are upstream deprecation
warnings. They did not affect the real-game run and were not hidden.
