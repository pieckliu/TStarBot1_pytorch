# TStarBot1 PyTorch Windows reproduction report

## Outcome

All completion criteria C1-C10 are satisfied for the requested PyTorch fork.
This includes a real full-game environment run and a distinct real-SC2 PPO
actor/learner update; neither result used a mock environment.

| Criterion | Result |
|---|---|
| C1 isolated environment | PASS - CPython 3.11.9 `tstarbot1_pytorch_env` |
| C2 core imports | PASS - package, PyTorch, Tencent PySC2, protocol and TechTree |
| C3 real SC2 launch | PASS - Base97563, Zerg vs Zerg on Abyssal Reef |
| C4 complete game | PASS - two random games completed (+1 win, then 0 draw) |
| C5 runtime spaces | PASS - 62 actions; float32 857-vector plus float32 62-mask |
| C6 macro actions | PASS - all major categories appeared in the real log |
| C7 reward/done | PASS - terminal +1 and normal done observed |
| C8 PPO update | PASS - one rollout transmitted and one optimizer update completed |
| C9 reproducible scripts | PASS - five requested PowerShell entry points |
| C10 preserve other work | PASS - TStarBot2 used read-only; BurnySC2 untouched |

## Test evidence

- Core tests: 51 passed, 2 sandbox-only ZeroMQ skips, 1 deselected SC2 test.
- Distributed tests outside the restricted socket sandbox: 4 passed.
- Local synthetic PPO: 10 updates, checkpoint reload, deterministic return 8.
- Real random games: two completed without abnormal termination; the first
  returned +1 after 577 decisions and the scripted rerun returned 0 after 899.
- Real distributed PPO: actor completed one 8-step unroll; learner completed one
  update and saved a 23,554,647-byte checkpoint.
- Checkpoint evaluation: reload succeeded and a full 899-decision game ended in
  a draw; evaluator exit code 0.
- Final `pip check`: no broken requirements. No residual `SC2_x64` process.

Evidence is in both `results/pytorch_smoke/` and `results/smoke_tests/`.
The latter contains the final scripted random-agent log and PPO actor/learner
logs plus `ppo/ppo_1.pt`; the former retains the checkpoint reload evaluation.

## Source changes

- `.gitignore`: excludes local environments, results, and vendor checkout.
- `README.md`: links this validated Windows reproduction.
- `sc2learner/bin/train_ppo.py`: optional finite smoke-test limits only.
- `sc2learner/envs/actions/placer.py`: modern NumPy integer compatibility.
- `sc2learner/envs/actions/zerg_action_wrappers.py`: explicit float32 mask.
- `patches/pysc2-tencent-py311.patch`: reproducible Python 3.11/NumPy 2 vendor fixes.
- `scripts/*.ps1`: environment, random, macro, PPO smoke, and evaluation runners.
- `docs/reproduction/*`: environment and behavior documentation.

No compatibility edit changes the intended action registry, reward definition,
PPO objective, or policy behavior. The repository itself is the user-requested
PyTorch port, so TensorFlow is absent by design.

## Next research step

A future TPG controller can replace only `PPOAgent.act`: accept the same
857-feature vector (and 62-bit validity mask), select a valid discrete action,
and leave `ZergActionWrapper`, persistent scripted execution, rewards, and
episode lifecycle unchanged. First freeze and version the feature/action schema,
then compare TPG and PPO on identical seeds and unshaped outcomes. No TPG code
was added in this task.
