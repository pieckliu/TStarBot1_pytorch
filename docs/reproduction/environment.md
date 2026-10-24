# Validated Windows environment

Validation date: 2026-10-24. Working directory:
`C:\linux_project\TStarBot1-dev-open`.

| Component | Validated value |
|---|---|
| Repository | `ramsayxiaoshao/TStarBot1_pytorch`, `dev-open` at `9c25ec19366e68640b228d32b87000b14d501f7c` |
| Python | CPython 3.11.9, isolated in `tstarbot1_pytorch_env` |
| PyTorch | 2.14.1+cpu |
| TensorFlow | Not installed or required by this PyTorch fork |
| Gymnasium | 1.4.0, for the migrated RL core |
| legacy Gym | 0.26.2, for Tencent PySC2 wrappers |
| NumPy | 2.4.6 |
| protobuf | 3.20.3 |
| s2clientprotocol | 4.6.0.67926.0 |
| Tencent PySC2 Extension | commit `4f790218e8c560a6c49df01b49708609f1d223e9`, locally patched |
| SC2 executable data | 5.0.16.97563 (`Base97563`) |
| TStarBot TechTree selection | 4.6 (`--game_version=4.6`) |
| Map | `Maps\Ladder2017Season4\AbyssalReefLE.SC2Map` |

`game_version=4.6` selects Tencent TechTree JSON/API data used to validate
build, production, and upgrade requirements. It does not select or downgrade
the installed SC2 executable. The existing TStarBot2 runtime supplies the game
binary, data, and map through `SC2PATH`; none of its files were modified.

The pinned package sets are in `requirements/constraints-py311.txt` and
`requirements/constraints-sc2-py311.txt`. Run
`scripts/check_environment.ps1` to check Python packages, imports, TechTree,
SC2 path, and the map.

CUDA was intentionally not installed for this correctness baseline. A CUDA
wheel suitable for the RTX 5090 can be tested later in another environment;
doing so is not necessary for environment, protocol, or PPO correctness.
