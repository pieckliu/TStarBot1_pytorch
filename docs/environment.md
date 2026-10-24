# Environment setup

## Core development environment

The framework-independent environment is intentionally separate from the
StarCraft II runtime. It targets CPython 3.11 and works on CPU.

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .venv/Scripts/Activate.ps1
python -m pip install -e ".[test]" -c requirements/constraints-py311.txt
python -m pytest -m "not sc2"
```

The validated dependency set is recorded in
`requirements/constraints-py311.txt`. CUDA is optional and is not required for
the correctness suite. Install a platform-appropriate PyTorch wheel only after
the CPU suite passes.

TensorFlow and StarCraft II are not dependencies of this core environment.

## StarCraft II integration environment

TStarBot1 cannot be switched to current upstream PySC2 without changing its
runtime contract. It depends on Tencent-specific modules such as
`pysc2.lib.tech_tree` and `pysc2.lib.unit_controls`, raw protobuf actions, and
the extension's packaged technology-tree data.

The Tencent extension README directs TStarBot1 users to commit `4f790218`
(2019-05-15) and requires an editable install:

```bash
git clone https://github.com/Tencent/PySC2TencentExtension.git
cd PySC2TencentExtension
git checkout 4f790218
git apply /path/to/TStarBot1_pytorch/patches/pysc2-tencent-py311.patch
python -m pip install -e . \
  -c /path/to/TStarBot1_pytorch/requirements/constraints-sc2-py311.txt
```

After installation, run the import/data probe without launching the game:

```bash
python -m tools.probe_sc2_compat
```

Only enable SC2 tests after the extension, StarCraft II, version 4.6 game data,
and the AbyssalReef map are available:

```bash
TSTARBOT_RUN_SC2_TESTS=1 python -m pytest -m sc2
```

On PowerShell, set the variable with
`$env:TSTARBOT_RUN_SC2_TESTS = "1"`. Upstream PySC2 documents Windows support,
but the historical Tencent commit remains unverified there. Linux is the
reference platform for headless actor processes.

The old Gym API returns one value from `reset()` and four from `step()`. New
code should pass results through `sc2learner.envs.api_compat`, which preserves
tuple observations while distinguishing termination from time-limit
truncation. Existing SC2 wrappers are not changed in Milestone 1.

The Python 3.11 import probe found two concrete incompatibilities in an
unconstrained installation:

1. Current protobuf rejects the old generated protocol descriptors. Pinning
   `protobuf==3.20.3` and the game-aligned
   `s2clientprotocol==4.6.0.67926.0` resolves this.
2. Python 3.11 removed the custom-random-function argument from
   `random.shuffle`. The repository patch reproduces the old constant-0.5
   Fisher-Yates operation exactly.

After those constraints and the patch, all imports exercised by
`tools.probe_sc2_compat` passed on Windows/Python 3.11. This validates imports
and packaged TechTree access only. It does not validate launching SC2, map
availability, a complete episode, or Linux behavior.
