# Milestone 1 report: environment and dependency preparation

Audit date: 2026-10-23

## Outcome

Milestone 1 establishes a reproducible Python 3.11 core/test environment and a
separate, tested compatibility recipe for the historical Tencent PySC2 fork.
No PPO or DQN algorithm/network implementation was changed.

## Files changed

- `pyproject.toml` and the `setup.py` compatibility shim: modern package
  discovery, Python range, core/test/SC2 dependency groups, pytest settings.
- `requirements/constraints-py311.txt`: exact core/test environment.
- `requirements/constraints-sc2-py311.txt`: exact import-tested Tencent PySC2
  environment.
- `sc2learner/envs/api_compat.py`: framework-neutral reset/step result adapter.
- `tests/support/mock_env.py`: masked/unmasked synthetic macro-action env with
  legacy and modern APIs and terminal/truncation variants.
- `tests/`: dependency, package discovery, schema, mask, seed, reset, and step
  contract tests plus an opt-in SC2 test.
- `tools/probe_sc2_compat.py`: import and TechTree/API compatibility probe.
- `patches/pysc2-tencent-py311.patch`: narrow Python 3.11 compatibility patch.
- `.github/workflows/core-tests.yml`: SC2-free Linux CI command.
- `docs/environment.md`: CPU, optional SC2, Windows, and Linux setup guidance.

## Executed compatibility matrix

| Surface | Versions | Platform | Result |
| --- | --- | --- | --- |
| Core/test | CPython 3.11.9, PyTorch 2.14.1+cpu, NumPy 2.4.6, Gymnasium 1.4.0, pytest 9.1.1 | Windows 10/11 build 26200 | 23 tests passed; `pip check` passed |
| SC2 unconstrained | Tencent PySC2 `4f790218`, s2clientprotocol 5.0.17, protobuf 7.36.2 | Same | Install succeeded; generated protobuf imports failed |
| SC2 protocol-pinned | Same fork, s2clientprotocol 4.6.0.67926.0, protobuf 3.20.3 | Same | Protocol, TechTree, enums and unit-control imports passed; `sc2_env` hit removed Python 3.11 `random.shuffle` argument |
| SC2 patched | Same pinned environment plus repository patch | Same | All probe imports passed; TechTree 4.6 initialized with 118 unit types and 85 upgrades |
| Existing TStarBot1 env imports | Patched SC2 environment plus transitional Gym 0.26.2 | Same | Raw, LAN, self-play, action-wrapper and observation-wrapper imports passed |

Gym 0.26.2 emits its upstream unmaintained/NumPy 2 warning. It is confined to
the SC2 compatibility extra. New framework-independent code uses Gymnasium.

## Commands executed

```text
python -m pip index versions <dependency>
python -m venv .venv
.venv/Scripts/python -m pip install -e ".[test]" -c requirements/constraints-py311.txt
.venv/Scripts/python -m pytest -m "not sc2"
.venv/Scripts/python -m pytest -m sc2
.venv/Scripts/python -m pip check
git clone --no-checkout https://github.com/Tencent/PySC2TencentExtension.git ...
git checkout 4f790218
python -m compileall -q <Tencent pysc2 package>
python -m venv .venv-sc2-probe
.venv-sc2-probe/Scripts/python -m pip install -e <Tencent checkout>
.venv-sc2-probe/Scripts/python -m tools.probe_sc2_compat
.venv-sc2-probe/Scripts/python -m pip install --force-reinstall protobuf==3.20.3 s2clientprotocol==4.6.0.67926.0
git apply patches/pysc2-tencent-py311.patch
.venv-sc2-probe/Scripts/python -m tools.probe_sc2_compat
git diff --check
```

The SC2-marked pytest was skipped as designed because
`TSTARBOT_RUN_SC2_TESTS` was not enabled and game assets were not verified.
One final editable-metadata rebuild initially failed with Windows sandbox
`WinError 5` in pip's external temporary directory. Re-running that same
non-network build outside the filesystem sandbox succeeded; the resulting
editable wheel was installed and the full test suite was rerun afterward.

## Preserved contracts

- Masked observations remain `(features, mask)`.
- The mask remains duplicated at the end of the synthetic non-spatial feature
  vector, matching the real wrapper contract.
- Action zero is a valid no-op, preventing an all-zero mask in normal use.
- Legacy four-value and modern five-value step results normalize to separate
  `terminated` and `truncated` fields.
- Truncations permit value bootstrapping; true terminals do not.
- Existing SC2 wrappers and all PPO/DQN files are unchanged.

## Limitations and remaining risks

- StarCraft II was not launched. SC2 4.6 binaries, AbyssalReef, complete reset/
  step behavior, raw protocol exchange, and a full episode remain untested.
- The Tencent import recipe was executed on Windows only. Linux is still the
  intended headless training platform and needs an asset-backed verification.
- Gym 0.26.2 is transitional. Replacing wrapper inheritance/imports with
  Gymnasium must be a separately tested environment change.
- NumPy 2 removes legacy aliases such as `np.bool`; the known uses are in PPO
  code scheduled for migration, not changed in this milestone.
- The SC2 patch must be applied to the external fork; the ignored probe clone
  is not part of the repository deliverable.
- CUDA was not tested. CPU remains the correctness baseline.

## Acceptance status

The Milestone 1 core acceptance criteria are met: installable Python 3.11 CPU
environment, complete package discovery, exact constraints, CI-suitable tests,
mock legacy/modern environments, schema/mask tests, an executed Tencent fork
probe with a documented narrow patch, and explicit SC2 skips/limitations.
Asset-backed SC2 execution is intentionally still pending availability of the
game and maps; no claim of game-runtime validation is made.
