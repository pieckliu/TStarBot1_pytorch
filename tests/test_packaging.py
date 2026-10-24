import ast
from pathlib import Path

from setuptools import find_packages


def test_every_source_package_is_discovered():
    root = Path(__file__).resolve().parents[1]
    expected = {
        str(path.parent.relative_to(root)).replace("\\", ".").replace("/", ".")
        for path in (root / "sc2learner").rglob("__init__.py")
    }
    discovered = set(find_packages(where=root, include=["sc2learner*"]))
    assert discovered == expected


def test_tensorflow_is_not_a_core_dependency():
    pyproject = (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text(
        encoding="utf-8"
    )
    dependency_block = pyproject.split("[project.optional-dependencies]", 1)[0]
    assert "tensorflow" not in dependency_block.lower()


def test_runtime_paths_do_not_import_tensorflow():
    root = Path(__file__).resolve().parents[1]
    migrated_paths = (
        "sc2learner/agents/ppo_agent.py",
        "sc2learner/agents/ppo_policies.py",
        "sc2learner/agents/ppo_local.py",
        "sc2learner/bin/train_ppo_local.py",
        "sc2learner/bin/evaluate.py",
        "sc2learner/bin/train_ppo.py",
        "sc2learner/bin/train_ppo_selfplay.py",
        "sc2learner/bin/play_vs_ppo_agent.py",
    )
    for relative_path in migrated_paths:
        source = (root / relative_path).read_text(encoding="utf-8")
        tree = ast.parse(source)
        imported_modules = {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        }
        imported_modules.update(
            node.module
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module is not None
        )
        assert not any(
            module == "tensorflow" or module.startswith("tensorflow.")
            for module in imported_modules
        ), relative_path


def test_obsolete_tensorflow_utilities_are_removed():
    root = Path(__file__).resolve().parents[1]
    assert not (root / "sc2learner/agents/utils_tf.py").exists()
