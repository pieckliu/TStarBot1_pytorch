"""Probe the Tencent PySC2 API required by TStarBot1 without launching SC2."""

import importlib
import importlib.metadata
import json
import platform
import sys
from pathlib import Path


REQUIRED_MODULES = (
    "pysc2.env.sc2_env",
    "pysc2.env.lan_sc2_env",
    "pysc2.lib.point",
    "pysc2.lib.tech_tree",
    "pysc2.lib.typeenums",
    "pysc2.lib.unit_controls",
    "s2clientprotocol.sc2api_pb2",
)


def _version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def probe():
    result = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "distributions": {
            name: _version(name) for name in ("PySC2", "s2clientprotocol", "protobuf")
        },
        "imports": {},
        "pysc2_root": None,
        "api_checks": {},
    }
    for module_name in REQUIRED_MODULES:
        try:
            module = importlib.import_module(module_name)
            result["imports"][module_name] = "ok"
            if module_name == "pysc2.lib.tech_tree":
                result["pysc2_root"] = str(Path(module.__file__).resolve().parents[1])
        except Exception as exc:
            result["imports"][module_name] = f"{type(exc).__name__}: {exc}"
    if result["imports"].get("pysc2.lib.tech_tree") == "ok":
        try:
            from pysc2.lib.tech_tree import TechTree

            tree = TechTree()
            tree.update_version("4.6")
            result["api_checks"]["tech_tree_4_6"] = {
                "unit_types": len(tree.m_unitTypeData),
                "upgrades": len(tree.m_upgradeData),
            }
        except Exception as exc:
            result["api_checks"]["tech_tree_4_6"] = f"{type(exc).__name__}: {exc}"
    if result["imports"].get("pysc2.lib.typeenums") == "ok":
        try:
            from pysc2.lib.typeenums import ABILITY_ID, UNIT_TYPEID, UPGRADE_ID

            result["api_checks"]["historical_enums"] = all(
                (
                    hasattr(UNIT_TYPEID, "ZERG_DRONE"),
                    hasattr(UPGRADE_ID, "BURROW"),
                    hasattr(ABILITY_ID, "BUILD_HATCHERY"),
                )
            )
        except Exception as exc:
            result["api_checks"]["historical_enums"] = f"{type(exc).__name__}: {exc}"
    return result


def main():
    result = probe()
    print(json.dumps(result, indent=2, sort_keys=True))
    imports_ok = all(value == "ok" for value in result["imports"].values())
    checks_ok = all(
        value is True or isinstance(value, dict)
        for value in result["api_checks"].values()
    )
    return 0 if imports_ok and checks_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
