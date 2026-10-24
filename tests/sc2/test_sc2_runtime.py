"""Opt-in checks for the external Tencent PySC2 installation."""

import os

import pytest


pytestmark = pytest.mark.sc2


def test_sc2_runtime_is_explicitly_opt_in():
    if os.environ.get("TSTARBOT_RUN_SC2_TESTS") != "1":
        pytest.skip("set TSTARBOT_RUN_SC2_TESTS=1 after installing SC2 assets")
    from tools.probe_sc2_compat import probe

    result = probe()
    assert result["imports"]["pysc2.env.sc2_env"] == "ok"
    assert result["imports"]["pysc2.lib.tech_tree"] == "ok"
    assert result["imports"]["pysc2.lib.unit_controls"] == "ok"
    assert result["api_checks"]["tech_tree_4_6"]["unit_types"] > 0
    assert result["api_checks"]["historical_enums"] is True
