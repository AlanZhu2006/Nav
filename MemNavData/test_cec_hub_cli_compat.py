from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from cec_hub_cli_compat import (
    EXPLICIT_CLI,
    LEGACY_SHARED_NATIVE_EXACT,
    inspect_hub_cli,
    resolve_hub_cli,
)


def _write(tmp_path: Path, source: str) -> Path:
    path = tmp_path / "hub.py"
    path.write_text(source)
    return path


def test_explicit_cli_accepts_both_frozen_policies(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        """
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--reject-policy")
""",
    )
    assert inspect_hub_cli(path) == EXPLICIT_CLI
    assert resolve_hub_cli(path, "shared_native_exact") == EXPLICIT_CLI
    assert resolve_hub_cli(path, "controller_native_exact") == EXPLICIT_CLI


def test_legacy_exact_contract_allows_only_redundant_shared_policy(
    tmp_path: Path,
) -> None:
    path = _write(
        tmp_path,
        """
plan = ComparisonPlan(
    controller="navdp",
    reject_policy="shared_native_exact",
    fallback_controller="navdp",
)
""",
    )
    assert inspect_hub_cli(path) == LEGACY_SHARED_NATIVE_EXACT
    assert (
        resolve_hub_cli(path, "shared_native_exact")
        == LEGACY_SHARED_NATIVE_EXACT
    )
    with pytest.raises(RuntimeError, match="cannot satisfy requested policy"):
        resolve_hub_cli(path, "controller_native_exact")


@pytest.mark.parametrize(
    "source",
    [
        "plan = ComparisonPlan(reject_policy='shared_native_exact')\n",
        "plan = ComparisonPlan(fallback_controller='navdp')\n",
        "reject_policy = 'shared_native_exact'\nfallback_controller = 'navdp'\n",
    ],
)
def test_unproved_legacy_contract_fails_closed(
    tmp_path: Path, source: str,
) -> None:
    path = _write(tmp_path, source)
    with pytest.raises(RuntimeError, match="neither an explicit"):
        inspect_hub_cli(path)


def test_repository_hub_uses_explicit_cli() -> None:
    root = Path(__file__).resolve().parent
    hub = root / "cec_controller_portability_hub.py"
    if not hub.is_file():
        pytest.skip("minimal runtime overlay intentionally uses the frozen task hub")
    assert inspect_hub_cli(hub) == EXPLICIT_CLI


def test_command_line_emits_one_auditable_mode(tmp_path: Path) -> None:
    hub = _write(
        tmp_path,
        "plan = ComparisonPlan(reject_policy='shared_native_exact', "
        "fallback_controller='navdp')\n",
    )
    helper = Path(__file__).resolve().parent / "cec_hub_cli_compat.py"
    completed = subprocess.run(
        [
            sys.executable,
            str(helper),
            "--hub-script",
            str(hub),
            "--reject-policy",
            "shared_native_exact",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert completed.stdout == LEGACY_SHARED_NATIVE_EXACT + "\n"
