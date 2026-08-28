#!/usr/bin/env python3
"""Resolve a reject-policy CLI contract without changing hub semantics.

The frozen HM3D source bundle predates the ``--reject-policy`` command-line
option.  Its hub nevertheless hard-codes the exact policy used by this
experiment: ``shared_native_exact`` with the mono-NavDP fallback.  This helper
allows an overlay runner to omit only that redundant option, and only after an
AST audit proves the legacy semantics.  Any other legacy policy fails closed.
"""

from __future__ import annotations

import argparse
import ast
from pathlib import Path


EXPLICIT_CLI = "explicit_cli"
LEGACY_SHARED_NATIVE_EXACT = "legacy_shared_native_exact"


def _call_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _literal_string(node: ast.AST | None) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def inspect_hub_cli(hub_script: Path) -> str:
    """Return the proved reject-policy CLI mode for *hub_script*.

    A legacy hub is accepted only when one ``ComparisonPlan`` construction
    literally seals both the shared-native policy and the NavDP fallback.
    Merely finding those strings elsewhere in the file is insufficient.
    """

    tree = ast.parse(hub_script.read_text(), filename=str(hub_script))
    explicit = False
    legacy_exact = False
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if _call_name(node.func) == "add_argument":
            explicit = explicit or any(
                _literal_string(argument) == "--reject-policy"
                for argument in node.args
            )
        if _call_name(node.func) != "ComparisonPlan":
            continue
        keywords = {
            keyword.arg: _literal_string(keyword.value)
            for keyword in node.keywords
            if keyword.arg is not None
        }
        legacy_exact = legacy_exact or (
            keywords.get("reject_policy") == "shared_native_exact"
            and keywords.get("fallback_controller") == "navdp"
        )
    if explicit:
        return EXPLICIT_CLI
    if legacy_exact:
        return LEGACY_SHARED_NATIVE_EXACT
    raise RuntimeError(
        "hub has neither an explicit --reject-policy CLI nor a proved "
        "legacy shared_native_exact/NavDP ComparisonPlan"
    )


def resolve_hub_cli(hub_script: Path, reject_policy: str) -> str:
    """Resolve the CLI mode while enforcing the requested policy."""

    if reject_policy not in {"shared_native_exact", "controller_native_exact"}:
        raise ValueError(f"unsupported reject policy: {reject_policy}")
    mode = inspect_hub_cli(hub_script)
    if mode == LEGACY_SHARED_NATIVE_EXACT and reject_policy != "shared_native_exact":
        raise RuntimeError(
            "legacy hub is sealed to shared_native_exact and cannot satisfy "
            f"requested policy {reject_policy}"
        )
    return mode


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hub-script", type=Path, required=True)
    parser.add_argument(
        "--reject-policy",
        choices=("shared_native_exact", "controller_native_exact"),
        required=True,
    )
    args = parser.parse_args()
    print(resolve_hub_cli(args.hub_script, args.reject_policy))


if __name__ == "__main__":
    main()
