#!/usr/bin/env python3
"""Dependency-free serialization of CEC authority diagnostics."""

from __future__ import annotations

from typing import Any, Mapping


AUTHORITY_PLAN_RECEIPT_FIELDS = (
    "certified_relocalization_authority",
    "certified_relocalization_authority_policy",
)


def authority_plan_receipt_fields(
        response: Mapping[str, Any]) -> dict[str, Any]:
    """Copy authority evidence into a plan receipt without changing control."""

    if not isinstance(response, Mapping):
        raise TypeError("authority receipt source must be a mapping")
    return {
        field: response.get(field)
        for field in AUTHORITY_PLAN_RECEIPT_FIELDS
    }


__all__ = [
    "AUTHORITY_PLAN_RECEIPT_FIELDS",
    "authority_plan_receipt_fields",
]
