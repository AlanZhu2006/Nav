"""Evaluator-only task profiles; no direction label reaches the policy."""
from __future__ import annotations

import math
from pathlib import Path

BALANCED_SCHEMA = "table2_balanced_actual_mono_20260911_v2"
FORWARD_SCHEMA = "table2_forward_novel_actual_mono_20260911_v3"
FORMAL_SCHEMAS = (BALANCED_SCHEMA, FORWARD_SCHEMA)
BINS = ("2_to_4", "4_to_6", "6_to_9")
ALL_CELLS = tuple(f"{b}/{s}" for b in BINS for s in ("front", "side", "rear"))
FORWARD_CELLS = tuple(f"{b}/front" for b in BINS)


def protocol_path(schema):
    names = {
        BALANCED_SCHEMA: "TABLE2_FULL_RERUN_PROTOCOL_20260911.md",
        FORWARD_SCHEMA: "TABLE2_FORWARD_NOVEL_PROTOCOL_20260911.md",
    }
    return Path(__file__).with_name(names[schema])


def role_cells(schema, role):
    if schema not in FORMAL_SCHEMAS or role not in ("novel", "revisit"):
        raise ValueError("Unknown Table-II construction profile or role")
    return FORWARD_CELLS if schema == FORWARD_SCHEMA and role == "novel" else ALL_CELLS


def verify_query_direction(query, schema):
    """Validate the measured initial route direction, not just its text label.

    This controls target construction, not intermediate robot actions. The
    robot keeps its actual start yaw; the shortest-path bearing is not a
    runtime guidance input.
    """
    if query["cell"] not in role_cells(schema, query["analysis_role"]):
        raise ValueError("Query direction is outside the declared role profile")
    if schema == FORWARD_SCHEMA and query["analysis_role"] == "novel":
        angle = float(query["initial_relative_route_angle_deg"])
        if not math.isfinite(angle) or abs(angle) > 60.0 + 1e-9:
            raise ValueError("Novel requires an initial route direction within +/-60 degrees")
