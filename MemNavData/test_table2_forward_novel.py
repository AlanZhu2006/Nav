from copy import deepcopy
import hashlib
import math
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from MemNavData.table2_balanced_sampling import balanced_assignment, spatial_menu
from MemNavData.table2_formal_population import assign_c_queries
from MemNavData.table2_mixed_hpc import formal, protocol_for, validate_plan
from MemNavData.table2_mixed_local import runtime_spec
from MemNavData.table2_sampling_profiles import (
    ALL_CELLS, BALANCED_SCHEMA, FORWARD_CELLS, FORWARD_SCHEMA,
    role_cells, verify_query_direction,
)


def candidate(cell):
    return dict(cell=cell, query=f"/{cell}.json", query_sha256="sealed")


def test_only_novel_cells_change_and_v2_remains_unchanged():
    assert role_cells(FORWARD_SCHEMA, "novel") == FORWARD_CELLS
    assert role_cells(FORWARD_SCHEMA, "revisit") == ALL_CELLS
    assert role_cells(BALANCED_SCHEMA, "novel") == ALL_CELLS
    assert role_cells(BALANCED_SCHEMA, "revisit") == ALL_CELLS


@pytest.mark.parametrize("angle", [-60., 0., 60.])
@pytest.mark.parametrize("stage", [0, 1, 2])
def test_all_novel_stages_accept_forward_route(angle, stage):
    verify_query_direction(dict(analysis_role="novel", cell="2_to_4/front",
        initial_relative_route_angle_deg=angle, stage_number=stage), FORWARD_SCHEMA)


@pytest.mark.parametrize("angle", [-180., -60.01, 60.01, 180., float("nan")])
def test_front_label_cannot_hide_nonforward_numeric_bearing(angle):
    with pytest.raises(ValueError, match="initial route"):
        verify_query_direction(dict(analysis_role="novel", cell="2_to_4/front",
            initial_relative_route_angle_deg=angle), FORWARD_SCHEMA)


def test_revisit_can_turn_back_but_novel_cannot_substitute_rear_goal():
    query = dict(analysis_role="revisit", cell="6_to_9/rear", initial_relative_route_angle_deg=170.)
    verify_query_direction(query, FORWARD_SCHEMA)
    query["analysis_role"] = "novel"
    with pytest.raises(ValueError, match="outside"):
        verify_query_direction(query, FORWARD_SCHEMA)


def test_distance_balance_uses_three_forward_cells_and_retains_empty_sources():
    menus = [dict(source_id=str(i), candidates=[candidate(c) for c in FORWARD_CELLS]) for i in range(12)]
    menus.append(dict(source_id="no_forward_target", candidates=[]))
    result = balanced_assignment(menus, key="B/novel", cells=FORWARD_CELLS)
    assert set(result["counts"].values()) == {4}
    assert result["empty_sources"] == ["no_forward_target"]
    assert len(result["selected"]) == 12 and result["exact_equal_cells"]
    with pytest.raises(ValueError, match="declared cell"):
        balanced_assignment([dict(source_id="wrong", candidates=[candidate("2_to_4/rear")])],
                            key="B/novel", cells=FORWARD_CELLS)


def test_spatial_constructor_filters_before_visual_checks_without_changing_yaw(monkeypatch):
    import MemNavData.table2_balanced_sampling as sampling
    monkeypatch.setattr(sampling, "SPATIAL_ATTEMPTS", 3)
    monkeypatch.setattr(sampling, "shortest", lambda pf, start, pos: (3., np.stack([start,pos])))
    monkeypatch.setitem(sys.modules, "MemNavData.generate_twoleg", SimpleNamespace(
        first_path_yaw=lambda points, start: math.atan2(-points[-1,0], -points[-1,2])))

    class Pathfinder:
        def __init__(self):
            self.points = iter(([0.,0.,3.], [3.,0.,0.], [0.,0.,-3.]))
        def seed(self, seed):
            self.seed_value = seed
        def get_random_navigable_point(self):
            return next(self.points)
        def is_navigable(self, point):
            return True
        def distance_to_closest_obstacle(self, point):
            return 1.

    pf = Pathfinder()
    result = spatial_menu(pf, np.zeros(3), 0., "recorded-end-yaw", cells=FORWARD_CELLS)
    assert [x["cell"] for x in result["candidates"]] == ["2_to_4/front"]
    assert result["counts"]["outside_role_direction"] == 2
    assert result["candidates"][0]["initial_relative_route_angle_deg"] == 0.


def test_c_source_mix_stays_half_native_novel_half_native_revisit():
    candidates = [dict(source_id=f"A{i}", collector="native", b_role=role,
        b_reached=True, both_c_constructed=True, b_task_index=i,
        menus={"novel":[candidate(c) for c in FORWARD_CELLS],
               "revisit":[candidate("2_to_4/rear")]})
        for i,role in enumerate(("novel", "revisit", "revisit"))]
    before = deepcopy(candidates)
    selected, assignments = assign_c_queries(candidates, schema=FORWARD_SCHEMA)
    assert [x["b_task_index"] for x in selected] == [0,1]
    assert [x["b_role"] for x in selected] == ["novel","revisit"]
    assert all(x["cell"].endswith("/front") for x in assignments["novel"]["selected"])
    assert all(x["cell"].endswith("/rear") for x in assignments["revisit"]["selected"])
    assert candidates == before


def test_forward_schema_is_explicit_and_mismatched_source_is_rejected():
    p = dict(schema=FORWARD_SCHEMA, formal_result=True,
        sources=[dict(index=0,scene="s",episode="e",source_id="s/e",initial_state=dict(schema=FORWARD_SCHEMA))],
        work_root="/tmp/memnav_table2_forward_test", arms=["native","cec"],
        c_selection="global_equal_native_B_roles")
    assert formal(validate_plan(p))
    assert protocol_for(p).name == "TABLE2_FORWARD_NOVEL_PROTOCOL_20260911.md"
    p["sources"][0]["initial_state"]["schema"] = BALANCED_SCHEMA
    with pytest.raises(ValueError, match="construction profile"):
        validate_plan(p)


def test_direction_constraint_is_not_a_runtime_guidance_input():
    keys = ("schema", "scene", "episode", "asset", "seed", "stage_number", "start_position", "start_yaw",
        "camera_height_m", "camera_intrinsic", "goal_rgb", "goal_rgb_sha256", "floor_position", "yaw_rad",
        "geodesic_m", "prefix_root", "prefix_trace_sha256")
    q = dict.fromkeys(keys, None)
    q.update(schema=FORWARD_SCHEMA, analysis_role="novel", direction_stratum="front",
             cell="2_to_4/front", initial_relative_route_angle_deg=10.)
    assert set(runtime_spec(q)) == set(keys)


def test_forward_bundle_inherits_runtime_instead_of_dirty_workspace(tmp_path, monkeypatch):
    from MemNavData import prepare_table2_mixed_bundle as packaging
    baseline, workspace, destination = (tmp_path / name for name in ("baseline", "workspace", "new"))
    relative = Path("NavDP/policy.py")
    (baseline / relative).parent.mkdir(parents=True)
    (baseline / relative).write_bytes(b"frozen policy\n")
    (workspace / relative).parent.mkdir(parents=True)
    (workspace / relative).write_bytes(b"unrelated new policy\n")
    digest = hashlib.sha256((baseline / relative).read_bytes()).hexdigest()
    (baseline / "SOURCE_BUNDLE.sha256").write_text(f"{digest}  {relative.as_posix()}\n")
    for name in packaging.FORWARD_OVERLAY:
        file = workspace / "MemNavData" / name
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text("direction-only test fixture\n")
    monkeypatch.setattr(packaging, "ROOT", workspace)
    packaging.prepare_forward(destination, baseline)
    assert (destination / relative).read_bytes() == b"frozen policy\n"
    assert (baseline / relative).read_bytes() == b"frozen policy\n"
    assert not (destination / "SOURCE_BUNDLE.sha256").exists()
    assert (destination / "FORWARD_OVERLAY.json").is_file()
    (baseline / relative).write_bytes(b"invalid baseline\n")
    with pytest.raises(ValueError, match="baseline bundle"):
        packaging.prepare_forward(tmp_path / "bad", baseline)
