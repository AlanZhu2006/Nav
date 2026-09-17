from collections import Counter
from copy import deepcopy
import itertools
from pathlib import Path

import pytest

from MemNavData.table2_balanced_sampling import (
    CELLS, SCHEMA, balanced_assignment, cell_of, novel_supported_as_task,
)
from MemNavData.table2_formal_population import assign_c_queries
from MemNavData.table2_mixed_local import base_source, runtime_spec
from MemNavData.table2_mixed_hpc import validate_plan, formal, protocol_for


def menu(identity, cells):
    return dict(source_id=identity, candidates=[dict(cell=c, query=f"/{identity}/{c}.json",
                query_sha256="immutable") for c in cells])


def test_global_matching_reassigns_flexible_source_to_preserve_scarce_cell():
    a, b = CELLS[:2]
    result = balanced_assignment([menu("flexible", [a,b]), menu("only-a", [a])], key="B/novel")
    assert Counter(r["cell"] for r in result["selected"]) == {a:1,b:1}
    assert {r["source_id"]:r["cell"] for r in result["selected"]}["only-a"] == a


def test_population_balance_is_exact_when_supply_allows_it():
    result = balanced_assignment([menu(str(i), CELLS) for i in range(36)], key="A/novel")
    assert set(result["counts"].values()) == {4}
    assert result["exact_equal_cells"] and not result["navigation_outcomes_read"]


def test_assignment_is_optimal_for_small_exhaustive_case():
    options = [CELLS[:2], CELLS[1:4], CELLS[2:3], CELLS[:4], CELLS[:1]]
    result = balanced_assignment([menu(str(i), c) for i,c in enumerate(options)], key="C/novel")
    score = sum(x*x for x in result["counts"].values())
    assert score == min(sum(v*v for v in Counter(x).values()) for x in itertools.product(*options))


def test_missing_cells_and_empty_sources_are_reported_not_filled():
    rows = [menu("a", [CELLS[0]]), menu("b", [CELLS[0]]), menu("empty", [])]
    original = deepcopy(rows)
    r = balanced_assignment(rows, key="B/revisit")
    assert r["empty_sources"] == ["empty"] and len(r["selected"]) == 2
    assert not r["exact_equal_cells"] and rows == original
    assert r == balanced_assignment(rows, key="B/revisit")


def test_empty_image_is_not_novel_and_initial_visibility_is_common_to_a_b_c():
    assert not novel_supported_as_task(0, [], 0)
    assert not novel_supported_as_task(100, [], .10)
    assert not novel_supported_as_task(100, [.02,.03], .20)
    assert not novel_supported_as_task(100, [.10,.01], 0)
    assert novel_supported_as_task(100, [], .09)
    assert novel_supported_as_task(100, [.01,.09], .09)
    with pytest.raises(ValueError):
        novel_supported_as_task(100, [float("nan")], 0)


@pytest.mark.parametrize("value,expected", [(2,"2_to_4"),(3.999,"2_to_4"),(4,"4_to_6"),(6,"6_to_9"),(9,"6_to_9")])
def test_distance_edges(value, expected):
    assert cell_of(dict(geodesic_m=value,direction_stratum="front")) == expected+"/front"


def test_c_mix_keeps_branch_isolation_and_native_only():
    sources = []
    for i, role in enumerate(("novel", "revisit", "revisit")):
        sources.append(dict(source_id=f"different-A-{i}", collector="native", b_role=role,
            b_reached=True, both_c_constructed=True, b_task_index=i,
            menus={r:menu(str(i),CELLS[:2])["candidates"] for r in ("novel","revisit")}))
    original = deepcopy(sources)
    chosen, assignments = assign_c_queries(sources)
    assert [r["b_task_index"] for r in chosen] == [0,1]
    assert Counter(r["b_role"] for r in chosen) == {"novel":1,"revisit":1}
    assert all(set(r["queries"]) == {"novel","revisit"} for r in chosen)
    assert sources == original and not assignments["novel"]["navigation_outcomes_read"]
    sources[0]["collector"] = "cec"
    with pytest.raises(ValueError):
        assign_c_queries(sources)


def test_declarative_start_needs_no_expert_parquet_or_memory():
    state = dict(scene="s", episode="episode_0000", seed=1, asset="s.glb",
        prefix_root=None, prefix_trace_sha256=None, start_position=[0,0,0])
    source = dict(scene="s", episode="episode_0000", seed=1, asset="s.glb", initial_state=state)
    assert base_source(source, 0) == state
    source["initial_state"]["prefix_root"] = "expert_history"
    with pytest.raises(ValueError):
        base_source(source,0)


def test_formal_uses_new_protocol_without_changing_old_pilot_schema():
    p = dict(schema=SCHEMA, formal_result=True,
        sources=[dict(index=0,scene="s",episode="episode_0000",source_id="s/episode_0000")],
        work_root="/tmp/memnav_table2_full_v1",arms=["native","cec"],
        c_selection="global_equal_native_B_roles")
    assert formal(validate_plan(p))
    assert protocol_for(p).name == "TABLE2_FULL_RERUN_PROTOCOL_20260911.md"
    p["formal_result"] = False
    with pytest.raises(ValueError):
        validate_plan(p)


def test_role_and_gt_support_do_not_become_policy_role_inputs():
    text = Path(__file__).with_name("table2_mixed_local.py").read_text()
    start = text.index("def runtime_spec(")
    end = text.index("def base_source(",start)
    function = text[start:end]
    assert '"analysis_role"' not in function
    assert '"covis_curve"' not in function
    assert '"direction_stratum"' not in function


def test_query_cli_does_not_require_an_expert_source_directory(tmp_path):
    from MemNavData.table2_mixed_local import query_command
    source = dict(scene="new_scene", episode="episode_0000", seed=13, asset="/scene.glb")
    command = query_command(source, dict(seed=13), tmp_path/"native", 21500, 21501, "native")
    assert "--role_pair_query_role" not in command
    assert command[command.index("--episode_root")+1] == str(tmp_path/"new_scene")
    assert "--exec_horizon" in command and command[command.index("--exec_horizon")+1] == "8"
