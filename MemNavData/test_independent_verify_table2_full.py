from copy import deepcopy
import math

import pytest

from MemNavData.independent_verify_table2_full import measure, independent_statistics, compare_statistics


def example():
    points=[[0,0,0],[1,0,0],[2.5,0,0]]
    terminal=dict(steps=2,end_position=points[-1],goal_xz_evaluator_only=[3,0],
        reached=1,geodesic_m=2,actual_path_len_m=2.5,final_goal_dist_m=.5,spl=.8)
    trace=dict(reached=1,end_position=points[-1],poses=[dict(step=i,x=p[0],y=p[1],z=p[2]) for i,p in enumerate(points[:-1])])
    actions=[dict(action_index=i,selected_mode="bounded_standard",position_before=a,actual_position=b)
             for i,(a,b) in enumerate(zip(points,points[1:]))]
    return terminal,trace,actions


def test_final_action_is_included_and_commanded_distance_is_not_used():
    terminal,trace,actions=example()
    terminal['commanded_path_len_m']=900
    result=measure(terminal,trace,actions)
    assert result['actual_path_len_m']==2.5 and result['spl']==.8


def test_missing_final_endpoint_does_not_silently_produce_an_spl():
    terminal,trace,actions=example()
    del terminal['end_position']
    with pytest.raises(KeyError):
        measure(terminal,trace,actions)


def test_last_action_and_terminal_endpoint_must_agree():
    terminal,trace,actions=example()
    actions[-1]['actual_position']=[2.4,0,0]
    with pytest.raises(ValueError,match='Numerical mismatch'):
        measure(terminal,trace,actions)


def test_missing_or_duplicate_action_is_detected():
    terminal,trace,actions=example()
    with pytest.raises(ValueError,match='Missing action'):
        measure(terminal,trace,actions[:-1])
    actions[1]['action_index']=0
    with pytest.raises(ValueError,match='action index'):
        measure(terminal,trace,actions)


def test_zero_progress_does_not_mean_success():
    terminal,trace,actions=example()
    terminal.update(reached=0,end_position=[0,0,0],actual_path_len_m=0,final_goal_dist_m=3,spl=0)
    trace.update(reached=0,end_position=[0,0,0])
    for p in trace['poses']:
        p['x']=0
    for a in actions:
        a.update(position_before=[0,0,0],actual_position=[0,0,0])
    assert measure(terminal,trace,actions)['reached']==0


def test_success_boundary_uses_frozen_strict_less_than_one_meter():
    terminal,trace,actions=example()
    terminal.update(goal_xz_evaluator_only=[3.5,0],final_goal_dist_m=1,reached=0,spl=0)
    trace['reached']=0
    assert measure(terminal,trace,actions)['spl']==0


def test_independent_statistics_match_producer_without_reusing_its_formula():
    from MemNavData.summarize_table1_repaired import statistics
    rows=[]
    outcomes=[(0,1),(0,1),(0,1),(1,0),(1,1),(0,0),(0,1),(1,1)]
    for i,(n,g) in enumerate(outcomes):
        rows.append(dict(scene=f's{i//2}',native=dict(reached=n,spl=n*.5),
            cec=dict(reached=g,spl=g*.7),pair=dict(cec_takeover=True,no_takeover_exact_native=False)))
    calculated=independent_statistics(rows)
    assert calculated['gain']==4 and calculated['loss']==1
    assert calculated['exact_mcnemar_p']==.375
    compare_statistics(calculated,statistics(rows))
    assert independent_statistics([]) is None


def test_display_count_cannot_be_changed_by_one_without_detection():
    from MemNavData.summarize_table1_repaired import statistics
    rows=[dict(scene='a',native=dict(reached=0,spl=0),cec=dict(reached=1,spl=.8),
               pair=dict(cec_takeover=True,no_takeover_exact_native=False))]
    expected=statistics(rows)
    expected['arms']['native']['successes']=1
    with pytest.raises(ValueError,match='Numerical mismatch'):
        compare_statistics(independent_statistics(rows),expected)
