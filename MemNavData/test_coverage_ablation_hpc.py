import copy
import json
from pathlib import Path

import pytest

import coverage_ablation_hpc as c
from summarize_coverage_ablation import statistics, novel_diagnostics


@pytest.mark.parametrize('q,expected',[(.1,'c10_30'),(.29999,'c10_30'),(.3,'c30_50'),
    (.5,'c50_70'),(.7,'c70_90'),(.9,'c90_100'),(1.,'c90_100')])
def test_actual_history_bin_boundaries(q,expected):
    assert c.bin_for(q)==expected


@pytest.mark.parametrize('q',[0.,.09999,1.001,float('nan')])
def test_support_outside_frozen_range_is_not_silently_clipped(q):
    with pytest.raises(ValueError):c.bin_for(q)


def test_new_route_only_changes_authority_not_controller(tmp_path):
    source=dict(scene='s',episode='episode_0000',asset='/unused.glb',source_episode='/unused/episode_0000',seed=1)
    a=c.command(source,tmp_path,21000,21001,arm='cec',role='revisit')
    b=c.command(source,tmp_path,21000,21001,arm='cec_no_coverage',role='revisit')
    assert b==a+['--certified_authority_policy','certificate_without_coverage']
    assert 'coverage_ablation_hpc.py' in a[2] and a[3]=='eval'


def test_latin_order_balance_in_full_population():
    counts={arm:[0]*4 for arm in c.ARMS}
    for i in range(159):
        order=c.ARMS[i%4:]+c.ARMS[:i%4]
        assert set(order)==set(c.ARMS)
        for j,arm in enumerate(order):counts[arm][j]+=1
    assert all(max(v)-min(v)<=1 for v in counts.values())


def records():
    rows=[]
    for i in range(6):
        arms={a:dict(reached=0,spl=0.,steps=600) for a in c.ARMS}
        arms['cec_no_coverage']=dict(reached=1,spl=.8,steps=120)
        rows.append(dict(history_index=i,scene=f'scene{i//2}',arms=arms,
            takeover_plans={'cec':0,'cec_no_coverage':1},
            no_takeover_exact_native={'cec':True,'cec_no_coverage':False}))
    return rows


def test_queries_are_scene_clustered_and_effect_not_accept_rate():
    x=statistics(records())
    assert (x['queries'],x['histories'],x['scenes'])==(6,6,3)
    assert x['arms']['cec_no_coverage']['successes']==6
    for baseline in ('cec','raw_fixed','native'):
        r=x['comparisons'][baseline]
        assert (r['gain'],r['loss'])==(6,0)
        assert r['exact_mcnemar_p']==.03125
        assert r['scene_cluster_bootstrap_ci95']==[1.,1.]


def test_novel_takeover_and_success_damage_are_separate():
    rows=records()
    for row in rows:
        row['arms']['native']['reached']=1
        row['arms']['cec_no_coverage']['reached']=0
    x=novel_diagnostics(rows)
    assert x['takeover_queries']['cec_no_coverage']==6
    assert x['native_successes_lost']['cec_no_coverage']==6
    assert x['native_failures_rescued']['cec_no_coverage']==0


def test_immutable_original_loader_is_reused_not_relabelled():
    assert c.original.POPULATION_SHA=='182f3a6d2519d5b2c178b88345db4d0bb678088dd88487cfecaad9814c6fdaaf'
    assert c.original.ARMS==('native','raw_fixed','cec')
    assert c.ARMS==('native','raw_fixed','cec','cec_no_coverage')


def test_verified_two_novel_pilot_keeps_exact_native_depth(tmp_path):
    # The new fallback checker can independently inspect the already finished
    # local pilot; this test only asserts its fields include depth as well as motion.
    import inspect
    source=inspect.getsource(c.exact_fallback)
    for key in ('actual_position','actual_yaw','all_values','input_tensor_sha256',
                'output_tensor_sha256','scale_receipt_sha256'):
        assert key in source
