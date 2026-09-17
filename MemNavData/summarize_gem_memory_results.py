"""Combine complete, verified GEM evidence without selecting favorable cases."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

MODES=('legacy','native_interval7','connected_reciprocal')


def load(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def describe(values):
    a=np.asarray(values,dtype=float)
    if not a.size:return dict(n=0,mean=None,median=None,p95=None,min=None,max=None)
    assert np.isfinite(a).all()
    return dict(n=len(a),mean=float(a.mean()),median=float(np.median(a)),
        p95=float(np.quantile(a,.95)),min=float(a.min()),max=float(a.max()))


def summarize(root):
    root=Path(root).resolve()
    paths={
        'navigation':root/'hpc_003/final/independent_reduction.json',
        'transfer':root/'hpc_003/final/transfer_verification.json',
        'export':root/'hpc_003/final/export_receipt.json',
        'diagnosis':root/'hpc_003/final/first_query_diagnosis.json',
        'population':root/'hpc_003/final/population_description.json',
        'resources':root/'resources_001/independent_reduction.json',
        'local':root/'independent_reduction.json',
        'local_plan':root/'navigation_local_plan.json',
        'continuous':root/'continuous_002/independent_reduction.json',
        'api':root/'api_final_main_002/result.json',
        'gpu':root/'api_guard_gpu_002/result.json',
        'development_readout':root/'readout_001/independent_verification.json',
    }
    data={k:load(p) for k,p in paths.items()}
    assert data['navigation']['complete'] and data['navigation']['verified_tasks']==210
    assert data['navigation']['expected_rollouts']==420 and data['navigation']['paired_histories']==70
    assert data['transfer']['verified'] and data['transfer']['rows']==420
    assert all(sha(root/'hpc_003/final'/p)==h for p,h in data['export']['artifact_sha256'].items())
    assert all(sha(root/'hpc_003/final'/p)==h for p,h in data['transfer']['extra_sha256'].items())
    assert data['resources']['complete'] and data['resources']['verified_tasks']==18
    assert data['local']['complete'] and data['local']['verified_tasks']==12
    assert data['local']['plan_sha256']==sha(paths['local_plan'])
    assert data['continuous']['complete'] and data['continuous']['verified_episodes']==12
    assert all(sha(p)==h for p,h in data['continuous']['verification_sha256'].items())
    assert data['api']['passed'] and data['api']['cpu_tests']==61
    assert data['gpu']['passed'] and data['gpu']['next_writes_match_after_goal_switches']
    assert sha(paths['gpu'])==data['api']['gpu_result_sha256']
    project=Path(__file__).resolve().parents[1]
    assert all(sha(project/p)==h['delivered_sha256'] for p,h in data['api']['core_files'].items())
    assert data['development_readout']['verified']
    assert data['diagnosis']['complete'] and data['diagnosis']['rows']==420
    assert data['population']['plan_sha256']==data['navigation']['plan_sha256']==data['diagnosis']['plan_sha256']
    records=data['diagnosis']['records']
    indexed={(r['index'],r['role'],r['mode']):r for r in records}
    assert len(indexed)==420
    population=data['population']['rows']
    assert len(population)==70
    result=dict(complete=True,navigation=data['navigation']['groups'],
        resources=data['resources']['groups'],population={},initial_queries={},
        local_integration=data['local']['groups'],development_complete_revisit={},
        continuous_integration=data['continuous'],
        regressions_against_current_gem=[],improvements_against_current_gem=[])
    for dataset in ('all','hm3d','mp3d'):
        cells=[r for r in population if dataset=='all' or r['dataset']==dataset]
        result['population'][dataset]=dict(histories=len(cells),
            scenes=len({(c['dataset'],c['scene']) for c in cells}),
            history_frames=describe([c['history_frames'] for c in cells]),
            design_overlap_histories=sum(c['candidate_design_scene'] for c in cells),
            geodesic_m={role:describe([c['query_geodesic_m'][role] for c in cells]) for role in ('novel','revisit')})
        for role in ('novel','revisit'):
            paired=[{m:indexed[(c['index'],role,m)] for m in MODES} for c in cells]
            joint=[p for p in paired if all(p[m]['first_query_bearing_error_deg'] is not None for m in MODES)]
            result['initial_queries'][f'{dataset}/{role}']=dict(n=len(paired),jointly_defined=len(joint),
                arms={m:dict(first_query_accepted=sum(p[m]['first_query_accepted'] for p in paired),
                    first_query_takeover=sum(p[m]['first_query_takeover'] for p in paired),
                    any_navigation_takeover=sum(p[m]['takeover_plans']>0 for p in paired),
                    joint_first_query_bearing_deg=describe([p[m]['first_query_bearing_error_deg'] for p in joint])) for m in MODES})
    for index in range(70):
        for role in ('novel','revisit'):
            a={m:indexed[(index,role,m)] for m in MODES}
            if a['connected_reciprocal']['reached']<a['legacy']['reached']:
                result['regressions_against_current_gem'].append(a)
            if a['connected_reciprocal']['reached']>a['legacy']['reached']:
                result['improvements_against_current_gem'].append(a)
    development=[r for r in data['development_readout']['rows'] if r['role']=='revisit' and r['complete_history']]
    joint=[r for r in development if all(r['bearing_deg'][m] is not None for m in ('native','camera','reciprocal'))]
    result['development_complete_revisit']=dict(n=len(development),jointly_defined=len(joint),
        arms={m:describe([r['bearing_deg'][m] for r in joint]) for m in ('native','camera','reciprocal')},
        scope='Correlated development geometry; not extra independent navigation trials')
    result['source_sha256']={k:dict(path=str(p),sha256=sha(p)) for k,p in paths.items()}
    result['script_sha256']=sha(__file__)
    result['interpretation']='Complete descriptive synthesis; scientific claims still require human assessment of scope, uncertainty and prior work'
    destination=root/'complete_evidence_summary.json'
    temporary=destination.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');temporary.replace(destination)
    print(json.dumps(dict(complete=True,navigation_rollouts=420,local_rollouts=24,resource_processes=18,
        continuous_episodes=12,continuous_goal_legs=data['continuous']['actual_goal_legs'],
        regressions=len(result['regressions_against_current_gem']),improvements=len(result['improvements_against_current_gem']))))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('root',type=Path)
    summarize(parser.parse_args().root)
