"""Fixed primary comparison, full spectrum and Novel interference diagnostics."""
import argparse
from pathlib import Path

import numpy as np

from coverage_ablation_hpc import ARMS, BINS, SCOPE, dump, load, read_plan, require, sha
from summarize_covisibility_repaired import exact_mcnemar, holm


def statistics(tasks):
    require(bool(tasks), 'empty prespecified analysis group')
    n=len(tasks); scenes=sorted({t['scene'] for t in tasks})
    result=dict(queries=n, histories=len({t['history_index'] for t in tasks}),
                scenes=len(scenes), arms={}, comparisons={})
    for arm in ARMS:
        rows=[t['arms'][arm] for t in tasks]
        result['arms'][arm]=dict(successes=sum(r['reached'] for r in rows),
            sr=float(np.mean([r['reached'] for r in rows])),
            spl=float(np.mean([r['spl'] for r in rows])),
            mean_steps=float(np.mean([r['steps'] for r in rows])))
    sampled=np.random.default_rng(20260910).integers(0,len(scenes),(20000,len(scenes)))
    for baseline in ('cec','raw_fixed','native'):
        effects=[int(t['arms']['cec_no_coverage']['reached'])-int(t['arms'][baseline]['reached']) for t in tasks]
        gain,loss=effects.count(1),effects.count(-1)
        totals=np.array([sum(e for t,e in zip(tasks,effects) if t['scene']==s) for s in scenes])
        counts=np.array([sum(t['scene']==s for t in tasks) for s in scenes])
        values=totals[sampled].sum(axis=1)/counts[sampled].sum(axis=1)
        result['comparisons'][baseline]=dict(gain=gain,loss=loss,risk_difference=sum(effects)/n,
            exact_mcnemar_p=exact_mcnemar(gain,loss),
            scene_cluster_bootstrap_ci95=np.quantile(values,[.025,.975]).tolist(),
            bootstrap_replicates=20000,bootstrap_seed=20260910)
    return result


def novel_diagnostics(tasks):
    result=statistics(tasks)
    result['takeover_queries']={arm:sum(t['takeover_plans'][arm]>0 for t in tasks)
                                for arm in ('cec','cec_no_coverage')}
    result['exact_native_queries']={arm:sum(t['no_takeover_exact_native'][arm] for t in tasks)
                                    for arm in ('cec','cec_no_coverage')}
    result['native_successes_lost']={arm:sum(t['arms']['native']['reached']==1 and
        t['arms'][arm]['reached']==0 for t in tasks) for arm in ('raw_fixed','cec','cec_no_coverage')}
    result['native_failures_rescued']={arm:sum(t['arms']['native']['reached']==0 and
        t['arms'][arm]['reached']==1 for t in tasks) for arm in ('raw_fixed','cec','cec_no_coverage')}
    return result


def summarize(plan_path, root, output):
    require(not output.exists(), 'do not overwrite an earlier summary')
    plan=read_plan(plan_path); rows=[]; missing=[]
    for task in plan['tasks']:
        folder=root/f'task_{task["task_index"]:03d}'
        path=folder/'archive_receipt.json'
        if not path.exists():
            missing.append(dict(task_index=task['task_index'],reason='archive missing'));continue
        archive=load(path)
        if not archive['completed']:
            missing.append(dict(task_index=task['task_index'],reason='task incomplete',exit_code=archive['exit_code']));continue
        verified=archive['verification']
        require(verified['verified'] and verified['task']==task and
                verified['plan_sha256']==sha(plan_path), 'verification belongs to another task/plan')
        require(archive['all_member_hashes_readback_verified'] and
                sha(archive['archive'])==archive['archive_sha256'], 'archive changed')
        require(verified==load(folder/'independent_verification.json'), 'verification copies differ')
        arms={r['arm']:r for r in verified['records']}
        require(len(verified['records'])==4 and set(arms)==set(ARMS), 'missing/duplicate arm')
        rows.append(dict(task,arms=arms,takeover_plans=verified['takeover_plans'],
                         no_takeover_exact_native=verified['no_takeover_exact_native']))
    if missing:
        dump(output,dict(complete=False,expected_queries=159,verified_queries=len(rows),
            missing_or_failed_tasks=missing,note='No complete-population SR; infrastructure attrition is not failure SR.'))
        raise RuntimeError(f'{len(missing)} incomplete tasks; see explicit missing list')
    support=[t for t in rows if t['analysis_role']=='revisit']
    low=[t for t in support if .1<=t['q_cec_frame8']<.5]
    high=[t for t in support if t['q_cec_frame8']>=.5]
    novel=[t for t in rows if t['analysis_role']=='novel']
    require((len(low),len(high),len(novel))==(40,91,28), 'analysis population changed')
    bins={b:statistics([t for t in support if t['analysis_bin']==b]) for b in BINS}
    for baseline in ('cec','raw_fixed','native'):
        adjusted=holm([bins[b]['comparisons'][baseline]['exact_mcnemar_p'] for b in BINS])
        for b,p in zip(BINS,adjusted):bins[b]['comparisons'][baseline]['holm_five_bins_p']=p
    result=dict(complete=True,scope=SCOPE,plan_sha256=sha(plan_path),query_count=159,query_arm_count=636,
        primary_low_support=statistics(low),high_support=statistics(high),cec_domain_bins=bins,
        natural_novel=novel_diagnostics(novel),
        natural_novel_without_original_boundary=novel_diagnostics([t for t in novel if t['history_index']!=22]),
        low_support_excluding_local_navigation_histories=statistics([t for t in low if not t['local_navigation_pilot_history']]),
        original_annotation_bins={b:statistics([t for t in support if t['original_bin']==b]) for b in BINS},
        paired_records=rows,summarizer_sha256=sha(__file__),
        caveat='Consumed development population; repeated goals clustered by scene; no pooled natural deployment SR or safety guarantee.')
    dump(output,result)
    print('SEALED 159 paired queries / 636 rollouts',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--evaluation-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    summarize(args.plan,args.evaluation_root,args.output)
