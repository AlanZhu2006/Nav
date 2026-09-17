"""Reduce the complete fixed twelve-episode continuous integration cohort."""
import argparse
import hashlib
import json
from pathlib import Path


def load(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def reduce(plan_path, root):
    plan = load(plan_path)
    expected = {(c['cell']['index'], m) for c in plan['cells'] for m in plan['modes']}
    indexed, receipts = {}, {}
    for folder in sorted((root / 'tasks').iterdir()):
        if not folder.is_dir():
            continue
        result = load(folder / 'independent_verification.json')
        key = result['index'], result['mode']
        assert key in expected and key not in indexed
        assert result['verified'] and result['plan_sha256'] == sha(plan_path)
        assert result['verifier_sha256'] == sha(Path(__file__).with_name('verify_gem_memory_continuous.py'))
        assert all(sha(p) == h for p, h in result['artifact_sha256'].items())
        indexed[key] = result
        receipts[str(folder / 'independent_verification.json')] = sha(folder / 'independent_verification.json')
    assert set(indexed) == expected, 'Full continuous cohort is not yet verified'
    groups = {}
    for mode in plan['modes']:
        rows = [indexed[(c['cell']['index'], mode)] for c in plan['cells']]
        stages = []
        for index in range(3):
            issued = [r['per_leg'][index] for r in rows if len(r['per_leg']) > index]
            success = sum(r['reached'] for r in issued)
            stages.append(dict(stage=index, goal=plan['cells'][0]['goals'][index]['name'],
                initial_n=len(rows), attempted=len(issued), successes=success,
                not_attempted=len(rows) - len(issued), cumulative_sr=success / len(rows),
                conditional_sr=success / len(issued) if issued else None,
                mean_spl_issued=sum(r['spl'] for r in issued) / len(issued) if issued else None))
        groups[mode] = dict(initial_n=len(rows), complete_chains=sum(r['joint_success'] for r in rows),
            mean_goals_completed=sum(r['goals_completed'] for r in rows) / len(rows), stages=stages,
            continuous_frames=[r['continuous_frames'] for r in rows])
    paired = []
    for entry in plan['cells']:
        index = entry['cell']['index']
        paired.append(dict(index=index, scene=entry['cell']['scene'],
            arms={m:dict(goals_completed=indexed[(index, m)]['goals_completed'],
                joint_success=indexed[(index, m)]['joint_success'],
                per_leg=indexed[(index, m)]['per_leg']) for m in plan['modes']}))
    result = dict(complete=True, verified_episodes=len(indexed),
        actual_goal_legs=sum(len(r['per_leg']) for r in indexed.values()),
        source_scope=plan['scope'], plan_sha256=sha(plan_path), groups=groups, pairs=paired,
        verification_sha256=receipts, reducer_sha256=sha(__file__),
        interpretation='All prior local integration histories, fixed first-view goal cycle. Descriptive integration evidence; no held-out or population generalization claim.')
    destination = root / 'independent_reduction.json'
    encoded = json.dumps(result, indent=2, allow_nan=False) + '\n'
    if destination.exists():
        assert destination.read_text() == encoded
    else:
        destination.write_text(encoded)
    print(json.dumps({k: result[k] for k in ('complete', 'verified_episodes', 'actual_goal_legs')}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--plan', type=Path, required=True)
    args = parser.parse_args()
    reduce(args.plan.resolve(), args.root.resolve())
