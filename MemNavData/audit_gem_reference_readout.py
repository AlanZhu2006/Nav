"""Exercise the new binding adapter with every saved real sparse query.

No geometry/matching is rerun and no evaluator truth is loaded. Common gauge
changes and node-specific corrections are algebraic interface checks, not
evidence that any estimated correction improves navigation.
"""
import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'MemNavData')]
from MemNavData.gem_reference_readout import ReferenceReadoutSession, pose_matrix
from NavDP.baselines.memnav.gem.bindings import EpisodicBindings


def load(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def sim3(scale, angle, translation):
    c, s = np.cos(angle), np.sin(angle)
    value = np.eye(4)
    value[:3, :3] = scale * np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    value[:3, 3] = translation
    return value


def expect_value_error(call):
    try:
        call()
    except ValueError:
        return
    raise AssertionError('The invalid read was accepted')


def audit(root, out, wait_for):
    original = root / 'readout_001'
    inputs = [original / n for n in ('manifest.json', 'completion.json',
        'inference.json', 'independent_verification.json')]
    sources = [Path(__file__), ROOT / 'MemNavData/gem_reference_readout.py',
        ROOT / 'NavDP/baselines/memnav/gem/bindings.py',
        ROOT / 'MemNavData/certified_relocalization_runtime.py']
    hashes = {str(p):sha(p) for p in inputs + sources}
    out.mkdir(parents=True, exist_ok=False)
    save(out / 'manifest.json', dict(schema='gem_reference_readout_actual_evidence_v1',
        source_sha256=hashes, pid=os.getpid(), wait_for=None if wait_for is None else str(wait_for),
        scope='Every453original arm/query result; coordinate-interface invariants only; no new matches, truth or navigation',
        counterfactuals='CommonSim3 gauge and fixed current-nodeSim3 correction; never scored as accuracy gains',
        started_at=time.time()))
    try:
        if wait_for is not None:
            while not wait_for.exists():
                time.sleep(5)
        assert all(sha(p) == h for p, h in hashes.items())
        assert load(original / 'completion.json')['completed']
        prior_audit = load(original / 'independent_verification.json')
        assert prior_audit['verified']
        assert all(sha(p) == h for p, h in prior_audit['source_sha256'].items())
        manifest = load(original / 'manifest.json')
        parent = Path(load(Path(manifest['native']) / 'manifest.json')['parent'])
        rows = load(original / 'inference.json')['rows']
        assert len(rows) == 151
        geometry_hashes, results = {}, []
        invalid_contracts = 0
        common = sim3(1.7, .21, [.3, -.2, .4])
        delta = sim3(1.2, -.17, [.2, -.1, .3])
        for case in manifest['cases']:
            identity = case['id']
            paths = {
                'native':(parent / identity / 'baseline.npz', 'pose_enc'),
                'camera':(Path(manifest['connected']) / identity / 'geometry.npz', 'camera_pose9'),
                'reciprocal':(Path(manifest['reciprocal']) / identity / 'geometry.npz', 'pose9')}
            geometry = {}
            for arm, (path, key) in paths.items():
                geometry_hashes[str(path)] = sha(path)
                with np.load(path) as data:
                    # Existing task PnP/bearing uses the FP32 camera boundary.
                    geometry[arm] = [pose_matrix(p) for p in data[key].astype(np.float32)]
            memories = {arm:EpisodicBindings() for arm in paths}
            selected = sorted([r for r in rows if r['case'] == identity],
                key=lambda r:(r['prefix'], r['query']))
            for record in selected:
                for arm, raw in record['arms'].items():
                    memory = memories[arm]
                    current = record['prefix'] - 1
                    while memory.version.observed_count <= current:
                        i = memory.version.observed_count
                        memory.append(i, geometry[arm][i])
                    version = memory.version
                    before = json.dumps(raw, sort_keys=True, allow_nan=False)
                    session = ReferenceReadoutSession(memory,
                        f'{identity}/{record["query"]}/{record["prefix"]}/{arm}')
                    initial = session.read(raw, version=version)
                    output = dict(case=identity, query=record['query'], prefix=record['prefix'],
                        arm=arm, accepted=raw['accepted'])
                    if not raw['accepted']:
                        assert initial == raw and session.goal is None
                        assert session.read(raw, version=version) == raw
                        output['rejection_preserved'] = True
                    else:
                        np.testing.assert_allclose(initial['aux_pose'], raw['aux_pose'], rtol=2e-9, atol=2e-9)
                        for field in ('terminal_yaw_right_deg', 'terminal_pitch_up_deg'):
                            np.testing.assert_allclose(initial[field], raw[field], rtol=2e-9, atol=2e-9)
                        goal = session.goal
                        h = raw['selected_anchor']
                        gauge_version = memory.publish({k:common @ t for k,t in version.nodes.items()},
                            expected_sequence=memory.version.sequence)
                        gauged = session.read(raw, version=gauge_version)
                        for field in ('aux_pose', 'terminal_yaw_right_deg', 'terminal_pitch_up_deg'):
                            np.testing.assert_allclose(gauged[field], initial[field], rtol=2e-9, atol=2e-9)
                        nodes = dict(version.nodes)
                        current_node = version.frames[current].node
                        nodes[current_node] = delta @ nodes[current_node]
                        updated_version = memory.publish(nodes, expected_sequence=memory.version.sequence)
                        updated = session.read(raw, version=updated_version)
                        # Independently apply the specified correction directly
                        # to the raw cameras, rather than calling binding.relative.
                        moved_current = delta @ geometry[arm][current]
                        moved_reference = geometry[arm][h]
                        same_node = version.frames[h].node == current_node
                        if same_node:
                            moved_reference = delta @ moved_reference
                        goal_world = moved_reference @ np.linalg.solve(geometry[arm][h], pose_matrix(raw['pnp']['pose9']))
                        expected = np.linalg.solve(moved_current, goal_world)
                        vector = expected[:3, 3][[2, 0]] * [1., -1.]
                        axis = expected[:3, 2]
                        yaw = float(np.degrees(np.arctan2(axis[0], axis[2])))
                        pitch = float(np.degrees(np.arctan2(-axis[1], np.linalg.norm(axis[[0, 2]]))))
                        np.testing.assert_allclose(updated['aux_pose'], vector, rtol=2e-9, atol=2e-9)
                        np.testing.assert_allclose(updated['terminal_yaw_right_deg'], yaw, rtol=2e-9, atol=2e-9)
                        np.testing.assert_allclose(updated['terminal_pitch_up_deg'], pitch, rtol=2e-9, atol=2e-9)
                        assert updated['pnp'] == raw['pnp'] and updated['certificate'] == raw['certificate']
                        assert session.goal is goal
                        stale_read = session.read(raw, version=version)
                        np.testing.assert_allclose(stale_read['aux_pose'], raw['aux_pose'], rtol=2e-9, atol=2e-9)
                        if invalid_contracts == 0:
                            malformed = deepcopy(raw); malformed['frame_idx'] -= 1
                            expect_value_error(lambda:session.read(malformed, version=updated_version))
                            altered = deepcopy(raw); altered['pnp']['pose9'][0] += .01
                            expect_value_error(lambda:session.read(altered, version=updated_version))
                            foreign = EpisodicBindings()
                            expect_value_error(lambda:session.read(raw, version=foreign.version))
                            switched = deepcopy(raw); switched['accepted'] = False
                            expect_value_error(lambda:session.read(switched, version=updated_version))
                            refused = ReferenceReadoutSession(memory, 'rejected-before-accepted')
                            refused.read(switched, version=updated_version)
                            expect_value_error(lambda:refused.read(raw, version=updated_version))
                            invalid_contracts = 5
                        memory.publish(dict(version.nodes), expected_sequence=memory.version.sequence)
                        output.update(unrevised_parity=True, gauge_invariant=True,
                            node_update_verified=True, original_snapshot_readable=True,
                            witness_unchanged=True, reference_and_current_share_node=same_node,
                            max_unrevised_vector_abs=float(np.max(np.abs(np.asarray(initial['aux_pose']) - raw['aux_pose']))))
                    assert json.dumps(raw, sort_keys=True, allow_nan=False) == before
                    results.append(output)
            print(json.dumps(dict(case=identity, completed_arm_queries=sum(r['case']==identity for r in results))), flush=True)
        assert len(results) == 453 and invalid_contracts == 5
        assert all(sha(p) == h for p, h in hashes.items())
        save(out / 'result.json', dict(passed=True, original_queries=151, arm_results=453,
            accepted=sum(r['accepted'] for r in results), rejected=sum(not r['accepted'] for r in results),
            invalid_contracts_rejected=invalid_contracts, rows=results, geometry_sha256=geometry_hashes,
            source_sha256=hashes, no_matching_or_truth=True,
            scope='Actual saved-witness adapter verification plus algebraic corrections; no estimated coordinate improvements or new navigation claim'))
        save(out / 'process_exit.json', dict(exit_code=0, completed_at=time.time()))
    except BaseException as error:
        save(out / 'failure.json', dict(type=type(error).__name__, error=str(error), time=time.time()))
        save(out / 'process_exit.json', dict(exit_code=1, completed_at=time.time()))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--wait-for', type=Path)
    args = parser.parse_args()
    audit(args.root.resolve(), args.out.resolve(), args.wait_for)
