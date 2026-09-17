"""Task-specific historical storage versus dense W64 and paper GEM, 70 histories.

All three configurations run sequentially in one Slurm array element. The
existing runner and independent per-arm auditor retain their original frozen
population plan; this outer plan specifies the actual configuration comparison.
Neither the original population file nor the production method is rewritten.
"""
import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'MemNavData')]
ARMS = (
    dict(name='legacy', mode='legacy', dense_window=32, geometry_storage=None),
    dict(name='native64', mode='native_interval7', dense_window=64, geometry_storage='dense'),
    dict(name='support64', mode='native_interval7', dense_window=64, geometry_storage='detector_support'))


def load(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def check_plan(path):
    plan = load(path)
    assert plan['schema'] == 'gem_support_navigation_paired_v1'
    assert plan['arms'] == list(ARMS)
    assert sha(plan['population_plan']) == plan['population_plan_sha256']
    population = load(plan['population_plan'])
    assert plan['cells'] == population['cells'] and len(plan['cells']) == 70
    assert plan['total_rollouts'] == 420
    for name, digest in plan['source_sha256'].items():
        assert sha(ROOT / name) == digest, name
    return plan


def freeze(args):
    original = load(args.source)
    assert len(original['cells']) == 70 and original['total_rollouts'] == 420
    assert original['max_steps'] == 600 and original['success_radius_m'] == 1.
    source_files = [Path(__file__), ROOT / 'MemNavData/run_gem_memory_navigation.py',
        ROOT / 'MemNavData/verify_gem_memory_navigation.py',
        ROOT / 'MemNavData/reduce_gem_support_navigation.py',
        ROOT / 'MemNavData/run_gem_support_navigation_hpc.sh',
        ROOT / 'MemNavData/slurm_gem_support_navigation.sbatch',
        ROOT / 'NavDP/baselines/memnav/policy_agent.py',
        ROOT / 'NavDP/baselines/memnav/memnav_server.py',
        ROOT / 'MemNavData/lingbot_pnp_localization.py',
        ROOT / 'MemNavData/certified_relocalization_runtime.py',
        ROOT / 'MemNavData/certified_relocalization_contract.py',
        *sorted((ROOT / 'NavDP/baselines/memnav/gem').glob('*.py'))]
    plan = dict(schema='gem_support_navigation_paired_v1',
        population_plan=str(args.population_plan), population_plan_sha256=sha(args.source),
        cells=original['cells'], arms=list(ARMS), total_rollouts=420,
        source_sha256={str(p.relative_to(ROOT)):sha(p) for p in source_files},
        gate_index=14, gate_selection='Longest original history,564frames; no outcome-based selection',
        max_steps=600, success_radius_m=1., exec_horizon=8,
        ordering='Cyclic arm order by history index modulo3; each arm starts fresh services',
        pairing='All arms of each history run in one array element on the same actual GPU UUID',
        scope='Complete previously used 70-history population; same-GPU paired comparison of paper GEM, native W64/full history maps, native W64/detector-support storage. Not new unseen scenes or thousand-frame closed-loop evidence.',
        production='Online support storage verified on234actual frames and24real image pairs; historical storage is the sole difference between native64/support64. The original SP/LG/PnP/certificate/controller is retained.',
        legacy_population_mode_catalog='Original plan includes connected backend for compatibility with the original runner/auditor; actual arms and reduction are exclusively specified here',
        created_at=time.time())
    save(args.out, plan)
    print(json.dumps(dict(histories=70, paired_tasks=70, rollouts=420, gate_index=14,
        plan_sha256=sha(args.out))), flush=True)


def gpu_binding(out):
    owned = load(out / 'owned_processes.json')
    pids = {int(p['pid']) for p in owned}
    output = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,gpu_uuid',
        '--format=csv,noheader,nounits'], text=True)
    bindings = {}
    for line in output.splitlines():
        pid, uuid = [part.strip() for part in line.split(',')]
        if int(pid) in pids:
            bindings.setdefault(pid, set()).add(uuid)
    assert set(map(int, bindings)) == pids, (pids, bindings)
    uuids = set().union(*bindings.values())
    assert len(uuids) == 1, bindings
    return dict(gpu_uuid=next(iter(uuids)), hostname=socket.gethostname(),
        slurm_job_id=os.environ.get('SLURM_JOB_ID'),
        slurm_array_task_id=os.environ.get('SLURM_ARRAY_TASK_ID'),
        cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
        owned_pid_gpu={p:sorted(v) for p,v in bindings.items()},
        owned_processes_sha256=sha(out / 'owned_processes.json'))


def run_arm(args):
    from MemNavData import run_gem_memory_navigation as base
    plan = check_plan(args.plan)
    arm = next(a for a in plan['arms'] if a['name'] == args.arm)
    original_servers = base.servers

    @contextmanager
    def configured_servers(out, mode, mem_port, nav_port):
        assert mode == arm['mode']
        with original_servers(out, mode, mem_port, nav_port, dense_window=arm['dense_window'],
                geometry_storage=arm['geometry_storage']):
            yield
            # NavDP loads its CUDA model on the evaluator's first reset, after
            # its HTTP port becomes ready. Inspect both owned processes after
            # the normal rollouts, while the service context still keeps them
            # alive; do not add a reset or relax the same-GPU assertion.
            save(out / 'paired_gpu_binding.json', dict(gpu_binding(out),
                capture_phase='after_rollouts_before_service_shutdown'))

    base.servers = configured_servers
    call = SimpleNamespace(plan=Path(plan['population_plan']), index=args.index,
        mode=arm['mode'], out=args.out, mem_port=args.mem_port, nav_port=args.nav_port)
    base.run(call)
    check_plan(args.plan)
    storage_sha = None
    if arm['geometry_storage'] is not None:
        storage_sha = sha(args.out / 'archive_storage_receipt.json')
        actual = load(args.out / 'archive_storage_receipt.json')
        assert actual['geometry_storage'] == arm['geometry_storage']
        for role in ('novel','revisit'):
            status = load(args.out / 'evaluation' / role / 'memory_resources.json')['memory']
            assert status.get('geometry_storage','dense') == arm['geometry_storage']
            assert status['frames'] == status['online_depth_frames']
    save(args.out / 'memory_arm_receipt.json', dict(completed=True, arm=arm,
        index=args.index, plan=str(args.plan.resolve()), plan_sha256=sha(args.plan),
        fixed_window_sha256=sha(args.out / 'fixed_window_receipt.json'),
        gpu_binding_sha256=sha(args.out / 'paired_gpu_binding.json'),
        geometry_storage_sha256=storage_sha,
        summary_sha256=sha(args.out / 'summary.json')))


def run_pair(args):
    plan = check_plan(args.plan)
    cell = plan['cells'][args.index]
    assert cell['index'] == args.index
    args.out.mkdir(parents=True, exist_ok=False)
    shift = args.index % len(ARMS)
    order = list(ARMS[shift:] + ARMS[:shift])
    save(args.out / 'pair_manifest.json', dict(index=args.index, cell=cell,
        plan=str(args.plan.resolve()), plan_sha256=sha(args.plan),
        order=[a['name'] for a in order], started_at=time.time(), pid=os.getpid()))
    began = time.monotonic()
    gpu = None
    rows = []
    try:
        for arm in order:
            target = args.out / arm['name']
            command = [sys.executable, '-u', str(Path(__file__).resolve()), 'arm',
                '--plan', str(args.plan.resolve()), '--index', str(args.index),
                '--arm', arm['name'], '--out', str(target.resolve()),
                '--mem-port', str(args.mem_port), '--nav-port', str(args.nav_port)]
            with (args.out / (arm['name'] + '.log')).open('x') as stream:
                code = subprocess.call(command, stdout=stream, stderr=subprocess.STDOUT)
            save(args.out / (arm['name'] + '_exit.json'), dict(exit_code=code, command=command))
            if code:
                raise RuntimeError(f'{arm["name"]} navigation process exited {code}')
            binding = load(target / 'paired_gpu_binding.json')
            if gpu is None:
                gpu = binding
            assert all(binding[k] == gpu[k] for k in (
                'gpu_uuid', 'hostname', 'slurm_job_id', 'slurm_array_task_id'))
            audit = [os.environ['REPAIRED_HAB_PY'],
                str(ROOT / 'MemNavData/verify_gem_memory_navigation.py'), 'task', str(target.resolve())]
            with (args.out / (arm['name'] + '_audit.log')).open('x') as stream:
                code = subprocess.call(audit, stdout=stream, stderr=subprocess.STDOUT)
            save(args.out / (arm['name'] + '_audit_exit.json'), dict(exit_code=code, command=audit))
            if code:
                raise RuntimeError(f'{arm["name"]} independent execution audit exited {code}')
            rows.append(dict(arm=arm, receipt_sha256=sha(target / 'memory_arm_receipt.json'),
                independent_verification_sha256=sha(target / 'independent_verification.json')))
            print(json.dumps(dict(index=args.index, arm=arm['name'], verified=True)), flush=True)
        check_plan(args.plan)
        save(args.out / 'completion.json', dict(completed=True, index=args.index,
            plan_sha256=sha(args.plan), gpu=gpu, arms=rows,
            seconds=time.monotonic()-began, completed_at=time.time()))
    except BaseException as error:
        save(args.out / 'failure.json', dict(type=type(error).__name__, error=str(error),
            elapsed_seconds=time.monotonic()-began, time=time.time()))
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['freeze', 'pair', 'arm'])
    parser.add_argument('--source', type=Path)
    parser.add_argument('--population-plan', type=Path)
    parser.add_argument('--plan', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--index', type=int)
    parser.add_argument('--arm', choices=[a['name'] for a in ARMS])
    parser.add_argument('--mem-port', type=int, default=19217)
    parser.add_argument('--nav-port', type=int, default=19218)
    args = parser.parse_args()
    if args.action == 'freeze':
        freeze(args)
    elif args.action == 'pair':
        run_pair(args)
    else:
        run_arm(args)


if __name__ == '__main__':
    main()
