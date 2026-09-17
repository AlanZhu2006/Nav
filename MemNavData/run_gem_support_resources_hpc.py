"""Run the unchanged 12-trial resource protocol in one Slurm GPU allocation.

The supervisor uses CPU only. It records actual worker GPU UUIDs and device
telemetry, supplementing the original per-process resource sampler.
"""
import argparse
import csv
import io
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from MemNavData.reduce_gem_native_budget_resources import load, sha


def save(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def query(arguments):
    result = subprocess.run(['nvidia-smi', *arguments], check=True,
        capture_output=True, text=True, timeout=15)
    return result.stdout


def main(args):
    if not os.environ.get('SLURM_JOB_ID') or 'login' in socket.gethostname():
        raise RuntimeError('This complete GPU experiment requires a Slurm compute allocation')
    plan = load(args.plan)
    assert len(plan['tasks']) == 12
    assert all(sha(p) == h for p, h in plan['source_sha256'].items())
    assert plan['execution_environment']['allocation'] == 'single_slurm_gpu_all_twelve_trials'
    out = args.plan.parent
    command = [sys.executable, '-u', str(ROOT/'MemNavData/benchmark_gem_support_resources.py'),
        'run', '--plan', str(args.plan.resolve())]
    save(out/'slurm_allocation.json', dict(hostname=socket.gethostname(),
        slurm_job_id=os.environ['SLURM_JOB_ID'], cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
        plan_sha256=sha(args.plan), supervisor_sha256=sha(__file__), command=command,
        gpu_inventory=query(['--query-gpu=uuid,name,memory.total', '--format=csv,noheader,nounits'])))
    gpu_samples = []
    with (out/'allocation_telemetry.jsonl').open('x', buffering=1) as telemetry, (out/'driver.log').open('x') as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        save(out/'driver_process.json', dict(pid=process.pid, command=command, plan_sha256=sha(args.plan)))
        try:
            while process.poll() is None:
                try:
                    processes = query(['--query-compute-apps=pid,gpu_uuid,used_gpu_memory', '--format=csv,noheader,nounits'])
                    devices = query(['--query-gpu=uuid,utilization.gpu,clocks.sm,power.draw,memory.used',
                        '--format=csv,noheader,nounits'])
                    row = dict(time=time.time(), processes_csv=processes, devices_csv=devices)
                    gpu_samples.append(row)
                except (subprocess.TimeoutExpired, subprocess.CalledProcessError) as error:
                    # A missed observation does not terminate or restart the
                    # experiment. Re-poll this same live process and record
                    # the gap; final provenance checks remain mandatory.
                    row = dict(time=time.time(), telemetry_error=str(error))
                telemetry.write(json.dumps(row)+'\n')
                time.sleep(1.)
            code = process.wait()
        except BaseException:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try: process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
            raise
    save(out/'driver_exit.json', dict(exit_code=code, plan_sha256=sha(args.plan)))
    if code != 0:
        raise RuntimeError('Full resource driver or its independent reduction failed; retain all trials')
    reduction = load(out/'independent_reduction.json')
    assert reduction['complete'] and reduction['verified_tasks'] == 12
    assert reduction['all_first_readouts_exact']
    workers = {int(load(out/'tasks'/str(t['index'])/'process.json')['pid']):t['index'] for t in plan['tasks']}
    observed = {p:set() for p in workers}
    parsed = []
    for sample in gpu_samples:
        values = [(int(p.strip()), u.strip()) for p, u, _ in csv.reader(io.StringIO(sample['processes_csv']))]
        parsed.append((sample['time'], values))
        for pid, uuid in values:
            if pid in observed: observed[pid].add(uuid)
    assert all(len(v) == 1 for v in observed.values()), observed
    assigned = set().union(*observed.values())
    assert len(assigned) == 1, assigned
    foreign = sorted({pid for _, values in parsed for pid, uuid in values if uuid in assigned and pid not in workers})
    proof = dict(complete=True, verified_tasks=12, plan_sha256=sha(args.plan),
        reduction_sha256=sha(out/'independent_reduction.json'),
        telemetry_sha256=sha(out/'allocation_telemetry.jsonl'),
        allocation_sha256=sha(out/'slurm_allocation.json'),
        all_first_readouts_exact=True, actual_gpu_uuid=next(iter(assigned)),
        worker_gpu_uuid={p:sorted(v) for p,v in observed.items()}, foreign_gpu_pids=foreign,
        no_observed_foreign_gpu_process=not foreign,
        scope='All12fresh trials on one actual Slurm GPU. Separate complete hardware-specific study; do not pool with paused local7 trials. Telemetry is sampled and does not exclude unobserved transient interference.')
    save(out/'allocation_proof.json', proof)
    if foreign:
        raise RuntimeError('Other GPU contexts observed on the assigned device; cost isolation is not established')
    print(json.dumps(proof), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', type=Path, required=True)
    main(parser.parse_args())
