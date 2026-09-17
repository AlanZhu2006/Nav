#!/usr/bin/env python3
"""Continue the verified local geometry smoke through the fixed training ablation."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--resume-after-overfit',action='store_true',
                   help='reuse completed geometry and overfit outputs; verify before the fixed full training')
    args=p.parse_args();root=args.root.resolve()
    inputs=root/'inputs/unpacked';smoke=root/'geometry_smoke'
    audit=json.loads((root/'input_verification.json').read_text())
    done=json.loads((smoke/'completion.json').read_text())
    if not audit['verified'] or audit['pairs']!=123 or not done['completed']:
        raise ValueError('complete input audit and genuine geometry smoke required')
    histories=json.loads((inputs/'history_inputs.json').read_text())['histories']
    remaining=sorted(set(histories)-{r['history'] for r in done['histories']})
    work=root/('workflow_resume_'+time.strftime('%Y%m%dT%H%M%SZ',time.gmtime())
               if args.resume_after_overfit else 'workflow_v1')
    work.mkdir(exist_ok=False)
    source_names=['anchor_relation_decoder.py','anchor_relation_geometry.py',
        'extract_anchor_relation_geometry.py','train_anchor_relation_geometry_probe.py',
        'verify_anchor_relation_geometry_probe.py','run_anchor_relation_geometry_local.py',
        'ANCHOR_RELATION_GEOMETRY_PROTOCOL_20260906.md','diag_m2p_s1_gct_query.py',
        'train_anchor_relation_probe.py']
    sources={str(Path('MemNavData')/name):hashlib.sha256((Path('MemNavData')/name).read_bytes()).hexdigest()
             for name in source_names}
    receipt={'pid':os.getpid(),'python':sys.executable,'cwd':str(Path.cwd()),
             'started_unix':time.time(),'remaining_histories':remaining,'sources':sources,
             'resume_after_overfit':args.resume_after_overfit,
             'reused_workflow':str(root/'workflow_v1') if args.resume_after_overfit else None,
             'navigation_SR':None,'hpc_gpu_job_submitted':False,
             'production_CEC_changed':False}
    (work/'launch_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    def run(stage,command):
        for path,sha in sources.items():
            if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=sha:
                raise RuntimeError(f'workflow source changed before {stage}: {path}')
        state={'stage':stage,'status':'running','command':command,'time_unix':time.time()}
        log_path=work/f'{stage}.log'
        with log_path.open('x') as log:
            proc=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                                  text=True,bufsize=1)
            state['child_pid']=proc.pid
            (work/'status.json').write_text(json.dumps(state,indent=2)+'\n')
            for line in proc.stdout:
                log.write(line);log.flush();print(line,end='',flush=True)
            code=proc.wait()
        state['exit_code']=code;state['status']='completed' if code==0 else 'failed'
        (work/'status.json').write_text(json.dumps(state,indent=2)+'\n')
        if code:raise RuntimeError(f'{stage} failed; inspect {log_path}')
    rest=root/'geometry_remainder'
    command=[sys.executable,'-u','-m','MemNavData.extract_anchor_relation_geometry',
             '--input-dir',str(inputs),'--out-dir',str(rest)]
    for key in remaining:command+=['--history',key]
    if not args.resume_after_overfit:
        run('extract_remaining',command)
    training=[sys.executable,'-u','-m','MemNavData.train_anchor_relation_geometry_probe',
              '--input-dir',str(inputs),'--geometry-dir',str(smoke),'--geometry-dir',str(rest)]
    if not args.resume_after_overfit:
        run('overfit8',training+['--out-dir',str(root/'overfit8_geometry_v1'),
                                '--overfit-pairs','8','--steps','600','--seed','11'])
    run('verify_overfit8',[sys.executable,'-m','MemNavData.verify_anchor_relation_geometry_probe',
                          str(root/'overfit8_geometry_v1')])
    run('scene32_8',training+['--out-dir',str(root/'scene32_8_geometry_v1')])
    run('verify_scene32_8',[sys.executable,'-m','MemNavData.verify_anchor_relation_geometry_probe',
                          str(root/'scene32_8_geometry_v1')])
    (work/'completion.json').write_text(json.dumps({'completed':True,
        'elapsed_seconds':time.time()-receipt['started_unix'],'navigation_SR':None,
        'report':str(root/'scene32_8_geometry_v1/report.json')},indent=2)+'\n')
    print('[workflow complete] All fixed seeds and independent recount completed.',flush=True)


if __name__=='__main__':main()
