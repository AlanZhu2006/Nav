"""One GPU correctness check for new fused decoding; no navigation/cost claim."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
from NavDP.baselines.memnav.gem import int8_state


def reference(state,current,commit):
    """Independent chronological gather using PyTorch, used only for testing."""
    previous=list(range(8,state.count))[-(64-int(commit)):]
    evicted=list(range(8,state.count-len(previous)))
    parts=[]
    if evicted:
        parts.append(state.specials[:,:,evicted].flatten(2,3))
    for frame in list(range(8))+previous:
        if frame<8:
            patch=state.initial[:,:,frame]
        else:
            slot=(frame-8)%64
            key=(state.codes[0,slot].float()*state.key_scales[slot,:,None,:].float()).to(torch.bfloat16)
            value=(state.codes[1,slot].float()*state.value_scales[slot].float()).to(torch.bfloat16)
            patch=torch.stack((key,value))
        parts.append(torch.cat((state.specials[:,:,frame],patch),dim=2))
    parts.append(current)
    return torch.cat(parts,dim=2)


def main(out):
    out.mkdir(exist_ok=False,parents=True)
    expected_gpu='GPU-87b92058-ab9f-9572-7cb4-f2a7536f0946'
    assert os.environ['CUDA_VISIBLE_DEVICES']==expected_gpu
    torch.set_num_threads(2)
    rows=[]
    started=time.time()
    for heads,dim in [(2,8),(16,64)]:
        tokens=1375
        template=(torch.arange(2*heads*tokens*dim,device='cuda',dtype=torch.float32)
            .reshape(2,heads,tokens,dim)%37-18)/16

        def frame(index):
            # Nonconstant head/channel/token values and chronological identity.
            return (template+index/32).to(torch.bfloat16).contiguous()

        initial=torch.stack([frame(i) for i in range(8)],dim=2)
        state=int8_state.Int8LayerState(initial)
        workspace=int8_state.Int8DecodeWorkspace()
        selected={8,9,71,72,73,228,229,286,319}
        for count in range(8,320):
            current=frame(count)
            if count in selected:
                for commit in [False,True]:
                    expected=reference(state,current,commit)
                    before=state.statistics()
                    qk,qv=workspace.decode(state,current,commit=commit)
                    torch.cuda.synchronize()
                    assert qk.is_contiguous() and qv.is_contiguous()
                    key_exact=torch.equal(qk[0],expected[0])
                    value_exact=torch.equal(qv[0],expected[1])
                    row=dict(heads=heads,dim=dim,count=count,commit=commit,
                        tokens=qk.shape[2],key_exact=key_exact,value_exact=value_exact,
                        key_max_abs=float((qk[0].float()-expected[0].float()).abs().max()),
                        value_max_abs=float((qv[0].float()-expected[1].float()).abs().max()),
                        state_unchanged=state.statistics()==before)
                    rows.append(row)
                    if not key_exact or not value_exact:
                        torch.save(dict(expected=expected.cpu(),actual_k=qk.cpu(),actual_v=qv.cpu()),out/'failed_tensors.pt')
                        raise RuntimeError(str(row))
                    assert row['state_unchanged']
                    del expected,qk,qv
            state.commit(current)
        assert state.count==320
        print(json.dumps(dict(heads=heads,dim=dim,checks=18,exact=True)),flush=True)
        del state,workspace,initial,template,current
    result=dict(complete=True,checks=len(rows),rows=rows,gpu=torch.cuda.get_device_name(),
        gpu_uuid=expected_gpu,torch_version=torch.__version__,seconds=time.time()-started,
        sources={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in
                 [Path(__file__).resolve(),Path(int8_state.__file__).resolve()]},
        scope=__doc__)
    with (out/'result.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True)
    main(p.parse_args().out.resolve())
