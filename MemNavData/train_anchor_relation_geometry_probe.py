#!/usr/bin/env python3
"""Matched visual/XYZ input ablation. Fixed supported anchors, not navigation SR."""
from __future__ import annotations
import argparse
import copy
import csv
import json
from pathlib import Path
import random
import time

import numpy as np
from PIL import Image,ImageOps
import torch

from MemNavData.anchor_relation_decoder import AnchorRelationDecoder,position_direction_loss
from MemNavData.anchor_relation_geometry import padded_image_mask,pooled_fraction
from MemNavData.train_anchor_relation_probe import prediction_metrics,sha256,PATCH_SHA,SOURCE_TABLE_SHA


def load_dataset(input_dir:Path,geometry_dirs:list[Path]):
    manifest=json.loads((input_dir/'probe_manifest.json').read_text())
    meta=json.loads((input_dir/'history_inputs.json').read_text())
    if manifest['deduplicated_pairs']!=123 or len(manifest['pairs'])!=123:
        raise ValueError('must keep the original 123 pairs')
    lookup={}
    for root in geometry_dirs:
        completion=json.loads((root/'completion.json').read_text())
        config=json.loads((root/'configuration.json').read_text())
        if not completion['completed'] or sha256(root/'configuration.json')!=completion['configuration_sha256']:
            raise ValueError('unfinished or changed geometry extraction')
        if config['history_inputs_sha256']!=sha256(input_dir/'history_inputs.json'):
            raise ValueError('geometry input universe changed')
        for record in completion['histories']:
            if record['history'] in lookup:
                raise ValueError('duplicate geometry history')
            path=root/record['path']
            if sha256(path)!=record['sha256']:
                raise ValueError(f'geometry cache changed: {path}')
            lookup[record['history']]=path
    image_root=Path('.diagnostics/certificate_distilled_compass_20260813/certificate_images')
    cache_path=Path('.diagnostics/certificate_distilled_compass_20260813/cdec_patch_cache_fixedbatch_v2.npz')
    if sha256(cache_path)!=PATCH_SHA:
        raise ValueError('frozen DINO cache changed')
    pairs=manifest['pairs']
    with np.load(cache_path,allow_pickle=False) as cache:
        if cache['rows_csv_sha256'].item()!=SOURCE_TABLE_SHA:
            raise ValueError('DINO row universe differs')
        by_path={p:i for i,p in enumerate(cache['relative_paths'].tolist())}
        tokens=cache['tokens']
        goal=np.stack([tokens[by_path[p['query_relative_path']]] for p in pairs])
        anchor=np.stack([tokens[by_path[p['candidate_relative_path']]] for p in pairs])
    xyz=[];valid=[];goal_valid=[];normalization=[];cache_handles={}
    try:
        for pair in pairs:
            key='/'.join(Path(pair['candidate_relative_path']).parts[:2]);a=pair['candidate_frame']
            if key not in cache_handles:
                cache_handles[key]=np.load(lookup[key],allow_pickle=False)
            g=cache_handles[key]
            if a not in g['anchors'] or max(a,63)>=pair['decision_frame']:
                raise ValueError('missing or noncausal anchor')
            xyz.append(g[f'{a}/xyz_normalized'])
            valid.append(g[f'{a}/valid_mask'])
            normalization.append(float(g[f'{a}/normalization_m']))
            path=image_root/pair['query_relative_path']
            if sha256(path)!=pair['query_sha256']:
                raise ValueError('query image changed')
            with Image.open(path) as image:
                w,h=ImageOps.exif_transpose(image).size
            goal_valid.append((pooled_fraction(padded_image_mask((h,w)))>=.5).flatten().numpy())
    finally:
        for c in cache_handles.values():c.close()
    data={'goal':goal,'anchor':anchor,'xyz':np.stack(xyz),
          'anchor_valid':np.stack(valid),'goal_valid':np.stack(goal_valid),
          'normalization_m':np.asarray(normalization)[:,None],
          'target':np.asarray([p['target_xy_m'] for p in pairs])}
    if not np.isfinite(data['xyz']).all() or min(normalization)<=0:
        raise ValueError('invalid real geometry or normalization')
    return manifest,data,lookup


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input-dir',type=Path,required=True)
    p.add_argument('--geometry-dir',type=Path,action='append',required=True)
    p.add_argument('--out-dir',type=Path,required=True)
    p.add_argument('--seed',type=int,action='append')
    p.add_argument('--steps',type=int,default=1200)
    p.add_argument('--overfit-pairs',type=int,default=0)
    p.add_argument('--device',default='cuda')
    args=p.parse_args()
    if args.out_dir.exists():raise FileExistsError(args.out_dir)
    if args.steps<1 or args.overfit_pairs<0:raise ValueError('invalid training budget')
    torch.set_num_threads(4)
    manifest,raw,lookup=load_dataset(args.input_dir,args.geometry_dir)
    pairs=manifest['pairs'];device=torch.device(args.device)
    data={k:torch.as_tensor(v,device=device,dtype=torch.bool if 'valid' in k else torch.float32)
          for k,v in raw.items()}
    train=[i for i,pair in enumerate(pairs) if pair['split']=='train']
    validation=[i for i,pair in enumerate(pairs) if pair['split']=='validation']
    if args.overfit_pairs:train=train[:args.overfit_pairs]
    args.out_dir.mkdir(parents=True)
    seeds=args.seed or [11,23,37]
    run_manifest={'schema':'anchor_relation_geometry_input_ablation_v1',
        'deployment_approved':False,'navigation_SR':None,
        'source_manifest_sha256':sha256(args.input_dir/'probe_manifest.json'),
        'protocol_sha256':sha256(Path('MemNavData/ANCHOR_RELATION_GEOMETRY_PROTOCOL_20260906.md')),
        'geometry_sha256':{key:sha256(path) for key,path in lookup.items()},
        'train_indices':train,'validation_indices':validation,
        'seeds':seeds,'steps':args.steps,'pairs':pairs,
        'input_ablation':'both arms share masks and anchor depth-based output normalization; only one receives XYZ',
        'target':'same audited anchor-base [forward,left] metres',
        'model_output':'dimensionless planar offset multiplied by causal anchor normalization_m'}
    (args.out_dir/'manifest.json').write_text(json.dumps(run_manifest,indent=2)+'\n')
    predictions=[];curves=[];reports={}
    scenes=[p['scene'] for p in pairs]
    def summarize(prediction):
        return {name:prediction_metrics(prediction[idx],raw['target'][idx],
                 [scenes[i] for i in idx]) for name,idx in [('train',train),('validation',validation)]}
    def record(arm,seed,pred):
        for i,pair in enumerate(pairs):
            predictions.append({'pair_id':pair['pair_id'],'scene':pair['scene'],
                'split':pair['split'],'arm':arm,'seed':seed,
                'target_forward_m':raw['target'][i,0],'target_lateral_m':raw['target'][i,1],
                'predicted_forward_m':pred[i,0],'predicted_lateral_m':pred[i,1]})
    for seed in seeds:
        torch.manual_seed(seed)
        common=AnchorRelationDecoder().state_dict()
        for name,use_geometry in [('visual_normalized',False),('visual_xyz_normalized',True)]:
            torch.manual_seed(seed)
            model=AnchorRelationDecoder(requires_geometry=use_geometry).to(device)
            incompat=model.load_state_dict(common,strict=False)
            if incompat.unexpected_keys or any(not k.startswith('geometry_encoding.') for k in incompat.missing_keys):
                raise ValueError('shared initial weights differ')
            torch.manual_seed(seed);np.random.seed(seed);random.seed(seed)
            rng=np.random.default_rng(seed)
            optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=.01)
            started=time.monotonic()
            def forward(indices,goal_override=None,xyz_override=None):
                goal=data['goal'][indices] if goal_override is None else goal_override
                xyz=(data['xyz'][indices] if xyz_override is None else xyz_override) if use_geometry else None
                return model(goal,data['anchor'][indices],xyz,
                    goal_valid=data['goal_valid'][indices],anchor_valid=data['anchor_valid'][indices])*data['normalization_m'][indices]
            for step in range(1,args.steps+1):
                indices=torch.as_tensor(rng.choice(train,min(16,len(train)),replace=False),device=device)
                model.train();optimizer.zero_grad(set_to_none=True)
                loss=position_direction_loss(forward(indices),data['target'][indices])
                if not torch.isfinite(loss):raise RuntimeError('nonfinite training loss')
                loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
                if step==1 or step%100==0:
                    curves.append({'arm':name,'seed':seed,'step':step,'loss':float(loss)})
            model.eval()
            with torch.inference_mode():
                pred=forward(torch.arange(len(pairs),device=device)).cpu().numpy()
            result=summarize(pred)
            result['training_loop_seconds']=time.monotonic()-started
            result['parameters']=sum(p.numel() for p in model.parameters())
            reports[f'{name}/seed{seed}']=result;record(name,seed,pred)
            torch.save({'state_dict':model.cpu().state_dict(),'seed':seed,'steps':args.steps,
                        'requires_geometry':use_geometry,'deployment_approved':False,
                        'requires_masks_and_external_normalization':True},args.out_dir/f'{name}_seed{seed}.pt')
            print(json.dumps({'arm':name,'seed':seed,
                'train_mae_m':result['train']['position_error_m']['mean'],
                'validation_mae_m':result['validation']['position_error_m']['mean'],
                'validation_within_05m':result['validation']['position_within_05m']},indent=2),flush=True)
            del model,optimizer
    baselines={'anchor_copy':np.zeros_like(raw['target']),
        'training_mean_position':np.broadcast_to(raw['target'][train].mean(0),raw['target'].shape),
        'existing_frozen_gct_query':np.asarray([p['gct_xy_m'] for p in pairs]),
        'existing_finite_pnp':np.asarray([p['pnp_xy_m'] if p['pnp_xy_m'] is not None else [np.nan,np.nan] for p in pairs])}
    for name,pred in baselines.items():reports[name]=summarize(pred);record(name,'reference',pred)
    for filename,rows in [('predictions.csv',predictions),('training_curve.csv',curves)]:
        with (args.out_dir/filename).open('w',newline='') as handle:
            writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    (args.out_dir/'report.json').write_text(json.dumps({'results':reports,'navigation_SR':None,
        'scope':'one train40 internal scene split, all fixed seeds; no proof or controller',
        'predictions_sha256':sha256(args.out_dir/'predictions.csv'),
        'manifest_sha256':sha256(args.out_dir/'manifest.json')},indent=2,allow_nan=False)+'\n')


if __name__=='__main__':main()
