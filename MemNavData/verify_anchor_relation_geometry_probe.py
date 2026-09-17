#!/usr/bin/env python3
"""Recount the geometry ablation from CSV using independent 2-D math."""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import struct


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prediction_value(text, *, legacy_float32):
    value = float(text)
    if legacy_float32:
        # V1 wrote NumPy float32 scalars using their shortest float32-roundtrip
        # decimal. Restore that original dtype before independent float64 math;
        # reading the decimal directly as float64 changes the saved prediction.
        value = struct.unpack('f', struct.pack('f', value))[0]
    return value


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('root',type=Path)
    args=p.parse_args();root=args.root
    out=root/'independent_verification.json'
    if out.exists():raise FileExistsError(out)
    manifest=json.loads((root/'manifest.json').read_text())
    report=json.loads((root/'report.json').read_text())
    encoding=manifest.get('prediction_csv_encoding','numpy_scalar_shortest')
    assert encoding in {'numpy_scalar_shortest','python_float_roundtrip'}
    assert digest(root/'manifest.json')==report['manifest_sha256']
    assert digest(root/'predictions.csv')==report['predictions_sha256']
    assert manifest['deployment_approved'] is False and report['navigation_SR'] is None
    pairs={p['pair_id']:p for p in manifest['pairs']}
    assert len(pairs)==123
    populations={split:{manifest['pairs'][i]['pair_id'] for i in manifest[f'{split}_indices']}
                 for split in ['train','validation']}
    scenes={split:{pairs[i]['scene'] for i in ids} for split,ids in populations.items()}
    assert scenes['train'].isdisjoint(scenes['validation'])
    grouped=defaultdict(list);identities=set()
    with (root/'predictions.csv').open(newline='') as f:
        for row in csv.DictReader(f):
            identity=(row['arm'],row['seed'],row['pair_id'])
            assert identity not in identities;identities.add(identity)
            pair=pairs[row['pair_id']]
            assert pair['split']==row['split'] and pair['scene']==row['scene']
            assert math.isclose(float(row['target_forward_m']),pair['target_xy_m'][0],abs_tol=1e-8)
            assert math.isclose(float(row['target_lateral_m']),pair['target_xy_m'][1],abs_tol=1e-8)
            if row['pair_id'] in populations[row['split']]:
                key=row['arm'] if row['seed']=='reference' else f"{row['arm']}/seed{row['seed']}"
                grouped[(key,row['split'])].append(row)
    checks={}
    for (key,split),rows in grouped.items():
        expected=report['results'][key][split]
        assert len(rows)==len(populations[split])==expected['pairs']
        errors=[];angles=[];eligible=0;scene_errors=defaultdict(list)
        for row in rows:
            tx,ty=float(row['target_forward_m']),float(row['target_lateral_m'])
            eligible+=math.hypot(tx,ty)>=.25
            legacy_float32=encoding=='numpy_scalar_shortest' and row['seed']!='reference'
            px,py=(prediction_value(row[column],legacy_float32=legacy_float32)
                   for column in ('predicted_forward_m','predicted_lateral_m'))
            if not math.isfinite(px) or not math.isfinite(py):continue
            error=math.hypot(px-tx,py-ty);errors.append(error);scene_errors[row['scene']].append(error)
            if math.hypot(tx,ty)>=.25 and math.hypot(px,py)>1e-6:
                delta=math.atan2(py,px)-math.atan2(ty,tx)
                angles.append(abs(math.degrees(math.atan2(math.sin(delta),math.cos(delta)))))
        assert len(errors)==expected['valid_predictions']
        assert eligible==expected['anchor_local_direction_eligible']
        assert len(angles)==expected['anchor_local_direction_valid']
        assert sum(x<=.5 for x in errors)==expected['position_within_05m']
        assert sum(x<=30 for x in angles)==expected['anchor_local_direction_within_30deg']
        assert sum(x>90 for x in angles)==expected['anchor_local_direction_over_90deg']
        for observed,reference in [(statistics.mean(errors),expected['position_error_m']['mean']),
                (statistics.median(errors),expected['position_error_m']['median']),
                (statistics.mean(statistics.mean(x) for x in scene_errors.values()),
                 expected['scene_macro_position_error_m'])]:
            assert math.isclose(observed,reference,abs_tol=1e-8)
        checks[f'{key}/{split}']={'pairs':len(rows),'valid':len(errors),
                                  'mean_position_error_m':statistics.mean(errors)}
    assert len(checks)==2*(2*len(manifest['seeds'])+4)
    result={'verified':True,'groups':checks,'navigation_SR':None,
            'prediction_csv_encoding':encoding,
            'legacy_float32_roundtrip_restored':encoding=='numpy_scalar_shortest',
            'report_sha256':digest(root/'report.json'),
            'scope':'independent saved prediction recount, not independent retraining'}
    out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'verified':True,'groups':len(checks),'output':str(out)}))


if __name__=='__main__':main()
