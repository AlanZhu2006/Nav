"""Conservative current-view covisibility upper bounds without rendering.

Ignoring occlusion can only increase visibility. Each PNG depth is treated as
an interval, including saturation at 65535, rather than as exact metric depth.
The stride and intrinsics match the original main-query annotation convention.
All poses/depths are offline evaluation truth, never navigation inputs.
"""
import argparse
import base64
import hashlib
import io
import math
from pathlib import Path

import numpy as np
from PIL import Image

from MemNavData.gem_bearing_attribution import dump, load, sha


def rotation(yaw):
    c,s=math.cos(yaw),math.sin(yaw)
    return np.array([[c,0,s],[0,1,0],[-s,0,c]])


def frustum_upper_bound(depth_u16, goal_position, goal_yaw, current_position, current_yaw,
                        *, stride=6, fx=355.81464, fy=351.687):
    depth=np.asarray(depth_u16)
    assert depth.shape == (270,480) and depth.min() >= 0 and depth.max() <= 65535
    height,width=depth.shape
    v,u=np.meshgrid(np.arange(0,height,stride),np.arange(0,width,stride),indexing="ij")
    encoded=depth[v,u].astype(float).reshape(-1)
    # Encoding is floor(clip(depth*10000,0,65535)); epsilon also covers float conversion.
    lower=encoded/10000-1e-6
    upper=(encoded+1)/10000+1e-6
    upper[encoded==65535]=np.inf
    definite=(lower>.15)&(upper<10)
    possible=(upper>.15)&(lower<10)
    lo=np.maximum(lower,.15)
    hi=np.minimum(upper,10.)
    rays=np.column_stack(((u.ravel()-240.)/fx,-(v.ravel()-135.)/fy,-np.ones(len(encoded))))
    relative=rotation(current_yaw).T@rotation(goal_yaw)
    a=rays@relative.T
    b=rotation(current_yaw).T@(np.asarray(goal_position)-np.asarray(current_position))
    # OpenGL optical camera: x right, y up, -z forward. Plane >=0 relaxes the strict edges.
    planes=np.array([[0,0,-1],[fx,0,-240.],[-fx,0,-(width-1-240.)],
                     [0,-fy,-135.],[0,fy,-(height-1-135.)]])
    constants=np.array([-.05,0.,0.,0.,0.])
    feasible=possible.copy()
    for plane,constant in zip(planes,constants):
        alpha=a@plane
        beta=float(b@plane)+constant+1e-7
        positive=alpha>1e-12
        negative=alpha< -1e-12
        flat=~(positive|negative)
        lo[positive]=np.maximum(lo[positive],-beta/alpha[positive])
        hi[negative]=np.minimum(hi[negative],-beta/alpha[negative])
        feasible&=~(flat & (beta<0))
    feasible &= lo<=hi
    # Unknown validity of saturated/out-of-range points: retain potentially
    # visible ones and omit the rest to maximize the visibility fraction.
    optional_visible=int(np.count_nonzero(feasible & ~definite))
    mandatory=int(np.count_nonzero(definite))
    visible=int(np.count_nonzero(feasible & definite))+optional_visible
    denominator=mandatory+optional_visible
    bound=visible/denominator if denominator else 0.
    return dict(current_covisibility_upper_bound=bound, mandatory_valid_goal_points=mandatory,
                optional_visible_points=optional_visible, possible_visible_points=visible,
                saturated_sample_points=int(np.count_nonzero(encoded==65535)),
                interpretation="Upper bound under the original annotation camera convention; occlusion omitted, quantization/saturation relaxed")


def audit(history_path, depth_path, output):
    histories=load(history_path)["rows"]
    depths={(r["dataset"],r["scene"],r["episode"]):r for r in load(depth_path)["rows"]}
    rows=[]
    for h in histories:
        c=h["cell"]
        d=depths[(c["dataset"],c["scene"],c["episode"])]
        q=next(q for q in h["queries"] if q["analysis_role"]=="revisit")
        data=base64.b64decode(d["png_base64"])
        assert hashlib.sha256(data).hexdigest()==q["goal_depth_sha256"]==d["goal_depth_sha256"]
        image=np.asarray(Image.open(io.BytesIO(data)))
        camera_offset=np.array([0.,d["camera_height_m"],0.])
        bound=frustum_upper_bound(image,np.asarray(q["floor_position"])+camera_offset,q["yaw_rad"],
                                  np.asarray(h["end_position"])+camera_offset,h["end_yaw"])
        curve=np.asarray(q["covis_curve"])
        fifo=h["decision_steps"][-7:]
        recent=float(curve[fifo].max())
        all_support=float(curve[8:].max())
        rows.append(dict(dataset=c["dataset"],scene=c["scene"],episode=c["episode"],
                         goal_depth_sha256=q["goal_depth_sha256"],**bound,
                         historical_support=all_support,prior_fifo_support=recent,
                         old_only_historical_support=all_support>=.5 and recent<.1,
                         strict_old_memory_subset=all_support>=.5 and recent<.1 and bound["current_covisibility_upper_bound"]<.1))
    result=dict(verified=True,rows=rows,queries=len(rows),strict_old_memory_queries=sum(r["strict_old_memory_subset"] for r in rows),
                history_export_sha256=sha(history_path),depth_export_sha256=sha(depth_path),source_sha256=sha(__file__),
                original_metric=dict(width=480,height=270,fx=355.81464,fy=351.687,cx=240,cy=135,stride=6,depth_min=.15,depth_max=10.),
                new_render_or_model_calls=0)
    dump(output,result)
    print({k:v for k,v in result.items() if k!="rows"},flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--histories",type=Path,required=True)
    parser.add_argument("--depths",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()
    audit(args.histories,args.depths,args.out)
