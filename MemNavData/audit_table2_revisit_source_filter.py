"""Check whether an early source-distance filter hides valid Revisit goals.

Construction-only: no model scores, certificate decisions or query rollouts.
Keep the goal-distance/support/perturbation requirements unchanged.
"""
import argparse
import math
from pathlib import Path

import numpy as np

from MemNavData.table2_mixed_local import load, dump, history_at, sha
from MemNavData.generate_twoleg import make_sim, render, covis_curve
from MemNavData.build_shared_online_double_revisit import goal_world_points, pixel_mae, jpeg_bytes
from MemNavData.build_shared_online_role_pairs import query_geometry
from MemNavData.build_final14_role_pair_scene import deterministic_pose_grid
from MemNavData.final14_role_pair_contract import DEPTH_TOLERANCE_M


def audit(prefix, source_file, out):
    source=load(source_file)
    history=history_at(prefix)
    endpoint=np.asarray(history["trace"]["end_position"])
    height=source["camera_height_m"]
    key=f"{source['scene']}/{source['episode']}/B/{prefix.name}"
    sim=make_sim(source["asset"],"",agent_radius=.30)
    checked=[]
    accepted=None
    out.mkdir(parents=True,exist_ok=False)
    try:
        for frame in list(range(39,len(history["poses"])-16,8))[:6]:
            origin=history["floor_positions"][frame]
            geo=query_geometry(sim.pathfinder,endpoint,origin)
            if geo is None:
                continue
            # Only examine sources discarded by the premature 2--9 m check.
            if 2. <= geo[0] <= 9.:
                continue
            for attempt,(radius,angle,dyaw) in enumerate(deterministic_pose_grid(key+f"/{frame}")):
                if radius>.8 or not 12.<=abs(dyaw)<=45.:
                    continue
                raw=origin+radius*np.array([math.cos(angle),0.,math.sin(angle)])
                pos=np.asarray(sim.pathfinder.snap_point(raw))
                if (not sim.pathfinder.is_navigable(pos) or abs(pos[1]-origin[1])>.20
                        or np.linalg.norm((pos-raw)[[0,2]])>.20
                        or not .20<=np.linalg.norm((pos-origin)[[0,2]])<=.80):
                    continue
                candidate_geo=query_geometry(sim.pathfinder,endpoint,pos)
                if candidate_geo is None or not 2.<=candidate_geo[0]<=9.:
                    continue
                yaw=(history["poses"][frame]["yaw"]+math.radians(dyaw)+math.pi)%(2*math.pi)-math.pi
                camera=pos+[0.,height,0.]
                rgb,depth=render(sim,camera,yaw)
                curve=covis_curve(goal_world_points(depth,camera,yaw),history["transforms"],history["depths"],tol=DEPTH_TOLERANCE_M)
                maximum=float(curve[8:].max())
                mae=pixel_mae(rgb,history["rgbs"][frame])
                row=dict(source_frame=frame,source_geodesic_m=geo[0],target_geodesic_m=candidate_geo[0],
                         runtime_max_covis=maximum,pixel_mae=mae,attempt=attempt,
                         translation_m=float(np.linalg.norm((pos-origin)[[0,2]])),yaw_delta_deg=abs(dyaw))
                checked.append(row)
                if mae>=5. and .55<=maximum<=.90:
                    accepted=dict(row,floor_position=pos.tolist(),yaw_rad=yaw,covis_curve=curve.tolist())
                    (out/"candidate.jpg").write_bytes(jpeg_bytes(rgb))
                    break
            if accepted:
                break
    finally:
        sim.close()
    result=dict(prefix_trace_sha256=sha(prefix/"online_a_trace.json"),source_sha256=sha(source_file),
                goal_constraints_changed=False,policy_outcomes_read=False,
                checked=checked,valid_goal_rejected_by_source_filter=accepted)
    dump(out/"audit.json",result)
    print({k:v for k,v in (accepted or {}).items() if k!='covis_curve'},flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prefix",type=Path,required=True)
    p.add_argument("--source",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    audit(a.prefix,a.source,a.out)
