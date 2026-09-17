"""A separate C interface probe using an already consumed actual-mono A/B.

This cannot populate the new-A Table II or claim a balanced C source mix.
The predeclared successful old native B is used to test branch restoration,
not selected using either of the new C outcomes.
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np

from MemNavData.table2_mixed_local import (
    HERE, ROOT, load, dump, sha, base_source, query_command, runtime_spec,
)

OLD = ROOT / ".diagnostics/repaired_fullmono_local_20260908/e2e_v1"
SCRIPT = Path(__file__).resolve()
SCENE = "pLe4wQe7qrG"


def run(out, mem_port, nav_port, reference_run=None, b_role="revisit"):
    from MemNavData.run_repaired_fullmono_local import (
        sources, private_servers, run_child, execution_environment, HAB_PY, hab_env, source_files,
    )
    out.mkdir(parents=True, exist_ok=False)
    (out/"logs").mkdir()
    source = sources()[1]
    assert source["scene"] == SCENE
    supplied_prefix = None
    if reference_run:
        assert load(reference_run/"verification/independent_verification.json")["verified"]
        choices = [r for r in load(reference_run/"summary.json")["C_sources"]
                   if r["scene"]==SCENE and r["b_role"]==b_role and r["b_reached"]]
        chosen, = choices
        supplied_prefix = Path(chosen["prefix"])
        receipt = load(supplied_prefix/"receipt.json")
        old_prefix = Path(receipt["parent_prefix"])
        old_b = Path(receipt["source_rollout"])/"actual_trace.json"
        trace = load(old_b)
    else:
        old_prefix = OLD / "construction" / SCENE / "online_a" / SCENE / "episode_0000"
        old_b = OLD / "evaluation" / SCENE / "novel/native/episode_0000_pair_00_novel_plans.json"
        original = load(old_b)
        trace = original["query_trace_payload"]
    assert trace["reached"] and trace["source_hybrid_route"] == "native_sidecar"
    old_a = load(old_prefix/"online_a_trace.json")
    if supplied_prefix:
        from MemNavData.table2_mixed_local import compose_prefix
        assert load(supplied_prefix/"online_a_trace.json")==compose_prefix(old_a,trace)
    else:
        assert original["rollout_traces"]["legA"] == old_a["poses"]
        assert trace["poses"] == original["rollout_traces"]["query"]
    spec = base_source(source, 1)
    spec["seed"] = trace["episode_seed"]
    dump(out/"source.json", spec)
    dump(out/"old_B_trace_copy/actual_trace.json", trace)
    prefix = supplied_prefix or out/"prefix_AB_novel"
    manifest = dict(scope="one successful actual native A+B prefix; C interface only; NOT balanced Table II",
        old_prefix=str(old_prefix), old_B=str(old_b), old_B_sha256=sha(old_b),
        old_A_sha256=sha(old_prefix/"online_a_trace.json"), formal_population_frozen=False,
        C_source_balance_claimed=False, new_A_rollouts=0, new_B_rollouts=0,
        evaluated_prefix=str(prefix), reference_run=str(reference_run) if reference_run else None,
        old_initialization=reference_run is None,
        reference_verification_sha256=sha(reference_run/"verification/independent_verification.json") if reference_run else None,
        runtime_profile="bounded_standard/rgb_v1/source_rgb/heading_on; canonical GEM",
        code_sha256={str(p):sha(p) for p in (HERE,SCRIPT)},
        code_and_weights={str(p):sha(p) for p in source_files()})
    dump(out/"manifest.json", manifest)
    if supplied_prefix is None:
        run_child([HAB_PY,"-u",str(HERE),"materialize","--source",str(out/"source.json"),
                   "--prefix",str(old_prefix),"--rollout",str(out/"old_B_trace_copy"),"--out",str(prefix)],
                  out/"logs/materialize.log", environment=hab_env())
    goals=out/"C_goals"
    run_child([HAB_PY,"-u",str(HERE),"construct","--source",str(out/"source.json"),
               "--stage","C","--prefix",str(prefix),"--out",str(goals)],
              out/"logs/construct.log",environment=hab_env())
    construction=load(goals/"construction.json")
    if reference_run:
        previous=load(reference_run/SCENE/f"C_after_{b_role}_goals/construction.json")
        if "novel" in previous["queries"]:
            old_goal=load(previous["queries"]["novel"])
            new_goal=load(construction["queries"]["novel"])
            for key in ("goal_rgb_sha256","floor_position","yaw_rad","prefix_trace_sha256"):
                assert old_goal[key]==new_goal[key],key
    dump(out/"queries_before_eval.json",dict(queries=construction["queries"],
         construction_sha256=sha(goals/"construction.json"),C_outcomes_read=False,
         scope=manifest["scope"]))
    records=[]
    if construction["queries"]:
        with private_servers(out,mem_port,nav_port):
            for j, role in enumerate(("novel","revisit")):
                if role not in construction["queries"]:
                    continue
                path=Path(construction["queries"][role])
                query=load(path)
                for arm in (("native","cec") if j==0 else ("cec","native")):
                    folder=out/"evaluation"/role/arm
                    pose_log=out/"lingbot_pose_readout.jsonl"
                    offset=pose_log.stat().st_size if pose_log.exists() else 0
                    print(f"START consumed C/{role}/{arm}",flush=True)
                    env=dict(execution_environment(),TABLE2_QUERY=str(path.with_name("runtime.json")))
                    wall=run_child(query_command(source,query,folder,mem_port,nav_port,arm),
                                   out/"logs"/f"eval_{role}_{arm}.log",environment=env)
                    with pose_log.open() as f:
                        f.seek(offset)
                        dump(folder/"lingbot_frame_poses.json",[json.loads(x) for x in f if x.strip()])
                    terminal,=load(folder/"terminal_measurements.json")
                    row=dict(terminal,scene=SCENE,role=role,arm=arm,directory=str(folder),
                             query_file=str(path),wall_seconds=wall)
                    dump(folder/"result.json",row)
                    records.append(row)
                    print(f"DONE consumed C/{role}/{arm}: SR={row['reached']} SPL={row['spl']:.3f}",flush=True)
    changed=[p for p,h in manifest["code_and_weights"].items() if sha(p)!=h]
    dump(out/"summary.json",dict(completed=True,records=records,scope=manifest["scope"],
         changed_source_files=changed,
         formal_result=False,constructed_roles=list(construction["queries"])))
    run_child([HAB_PY,"-u",str(SCRIPT),"verify","--out",str(out)],out/"logs/verify.log",environment=hab_env())


def verify(out):
    from MemNavData.verify_repaired_fullmono_local import verify_rollout
    from MemNavData.table2_mixed_local import compose_prefix
    manifest, summary = load(out/"manifest.json"),load(out/"summary.json")
    assert not summary.get("changed_source_files",[])
    assert sha(manifest["old_B"])==manifest["old_B_sha256"]
    a=load(Path(manifest["old_prefix"])/"online_a_trace.json")
    b=load(out/"old_B_trace_copy/actual_trace.json")
    expected=compose_prefix(a,b)
    prefix=Path(manifest.get("evaluated_prefix",str(out/"prefix_AB_novel")))
    assert load(prefix/"online_a_trace.json")==expected
    verified,indexed=[],{}
    for row in summary["records"]:
        result,evidence,plans,actions=verify_rollout(row)
        query=load(row["query_file"])
        folder=Path(row["directory"])
        assert load(folder/"query_receipt.json")["query"]==runtime_spec(query)
        replay=load(folder/"prefix_replay.json")
        assert replay["online_frames"]==len(expected["poses"]) and replay["diffusion_samples_during_replay"]==0
        np.testing.assert_allclose([evidence["rollout_trace"][0][k] for k in "xyz"],expected["end_position"],atol=1e-8,rtol=0)
        assert evidence["rollout_trace"][0]["yaw"]==expected["end_yaw"]
        indexed[(row["role"],row["arm"])]=(row,plans,actions,replay)
        verified.append(result)
    pairs=[]
    for role in summary["constructed_roles"]:
        n,g=(indexed[(role,arm)] for arm in ("native","cec"))
        assert n[3]==g[3] and n[0]["first_query_rgb_sha256"]==g[0]["first_query_rgb_sha256"]
        take=any(p["receipt"]["revisit_adapter_takeover"] is True for p in g[1])
        if not take:
            assert len(n[1])==len(g[1]) and len(n[2])==len(g[2])
            for x,y in zip(n[1],g[1]):
                for key in ("selected_trajectory","all_trajectory","all_values","position","yaw"):
                    np.testing.assert_array_equal(x[key],y[key])
            for x,y in zip(n[2],g[2]):
                assert x["actual_position"]==y["actual_position"] and x["actual_yaw"]==y["actual_yaw"]
        pairs.append(dict(role=role,native=n[0]["reached"],cec=g[0]["reached"],
                          takeover=take,reject_exact_native=not take))
    dump(out/"independent_verification.json",dict(verified=True,records=verified,pairs=pairs,
         actual_prefix_frames=len(expected["poses"]),old_initialization=manifest.get("old_initialization",True),
         formal_result=False,C_source_balance_claimed=False))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode",choices=("run","verify"))
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--memnav-port",type=int,default=19781)
    p.add_argument("--navdp-port",type=int,default=19782)
    p.add_argument("--reference-run",type=Path)
    p.add_argument("--b-role",choices=("novel","revisit"),default="revisit")
    args=p.parse_args()
    if args.mode=="run":
        run(args.out.resolve(),args.memnav_port,args.navdp_port,
            args.reference_run.resolve() if args.reference_run else None,args.b_role)
    else:
        verify(args.out.resolve())
