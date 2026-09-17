"""Attach read-only final verification to the existing Table-II stage DAG."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


FILES = ("table2_final_postprocess.py", "slurm_table2_final_postprocess.sbatch",
         "independent_verify_table2_full.py", "report_table2_full.py")
NEXT = {"after_A": ("B", "after_B"), "after_B": ("C", "verify")}


def load(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze(root):
    contents = "".join(f"{sha(root/name)}  {name}\n" for name in FILES)
    with (root/"POSTPROCESSING.sha256").open("x") as stream:
        stream.write(contents)


def next_dependency(stage, submission, plan_sha):
    expected, next_stage = NEXT[stage]
    if submission["TABLE2_STAGE"] != expected or submission["EXPECTED_PLAN_SHA"] != plan_sha:
        raise ValueError("Downstream submission belongs to another stage or plan")
    return int(submission["TABLE2_REDUCE_JOB"]), next_stage


def dependency_arguments(job):
    # Torch retains finished scheduler records for only 64 s. A CPU handoff
    # can start later than that; a verified successful parent needs no wait.
    rows = subprocess.check_output(["sacct", "-X", "-nP", "-j", str(int(job)),
        "--format=State,ExitCode"], text=True).strip().splitlines()
    if len(rows) != 1:
        raise ValueError("Cannot establish the reducer's actual state")
    state, code = rows[0].split("|")[:2]
    if state == "COMPLETED" and code == "0:0":
        return []
    if state in ("PENDING", "RUNNING", "CONFIGURING", "COMPLETING", "SUSPENDED"):
        return [f"--dependency=afterok:{int(job)}", "--kill-on-invalid-dep=yes"]
    raise ValueError(f"Reducer did not complete successfully: {state} {code}")


def submit(root, bundle, stage, dependency):
    large = stage == "verify"
    export = (f"ALL,TABLE2_POST_RUN={root},TABLE2_POST_BUNDLE={bundle},"
              f"TABLE2_POST_STAGE={stage},TABLE2_POST_PLAN_SHA={sha(root/'plan.json')},"
              f"TABLE2_POST_CODE_SHA={sha(root/'POSTPROCESSING.sha256')}")
    arguments = ["--parsable", "--partition=cpu_short", "--account=torch_pr_769_tandon_advanced",
        "--cpus-per-task="+("2" if large else "1"), "--mem="+("16G" if large else "1G"),
        "--time="+("01:00:00" if large else "00:10:00"),
        *dependency_arguments(dependency),
        f"--job-name=table2_post_{stage}", "--export="+export,
        str(root/"slurm_table2_final_postprocess.sbatch")]
    command = ["bash", "-c", 'source "$1"; shift; safe_sbatch --lint-fatal "$@"',
        "table2-post-submit", str(bundle/"MemNavData/slurm_safe_submit.sh"), *arguments]
    job = subprocess.check_output(command, text=True).strip()
    return int(job.split(";")[0])


def run(root, bundle, stage):
    plan_sha = sha(root/"plan.json")
    if os.environ["TABLE2_POST_PLAN_SHA"] != plan_sha:
        raise ValueError("The frozen experiment plan changed")
    if sha(bundle/"SOURCE_BUNDLE.sha256") != load(root/"plan.json")["runtime_sha256"]:
        raise ValueError("Wrong source bundle for the postprocessing handoff")
    out = root/"postprocessing"/f"{stage}.json"
    if out.exists():
        raise FileExistsError(out)
    receipt = dict(stage=stage, plan_sha256=plan_sha,
        postprocessing_sha256=sha(root/"POSTPROCESSING.sha256"),
        started_at=time.time(), job_id=os.environ["SLURM_JOB_ID"])
    if stage in NEXT:
        predecessor_stage, _ = NEXT[stage]
        submitted = load(root/f"{predecessor_stage}_submission.json")
        parent, child_stage = next_dependency(stage, submitted, plan_sha)
        child = submit(root, bundle, child_stage, parent)
        receipt.update(next_stage=child_stage, next_job_id=child, dependency_job=parent)
        print(f"SUBMITTED {child_stage} job={child} afterok={parent}", flush=True)
    else:
        py = "/scratch/lg154/conda-envs/memnav/bin/python"
        summary = root/"paired_summary.json"
        verified = root/"full_independent_verification.json"
        subprocess.run([py, "-u", str(root/"independent_verify_table2_full.py"),
            "--plan", str(root/"plan.json"), "--run", str(root), "--summary", str(summary),
            "--out", str(verified)], check=True)
        subprocess.run([py, str(root/"report_table2_full.py"), "--summary", str(summary),
            "--verification", str(verified), "--out", str(root/"TABLE2_FULL_RESULT.md")], check=True)
        receipt.update(verified=load(verified)["verified"], verification_sha256=sha(verified),
                       report_sha256=sha(root/"TABLE2_FULL_RESULT.md"))
    receipt["finished_at"] = time.time()
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x") as stream:
        json.dump(receipt, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--stage", choices=("freeze", "after_A", "after_B", "verify"), required=True)
    args = parser.parse_args()
    if args.stage == "freeze":
        freeze(args.root)
    else:
        if args.bundle is None:
            parser.error("--bundle is required to run a DAG stage")
        run(args.root, args.bundle, args.stage)
