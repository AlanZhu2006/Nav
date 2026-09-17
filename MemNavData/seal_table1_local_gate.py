"""Bind three completed local controller pilots to the exact HPC source."""
import argparse
from pathlib import Path

from table1_repaired_eval import CONTROLLERS, ROOT, load, sha, dump


def seal(bundle, folders, output):
    if output.exists():
        raise FileExistsError(output)
    results = {}
    for folder in folders:
        manifest, summary, verified = (load(folder / n) for n in (
            "manifest.json", "summary.json", "independent_verification.json"))
        assert summary["completed"] and verified["verified"] and len(verified["records"]) == 4
        controller = verified["cell"]["controller"]
        assert controller not in results and verified["cell"] == manifest["cell"] == summary["cell"]
        for path, digest in manifest["source_hashes"].items():
            source = Path(path)
            assert sha(source) == digest == sha(bundle / source.relative_to(ROOT))
        results[controller] = dict(path=str(folder),verification_sha256=sha(folder/"independent_verification.json"),
            pairs=verified["pairs"],records=verified["records"])
    assert set(results) == set(CONTROLLERS)
    dump(output,dict(verified=True,controllers=list(CONTROLLERS),rollouts=12,
        runtime_sha256=sha(bundle / "SOURCE_BUNDLE.sha256"),results=results,
        gate="All controllers executed and independently verified; navigation success is not a gate criterion."))


if __name__ == "__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--bundle",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("folders",type=Path,nargs=3)
    a=p.parse_args()
    seal(a.bundle.resolve(),[x.resolve() for x in a.folders],a.out.resolve())
