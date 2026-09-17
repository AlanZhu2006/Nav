"""Read-only local verification of every published A query and its source."""
import argparse
from collections import Counter
from pathlib import Path

import numpy as np

from MemNavData.table2_mixed_local import load, dump, sha, runtime_spec
from MemNavData.table2_balanced_sampling import CELLS, cell_of, novel_supported_as_task
from MemNavData.table2_sampling_profiles import role_cells, verify_query_direction


def verify(payload, out):
    from PIL import Image
    published = load(payload/"published_sources.json")
    declared = load(payload/"source_declaration.json")
    remote = Path(published["remote_payload"])
    assert published["source_declaration_sha256"] == sha(payload/"source_declaration.json")
    assert len(published["sources"]) == 144 and published["source_scenes"] == 18
    assert len({s["source_id"] for s in published["sources"]}) == 144
    counts, checked = Counter(), []
    for source, original in zip(published["sources"], declared["sources"]):
        assert source["source_id"] == original["source_id"]
        assert source["carrier_source_id"] == original["carrier_source_id"]
        selection = source["a_selection"]
        if selection is None:
            continue
        path = payload/Path(selection["query"]).relative_to(remote)
        assert sha(path) == selection["query_sha256"]
        q = load(path)
        assert q["schema"] == published["schema"]
        assert (q["scene"],q["episode"],q["seed"]) == (source["scene"],source["episode"],source["seed"])
        assert q["stage_number"] == 0 and q["prefix_root"] is None and q["history_frames"] == 0
        for k in ("start_position","start_yaw","camera_height_m","camera_intrinsic"):
            assert q[k] == original["initial_state"][k]
        assert np.asarray(q["camera_intrinsic"]).shape == (3,3)
        assert q["camera_height_m"] == .5
        assert q["cell"] == cell_of(q) == selection["cell"]
        verify_query_direction(q, published["schema"])
        assert novel_supported_as_task(q["goal_surface_points"],q["covis_curve"],q["current_view_covis"])
        rgb = payload/Path(q["goal_rgb"]).relative_to(remote)
        assert sha(rgb) == q["goal_rgb_sha256"]
        image = Image.open(rgb)
        assert image.mode == "RGB" and image.size == (480,270)
        runtime = load(path.with_name("runtime.json"))
        assert runtime == runtime_spec(q)
        assert not {"analysis_role","covis_curve","cell","direction_stratum"}.intersection(runtime)
        counts[q["cell"]] += 1
        checked.append(dict(source_id=source["source_id"],cell=q["cell"],query_sha256=sha(path),
                            goal_rgb_sha256=sha(rgb),current_view_covis=q["current_view_covis"]))
    assert {c:counts[c] for c in role_cells(published["schema"], "novel")} == published["A_assignment"]["counts"]
    for line in (payload/"PAYLOAD.sha256").read_text().splitlines():
        digest, relative = line.split("  ",1)
        assert sha(payload/relative) == digest
    result = dict(verified=True,queries=len(checked),scenes=published["source_scenes"],
        cell_counts=dict(counts), max_initial_covis=max(r["current_view_covis"] for r in checked),
        all_sources_retained=len(published["sources"])==len(declared["sources"]),
        unconstructible_sources=[s["source_id"] for s in published["sources"] if s["a_selection"] is None],
        no_navigation_performed=True,
        payload_receipt_sha256=sha(payload/"PAYLOAD.sha256"), records=checked)
    dump(out,result)
    print({k:v for k,v in result.items() if k!="records"})


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--payload",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()
    verify(args.payload,args.out)
