"""Bind declared HM3D gate sources, without reading navigation outcomes."""
import argparse
import hashlib
import json
from pathlib import Path


PARENT_SHA = "a96a0b96fab7b7b47709b36cb8eeb9410b42b09f095f87ef01304a68de716dd5"
SCENES = ("rJhMRvNn4DS", "6D36GQHuP8H")
EXTENSION_SCENES = ("nrA1tAA17Yp", "jgPBycuV1Jq", "BFRyYbPCCPE", "X7gTkoDHViv")


def selected_sources(parent, phase="initial", shard=0):
    if phase == "initial":
        if shard != 0 or tuple(parent["scenes"][:2]) != SCENES:
            raise ValueError("Frozen initial source order changed")
        return tuple(enumerate(SCENES))
    if phase != "extension" or shard not in (0, 1):
        raise ValueError("Only the two frozen extension shards are allowed")
    if tuple(parent["scenes"][2:6]) != EXTENSION_SCENES:
        raise ValueError("Frozen extension source order changed")
    start = 2 + 2 * shard
    return tuple((rank, parent["scenes"][rank]) for rank in range(start, start + 2))


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def build(parent_path, protocol_path, *, phase="initial", shard=0):
    if digest(parent_path) != PARENT_SHA:
        raise ValueError("Unexpected parent population")
    parent = json.loads(parent_path.read_text())
    sources = []
    for scene_rank, scene in selected_sources(parent, phase, shard):
        row = parent["episodes"][scene][0]
        if row["episode"] != "episode_0000":
            raise ValueError("Unexpected first source episode")
        asset = parent["assets"][scene]
        files = {asset["glb_path"]: asset["glb_sha256"],
                 **{v["path"]: v["sha256"] for v in row["files"].values()},
                 str(parent_path.resolve()): PARENT_SHA,
                 str(protocol_path.resolve()): digest(protocol_path)}
        for path, expected in files.items():
            if digest(path) != expected:
                raise ValueError(f"Source content changed: {path}")
        episode = Path(parent["paths"]["generated_root"]) / scene / row["episode"]
        metadata = json.loads((episode / "meta/gen_meta.json").read_text())
        goal = episode / "videos/chunk-000/observation.images.rgb" / f"{int(metadata['switch_idx'])-1}.jpg"
        files[str(goal)] = digest(goal)
        sources.append({"scene": scene, "episode": row["episode"],
                        "asset": asset["glb_path"], "source_episode": str(episode),
                        "source_episode_root": str(episode.parent.parent),
                        "seed": 2026082200 + 100 * scene_rank,
                        "source_files": files, "source_scene_rank": scene_rank})
    return {"schema": "repaired_hm3d_two_source_gate_v1_20260908",
            "scope": ("two consumed HM3D sources: runtime integration gate, not confirmation"
                      if phase == "initial" else
                      "fixed four-source extension, two-source shard: runtime gate, not confirmation"),
            "selection": ("first source of parent scenes 0 and 1; no old A or query outcomes read"
                          if phase == "initial" else
                          "first source of parent scenes 2 through 5; no old A or query outcomes read"),
            "phase": phase, "shard": shard,
            "parent_manifest_sha256": PARENT_SHA, "sources": sources}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("parent", type=Path)
    parser.add_argument("protocol", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--phase", choices=("initial", "extension"), default="initial")
    parser.add_argument("--shard", type=int, choices=(0, 1), default=0)
    args = parser.parse_args()
    payload = build(args.parent.resolve(), args.protocol.resolve(), phase=args.phase, shard=args.shard)
    with args.output.open("x") as stream:
        stream.write(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"source_count": len(payload["sources"]),
                      "sources": [(s["scene"], s["episode"], s["seed"]) for s in payload["sources"]]}))
