"""Clone a frozen runtime with only three private-server transport hooks."""
import argparse
import json
from pathlib import Path
import shutil

from prepare_repaired_fullmono_bundle import ROOT, digest, seal

CHANGED = (
    "lingbot_pose_diagnostic_server.py",
    "navdp_depth_raster_audit_server.py",
    "image_controller_repaired_server.py",
)
EXTRA = (
    "multipart_crlf_repair.py", "test_multipart_crlf_repair.py",
    "audit_table1_multipart_repair.py", "prepare_multipart_repair_bundle.py",
    "submit_table1_multipart_repair.sh", "TABLE1_MULTIPART_REPAIR_PROTOCOL_20260911.md",
)


def build(original, destination, table2=False):
    manifest = original / "SOURCE_BUNDLE.sha256"
    rows = [line.split("  ", 1) for line in manifest.read_text().splitlines()]
    assert all(digest(original / name) == value for value, name in rows)
    destination.mkdir(parents=True, exist_ok=False)
    changes = {}
    for value, name in rows:
        source = original / name
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if name in {"MemNavData/" + item for item in CHANGED}:
            replacement = ROOT / name
            before, after = source.read_text(), replacement.read_text()
            hook = ("    from MemNavData.multipart_crlf_repair import install as install_transport\n"
                    "    install_transport()\n" if "navdp_depth_raster" in name else
                    "    from MemNavData.multipart_crlf_repair import install\n    install()\n")
            assert after.count(hook) == 1 and after.replace(hook, "", 1) == before
            shutil.copy2(replacement, target)
            changes[name] = dict(before=value, after=digest(target))
        else:
            shutil.copy2(source, target)
            assert digest(target) == value
    for name in EXTRA:
        target = destination / "MemNavData" / name
        assert not target.exists()
        shutil.copy2(ROOT / "MemNavData" / name, target)
    if table2:
        # Bookkeeping only: report the actual repair bundle separately from the
        # immutable source/goal plan. Completed A archives retain their identity.
        for name in ("table2_mixed_hpc.py", "preflight_table2_mixed_hpc.py"):
            target = destination / "MemNavData" / name
            before = digest(target)
            target.chmod(target.stat().st_mode | 0o200)
            shutil.copy2(ROOT / "MemNavData" / name, target)
            changes["MemNavData/" + name] = dict(before=before, after=digest(target))
    with (destination / "multipart_repair_manifest.json").open("x") as stream:
        json.dump(dict(original_bundle=str(original), original_bundle_sha256=digest(manifest),
            changed_original_files=changes, preserved_original_files=len(rows)-len(changes),
            added_files=list(EXTRA), model_parameters_changed=False,
            controller_changed=False, jpeg_validation_removed=False), stream, indent=2)
    seal(destination)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("original", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--table2", action="store_true")
    args = parser.parse_args()
    build(args.original.resolve(), args.destination.resolve(), table2=args.table2)
