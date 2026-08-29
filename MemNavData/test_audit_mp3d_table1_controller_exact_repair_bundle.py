import json
from pathlib import Path

import pytest

from MemNavData.audit_mp3d_table1_controller_exact_repair_bundle import audit


ROOT = Path(__file__).resolve().parents[1]


def test_repository_exact_repair_bundle_contract_passes():
    result = audit(ROOT)
    assert result["verified"] is True
    assert result["navdp_exact_histories"] == [29, 30]
    assert result["vint_exact_histories"] == [24]


def test_outcome_visibility_disclosure_is_fail_closed(tmp_path):
    target = tmp_path / "MemNavData"
    target.mkdir()
    names = (
        "mp3d_table1_controller_exact_repair_protocol_20260829.json",
        "slurm_hm3d_table1_navdp_pair.sbatch",
        "slurm_hm3d_table1_vint_pair.sbatch",
        "slurm_port_pair.sh",
        "submit_mp3d_table1_controller_exact_repair_remote.sh",
    )
    for name in names:
        (target / name).write_bytes((ROOT / "MemNavData" / name).read_bytes())
    path = target / names[0]
    payload = json.loads(path.read_text())
    payload["outcome_visibility_incident"]["occurred"] = False
    path.write_text(json.dumps(payload))
    with pytest.raises(RuntimeError, match="visibility incident omitted"):
        audit(tmp_path)
