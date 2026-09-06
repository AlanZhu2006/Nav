import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MEM = ROOT / "MemNavData"


def test_length_cache_exact_retry_contract_is_exact_and_fail_closed():
    protocol = json.loads(
        (MEM / "hm3d_table3_length_cache_exact_retry_protocol_20260902.json").read_text()
    )
    script = (
        MEM / "submit_hm3d_table3_length_cache_exact_retry_remote.sh"
    ).read_text()

    assert protocol["frozen_population"]["histories"] == 48
    assert protocol["retained_completion_count"] == 45
    assert protocol["exact_retry_indices"] == [35, 41, 44]
    assert protocol["guards"]["completed_indices_rerun"] is False
    assert protocol["guards"]["fallback_completion_allowed"] is False
    assert "--array=35,41,44%3" in script
    assert "missing != [35,41,44]" in script
    assert "navigation_success_or_distance_read':False" in script
    assert "--dependency=afterok:${repair}" in script
    assert "--dependency=afterok:${summary}" in script
    assert 'source "${WRAPPER_ROOT}/MemNavData/slurm_safe_submit.sh"' in script


def test_replacement_server_composes_authority_and_cache_receipts():
    protocol = json.loads(
        (MEM / "hm3d_table3_length_cache_exact_retry_protocol_20260902.json").read_text()
    )
    script = (
        MEM / "submit_hm3d_table3_length_cache_exact_retry_remote.sh"
    ).read_text()

    source = protocol["source_composition"]
    assert source["replacement_server_receipt_sha256"].startswith("eb7cdf82477f6aa1")
    assert "The cache is keyed by image digest" in script
    assert "authority_policy = request.form.get" in script
    assert "cmp -s \"${OLD_SERVER}/${path}\" \"${SERVER}/${path}\"" in script
