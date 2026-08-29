from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SBATCH = ROOT / "MemNavData/slurm_hm3d_table1_navdp_pair.sbatch"
RUNNER = ROOT / "MemNavData/run_hm3d_fullmono_server_scene.sh"
SUBMITTER = (
    ROOT / "MemNavData/submit_hm3d_table1_controller_portability_hpc.sh"
)


def test_navdp_pair_requires_a_separately_sealed_server_overlay():
    text = SBATCH.read_text()
    assert ': "${SERVER_SOURCE_ROOT:?}" "${SERVER_SOURCE_RECEIPT:?}"' in text
    assert ': "${EXPECTED_SERVER_SOURCE_RECEIPT_SHA:?}"' in text
    assert 'SERVER_SOURCE_ROOT="${SERVER_SOURCE_ROOT}"' in text
    assert 'SERVER_SOURCE_RECEIPT="${SERVER_SOURCE_RECEIPT}"' in text
    assert 'SERVER_SOURCE_ROOT="${BASE_SOURCE_ROOT}"' not in text


def test_smoke_retry_is_additive_and_confined_to_the_formal_run():
    text = SBATCH.read_text()
    assert "NAVDP_SMOKE_RUN_ROOT:-${FORMAL_RUN_ROOT}/smoke/navdp" in text
    assert '"${FORMAL_RUN_ROOT}"/smoke/navdp*)' in text
    assert "NavDP smoke root escaped the formal run" in text


def test_primary_submitter_pins_the_verified_authority_transaction_overlay():
    text = SUBMITTER.read_text()
    assert "hm3d_table1_navdp_authority_transaction_ef4f30de3103d7af" in text
    assert (
        "EXPECTED_NAVDP_SERVER_RECEIPT_SHA="
        "ef4f30de3103d7af742137d8c63790e0f107afb880ad0650f9f98c649c05472d"
    ) in text
    assert "def append_request_frame" in text
    assert "require_monocular_depth_transaction" in text


def test_server_overlay_can_resolve_receipt_bound_base_runtime_assets():
    text = RUNNER.read_text()
    assert "${SERVER_SOURCE_ROOT}/NavDP/baselines/navdp" in text
    assert "${SERVER_SOURCE_ROOT}/NavDP/baselines/memnav" in text
    assert "${BASE_SOURCE_ROOT}/NavDP/baselines/navdp" in text
    assert "${BASE_SOURCE_ROOT}/NavDP/baselines/memnav" in text
