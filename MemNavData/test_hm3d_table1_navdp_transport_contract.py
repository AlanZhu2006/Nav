from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SBATCH = ROOT / "MemNavData/slurm_hm3d_table1_navdp_pair.sbatch"
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


def test_primary_submitter_pins_the_verified_transaction_overlay():
    text = SUBMITTER.read_text()
    assert "hm3d_fullmono_transaction_repair_67e1132783ce2cb1" in text
    assert (
        "EXPECTED_NAVDP_SERVER_RECEIPT_SHA="
        "05ce401aac8c2e7e31e8a8d820613d30b3a03856a35c8750085b93d5a1539a97"
    ) in text
    assert "def append_request_frame" in text
    assert "require_monocular_depth_transaction" in text
