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
    assert "hm3d_table1_navdp_authority_transaction_718661db1733d5de" in text
    assert (
        "EXPECTED_NAVDP_SERVER_RECEIPT_SHA="
        "718661db1733d5de16cd86687eec880a8d02fc5ae5ca982e1ab7d5bde5e96f7d"
    ) in text
    assert "def append_request_frame" in text
    assert "require_monocular_depth_transaction" in text
    assert "def causal_goal_support_indices" in text
    assert "import policy_agent,router_candidates" in text


def test_server_overlay_can_resolve_receipt_bound_base_runtime_assets():
    text = RUNNER.read_text()
    assert "${SERVER_SOURCE_ROOT}/NavDP/baselines/navdp" in text
    assert "${SERVER_SOURCE_ROOT}/NavDP/baselines/memnav" in text
    assert "${BASE_SOURCE_ROOT}/NavDP/baselines/navdp" in text
    assert "${BASE_SOURCE_ROOT}/NavDP/baselines/memnav" in text
