from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SBATCH = ROOT / "MemNavData/slurm_hm3d_table1_navdp_pair.sbatch"
RUNNER = ROOT / "MemNavData/run_hm3d_fullmono_server_scene.sh"
SUBMITTER = (
    ROOT / "MemNavData/submit_hm3d_table1_controller_portability_hpc.sh"
)
PORTABILITY_RUNNER = (
    ROOT / "MemNavData/run_cec_controller_portability_smoke_local.sh"
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


def test_server_processes_have_namespace_specific_sibling_precedence():
    text = RUNNER.read_text()
    memnav_order = (
        "${SERVER_SOURCE_ROOT}/NavDP/baselines/memnav:"
        "${BASE_SOURCE_ROOT}/NavDP/baselines/memnav:"
        "${SERVER_SOURCE_ROOT}/NavDP/baselines/navdp"
    )
    navdp_order = (
        "${SERVER_SOURCE_ROOT}/NavDP/baselines/navdp:"
        "${BASE_SOURCE_ROOT}/NavDP/baselines/navdp:"
        "${SERVER_SOURCE_ROOT}/NavDP/baselines/memnav"
    )
    assert memnav_order in text
    assert navdp_order in text
    assert 'PYTHONPATH="${MEMNAV_PYTHONPATH_VALUE}"' in text
    assert 'PYTHONPATH="${NAVDP_PYTHONPATH_VALUE}"' in text
    assert 'assert hasattr(policy_agent,\\"NavDP_Agent\\")' in SUBMITTER.read_text()


def test_portability_runner_accepts_the_frozen_mp3d_replication_scope():
    text = PORTABILITY_RUNNER.read_text()
    assert "consumed_integration|paper_heldout|paper_replication" in text
    assert (
        '"${ROLE_PAIR_SCOPE}" == paper_heldout \\\n'
        '     || "${ROLE_PAIR_SCOPE}" == paper_replication'
    ) in text
    assert "bounded CEC alignment requires a frozen complete population" in text


def test_submitter_has_a_result_blind_vint_only_scope_repair():
    text = SUBMITTER.read_text()
    assert "vint_scope_repair" in text
    assert "failed_before_policy_outcomes" in text
    assert "scientific_factors_changed':False" in text
    assert "EXISTING_NAVDP_VERIFY_JOB" in text
    assert "FAILED_VINT_SMOKE_JOB" in text
    assert "vint_scope_repair_submission.json" in text
