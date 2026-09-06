from MemNavData.hm3d_table1_authority_spectrum import (
    ARMS,
    AUTHORITY_POLICY,
    DEPTH_SOURCE,
    EVALUATOR_ARM,
    HYBRID_ROUTE,
    REVISIT_ADAPTER,
    selected_arm_order,
)


def test_authority_spectrum_is_ordered_and_balanced():
    assert ARMS == (
        "mono_native",
        "mono_raw_fixed",
        "mono_unthresholded_witness",
        "mono_cec",
    )
    assert selected_arm_order(0, ARMS) == ARMS
    assert selected_arm_order(1, ARMS) == ARMS[1:] + ARMS[:1]
    assert selected_arm_order(3, ARMS) == ARMS[3:] + ARMS[:3]
    assert selected_arm_order(4, ARMS) == ARMS


def test_only_authority_changes_between_matched_proposal_arms():
    strict = "mono_cec"
    witness = "mono_unthresholded_witness"
    assert DEPTH_SOURCE[strict] == DEPTH_SOURCE[witness] == "monocular_sidecar"
    assert REVISIT_ADAPTER[strict] == REVISIT_ADAPTER[witness]
    assert EVALUATOR_ARM[strict] == "certified"
    assert EVALUATOR_ARM[witness] == "unthresholded_witness"
    assert HYBRID_ROUTE[strict] == "certified_relocalization"
    assert HYBRID_ROUTE[witness] == "certified_unthresholded_witness"
    assert AUTHORITY_POLICY == {
        witness: "pnp_pose_available",
        strict: "strict_certificate",
    }


def test_same_scene_histories_have_distinct_runtime_directories():
    import subprocess
    from pathlib import Path

    here = Path(__file__).parent
    submission = (here / "slurm_hm3d_table1_authority_spectrum.sbatch").read_text()
    assert 'RUNTIME_ATTEMPT="history_${SLURM_ARRAY_TASK_ID}"' in submission
    wrapper = (here / "run_hm3d_fullmono_server_scene.sh").read_text()
    snippet = wrapper.split("task_label=${MODE}_${SCENE_INDEX}", 1)[1]
    snippet = "task_label=${MODE}_${SCENE_INDEX}" + snippet.split(
        '[[ ! -e "${task_run}" ]]', 1)[0]
    outputs = []
    for index in (2, 3):
        output = subprocess.check_output([
            "bash", "-c", "MODE=eval; SCENE_INDEX=10; RUN_ROOT=/results; "
            f"RUNTIME_ATTEMPT=history_{index};\n" + snippet +
            '\nprintf "%s" "$task_run"',
        ], text=True)
        outputs.append(output)
    assert outputs == ["/results/runtime/eval_10_history_2",
                       "/results/runtime/eval_10_history_3"]


def test_gpu_template_uses_documented_partitions_and_one_hour():
    from pathlib import Path

    template = (Path(__file__).parent /
                "slurm_hm3d_table1_authority_spectrum.sbatch").read_text()
    assert "#SBATCH --partition=h100_tandon,a100_tandon" in template
    assert "#SBATCH --time=01:00:00" in template
    assert "h200_public" not in template
