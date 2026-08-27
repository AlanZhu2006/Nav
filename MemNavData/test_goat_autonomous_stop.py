import pytest

from MemNavData.goat_autonomous_stop import (
    AutonomousVisualStopSearch,
    SearchDisposition,
    TerminalSearchConfig,
    arrival_evidence_from_payload,
    build_terminal_view_schedule,
    schedule_net_discrete_rotation,
)
from MemNavData.goat_certified_arrival_contract import ArrivalEvidence


def rejected_evidence(frame_count=80):
    return ArrivalEvidence(
        native_zero_proposal=True,
        stream_frame_count=frame_count,
        certificate_accepted=False,
        predicted_distance_m=None,
        metric_scale_available=False,
    )


def accepted_evidence(distance=0.05, frame_count=80):
    return ArrivalEvidence(
        native_zero_proposal=True,
        stream_frame_count=frame_count,
        certificate_accepted=True,
        predicted_distance_m=distance,
        metric_scale_available=True,
    )


def test_novel_schedule_is_one_closed_360_degree_sweep():
    schedule = build_terminal_view_schedule()
    assert len(schedule) == 12
    assert {item.action for item in schedule} == {"turn_right"}
    assert schedule_net_discrete_rotation(schedule) == (12, 0)


def test_revisit_alignment_is_directed_then_scans_then_restores():
    schedule = build_terminal_view_schedule(
        revisit_yaw_right_deg=-88.0,
        revisit_pitch_up_deg=29.0,
    )
    assert [item.action for item in schedule[:4]] == [
        "turn_left", "turn_left", "turn_left", "look_up"]
    assert [item.phase for item in schedule[4:16]] == [
        "full_yaw_sweep"] * 12
    assert [item.action for item in schedule[-4:]] == [
        "look_down", "turn_right", "turn_right", "turn_right"]
    # Twelve right turns are one full revolution; directed motion is inverted.
    assert schedule_net_discrete_rotation(schedule) == (12, 0)


def test_false_arrival_exhausts_search_without_stop():
    search = AutonomousVisualStopSearch()
    decisions = []
    while not search.finished:
        decisions.append(search.observe(rejected_evidence()))
    assert decisions[-1].disposition is SearchDisposition.REPLAN
    assert all(item.disposition is not SearchDisposition.STOP
               for item in decisions)
    assert search.motion_count == 12
    assert search.probe_count == 13  # initial view plus twelve turned views


def test_geometric_metric_certificate_is_the_only_stop_authority():
    search = AutonomousVisualStopSearch(revisit_yaw_right_deg=180.0)
    first = search.observe(rejected_evidence())
    assert first.disposition is SearchDisposition.MOTION

    # An accepted geometric match beyond the frozen metric threshold cannot stop.
    second = search.observe(accepted_evidence(distance=0.20))
    assert second.disposition is SearchDisposition.MOTION
    assert not search.authorized_stop

    third = search.observe(accepted_evidence(distance=0.05))
    assert third.disposition is SearchDisposition.STOP
    assert search.authorized_stop
    assert third.action is None
    with pytest.raises(RuntimeError):
        search.observe(accepted_evidence())


def test_payload_projection_never_uses_role_or_ground_truth_distance():
    evidence = arrival_evidence_from_payload({
        "frame_count": 91,
        "certificate_accepted": True,
        "predicted_distance_m": 0.04,
        "metric_scale_available": True,
        "analysis_role": "novel",
        "official_distance_to_goal": 8.0,
    })
    assert evidence == accepted_evidence(distance=0.04, frame_count=91)


def test_invalid_nonclosing_scan_is_rejected():
    with pytest.raises(ValueError):
        build_terminal_view_schedule(
            config=TerminalSearchConfig(full_yaw_steps=8))
