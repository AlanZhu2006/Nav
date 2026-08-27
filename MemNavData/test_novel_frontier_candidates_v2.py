import copy
import math
import unittest

import numpy as np

from MemNavData.novel_frontier_candidates_v2 import (
    FrontierConfig,
    FrontierProposalError,
    PlanarScan,
    ProxyMeasurement,
    SE2Pose,
    attach_proposal_proxy_labels,
    canonical_sha256,
    deployment_shortlist,
    depth_to_planar_scan,
    deterministic_spatial_bearing_nms,
    generate_frontier_proposals,
    invalid_proposal,
)


def synthetic_scans():
    bearings = tuple(math.radians(value) for value in (
        -60, -40, -20, 0, 20, 40, 60))
    return [
        PlanarScan(
            frame_index=frame,
            pose=SE2Pose(x_m, 0.0, 0.0),
            ranges_m=(3.0,) * len(bearings),
            bearings_rad=bearings,
            hit=(True,) * len(bearings),
        )
        for frame, x_m in enumerate((0.0, 0.3, 0.6, 0.9))
    ]


def candidate(candidate_id, *, x, y, bearing, topology, patch=None):
    return {
        "candidate_id": candidate_id,
        "map_xy_m": [float(x), float(y)],
        "subgoal_forward_m": float(x),
        "subgoal_left_m": float(y),
        "distance_m": float(math.hypot(x, y)),
        "bearing_rad": float(bearing),
        "frontier_normal_bearing_rad": float(bearing),
        "resolution_m": 0.2,
        "grid_cell": [round(x / 0.2), round(y / 0.2)],
        "frontier_boundary_m": 1.0,
        "frontier_novelty_m": 0.5,
        "clearance_lower_m": 0.5,
        "topology_score": float(topology),
        "context_frame_indices": [0],
        "goal_patch_relation_score": float(patch or 0.0),
        "goal_patch_relation_present": patch is not None,
        "selection_sources": [],
        "source_scales_m": [0.2],
    }


class FakeProxyLabeler:
    def __init__(self, positive_ids=()):
        self.positive_ids = set(positive_ids)

    def provenance(self):
        return {
            "kind": "fake_unit_test_only",
            "pathfinder_used": False,
        }

    def label(self, *, sample_id, arm, candidate):
        del sample_id, arm
        return ProxyMeasurement(
            reachable=True,
            progress_m=(1.0 if candidate["candidate_id"] in self.positive_ids
                        else -1.0),
        )


class MutatingProxyLabeler(FakeProxyLabeler):
    def label(self, *, sample_id, arm, candidate):
        candidate["topology_score"] = 999.0
        return super().label(
            sample_id=sample_id, arm=arm, candidate=candidate)


class NovelFrontierCandidatesV2Test(unittest.TestCase):
    def test_multiscale_union_is_deterministic_and_bounded(self):
        scans = synthetic_scans()
        first = generate_frontier_proposals(scans, scans[-1].pose)
        second = generate_frontier_proposals(scans, scans[-1].pose)
        self.assertEqual(first, second)
        self.assertEqual(
            [row["resolution_m"] for row in first["scale_summaries"]],
            [0.15, 0.20, 0.30],
        )
        self.assertGreater(first["raw_candidate_count"], 0)
        self.assertGreater(first["nms_candidate_count"], 0)
        self.assertLessEqual(first["shortlist_count"], 6)
        self.assertTrue(all(
            set(row["source_scales_m"]) <= {0.15, 0.20, 0.30}
            for row in first["candidate_universe"]
        ))

    def test_depth_ray_uses_forward_left_sign(self):
        depth = np.full((10, 11), 2.0, dtype=np.float32)
        intrinsic = np.asarray([
            [10.0, 0.0, 5.0],
            [0.0, 10.0, 5.0],
            [0.0, 0.0, 1.0],
        ])
        scan = depth_to_planar_scan(
            depth, intrinsic, SE2Pose(0.0, 0.0, 0.0), 0,
            column_stride=5)
        self.assertGreater(scan.bearings_rad[0], 0.0)  # left image -> left
        self.assertAlmostEqual(scan.bearings_rad[1], 0.0)
        self.assertLess(scan.bearings_rad[2], 0.0)

    def test_invalid_depth_is_skipped_and_saturation_is_not_an_obstacle(self):
        intrinsic = np.asarray([
            [10.0, 0.0, 5.0],
            [0.0, 10.0, 5.0],
            [0.0, 0.0, 1.0],
        ])
        depth = np.zeros((10, 11), dtype=np.float64)
        with self.assertRaisesRegex(FrontierProposalError, "no valid"):
            depth_to_planar_scan(
                depth, intrinsic, SE2Pose(0.0, 0.0, 0.0), 0,
                valid_mask=np.zeros_like(depth, dtype=bool),
                truncated_mask=np.zeros_like(depth, dtype=bool),
                column_stride=5,
            )

        saturated = np.full((10, 11), 6.5535, dtype=np.float64)
        scan = depth_to_planar_scan(
            saturated, intrinsic, SE2Pose(0.0, 0.0, 0.0), 0,
            valid_mask=np.ones_like(saturated, dtype=bool),
            truncated_mask=np.ones_like(saturated, dtype=bool),
            column_stride=5,
        )
        self.assertTrue(scan.ranges_m)
        self.assertFalse(any(scan.hit))

    def test_spatial_and_bearing_nms_are_order_independent(self):
        rows = [
            candidate("best", x=1.0, y=0.0, bearing=0.0, topology=3.0),
            candidate("spatial", x=1.1, y=0.0, bearing=0.2, topology=2.0),
            candidate("same_ray", x=1.6, y=0.05, bearing=0.01, topology=1.5),
            candidate("other", x=0.0, y=1.5, bearing=math.pi / 2,
                      topology=1.0),
        ]
        forward = deterministic_spatial_bearing_nms(rows)
        reverse = deterministic_spatial_bearing_nms(list(reversed(rows)))
        self.assertEqual(forward, reverse)
        self.assertEqual(
            [row["candidate_id"] for row in forward[0]], ["best", "other"])
        self.assertEqual(
            {row["reason"] for row in forward[1]},
            {"spatial", "bearing_radial"},
        )

    def test_shortlist_uses_exact_three_deployment_slots(self):
        rows = [
            candidate(
                f"c{index}",
                x=1.0 + index * 0.2,
                y=(-1) ** index * (0.3 + index * 0.2),
                bearing=-1.2 + index * 0.3,
                topology=8 - index,
                patch=(0.1 * index if index in (5, 6, 7) else None),
            )
            for index in range(8)
        ]
        shortlist = deployment_shortlist(rows)
        self.assertLessEqual(len(shortlist), 6)
        sources = {
            source
            for row in shortlist
            for source in row["selection_sources"]
        }
        self.assertEqual(sources, {
            "goal_patch_top2", "topology_top2", "angular_diverse_top2"})
        self.assertEqual(
            [row["candidate_id"] for row in shortlist[:2]], ["c7", "c6"])

    def test_missing_patch_is_masked_and_does_not_invent_patch_slot(self):
        scans = synthetic_scans()
        report = generate_frontier_proposals(scans, scans[-1].pose)
        self.assertEqual(report["goal_patch_relation_mask"], 0)
        self.assertFalse(report["goal_patch_relation_present"])
        self.assertTrue(all(
            not row["goal_patch_relation_present"]
            and row["goal_patch_relation_score"] == 0.0
            and "goal_patch_top2" not in row["selection_sources"]
            for row in report["shortlist"]
        ))

    def test_patch_scores_cannot_reference_future_frames(self):
        scans = synthetic_scans()
        with self.assertRaisesRegex(FrontierProposalError, "future"):
            generate_frontier_proposals(
                scans, scans[-1].pose, patch_scores_by_frame={9: 0.5})

    def test_proxy_labels_are_separate_and_cannot_change_selection(self):
        scans = synthetic_scans()
        proposal = generate_frontier_proposals(scans, scans[-1].pose)
        frozen = copy.deepcopy(proposal)
        candidate_id = proposal["candidate_universe"][0]["candidate_id"]
        report = attach_proposal_proxy_labels(
            sample_id="scene/episode/state/goal",
            arm="teacher_pose",
            proposal=proposal,
            labeler=FakeProxyLabeler([candidate_id]),
        )
        self.assertEqual(proposal, frozen)
        self.assertEqual(report["proposal_sha256"], canonical_sha256(proposal))
        self.assertTrue(report["universe_has_positive"])
        proposal_text = str(proposal).lower()
        self.assertNotIn("progress_m", proposal_text)
        self.assertNotIn("reachable", proposal_text)

    def test_unreachable_proxy_cannot_carry_progress(self):
        with self.assertRaisesRegex(FrontierProposalError, "zero progress"):
            ProxyMeasurement(reachable=False, progress_m=1.0)

    def test_proxy_mutation_is_isolated_from_frozen_proposal(self):
        scans = synthetic_scans()
        proposal = generate_frontier_proposals(scans, scans[-1].pose)
        frozen = copy.deepcopy(proposal)
        attach_proposal_proxy_labels(
            sample_id="sample",
            arm="teacher_pose",
            proposal=proposal,
            labeler=MutatingProxyLabeler(),
        )
        self.assertEqual(proposal, frozen)

    def test_invalid_proposal_has_no_candidates(self):
        report = invalid_proposal("missing_ground_h_est")
        self.assertFalse(report["valid"])
        self.assertEqual(report["candidate_universe"], [])
        self.assertEqual(report["shortlist"], [])

    def test_configuration_rejects_non_protocol_scale(self):
        with self.assertRaisesRegex(FrontierProposalError, "resolutions"):
            FrontierConfig(resolutions_m=(0.2,))


if __name__ == "__main__":
    unittest.main()
