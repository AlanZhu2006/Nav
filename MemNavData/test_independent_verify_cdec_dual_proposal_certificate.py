import json
import unittest

from MemNavData.independent_verify_cdec_dual_proposal_certificate import (
    CDEC_ORIGIN,
    GEOMETRY_ORIGIN,
    center_pnp,
    has_certificate,
    is_actionable,
    reconstruct,
    verify_subset,
)


def record(*, certificate: bool, actionable: bool, frame: int, scene: str = "s"):
    return {
        "session_id": f"session_{frame}",
        "scene": scene,
        "candidate_frame": frame,
        "candidate_path": f"/{frame}.jpg",
        "teacher_candidate_label": int(actionable),
        "session_has_positive": actionable,
        "session_is_strict_no_match": not actionable,
        "certificate": certificate,
        "actionable": actionable,
        "certified_actionable": certificate and actionable,
        "certificate_false_positive": certificate and not actionable,
        "pnp_status": "ok" if certificate else "failed",
        "pnp_inliers": 20 if certificate else 0,
    }


class IndependentCDECVerifierTest(unittest.TestCase):
    def test_structural_certificate_and_actionability_are_independent(self) -> None:
        pnp = {
            "status": "ok",
            "inliers": 16,
            "query_inlier_coverage": 0.05,
            "reference_inlier_coverage": 0.05,
            "reprojection_rmse_px": 2.0,
            "relative_position_error_m": 0.75,
        }
        self.assertTrue(has_certificate(pnp))
        self.assertTrue(is_actionable(pnp))
        pnp["relative_position_error_m"] = 0.76
        self.assertTrue(has_certificate(pnp))
        self.assertFalse(is_actionable(pnp))

    def test_center_parser_requires_exactly_one_zero_offset(self) -> None:
        payload = json.dumps(
            [{"offset": -1}, {"offset": 0, "pnp_lightglue": {"status": "ok"}}]
        )
        self.assertEqual(center_pnp(payload)["status"], "ok")
        with self.assertRaises(RuntimeError):
            center_pnp(json.dumps([{"offset": 1}]))

    def test_geometry_first_cascade_has_safe_complementary_gain(self) -> None:
        geometry = {
            "a": record(certificate=True, actionable=True, frame=1),
            "b": record(certificate=False, actionable=True, frame=2),
        }
        learned = {
            "a": record(certificate=True, actionable=False, frame=3),
            "b": record(certificate=True, actionable=True, frame=4),
        }
        result = reconstruct(geometry, learned)
        contrast = result["paired"][
            "geometry_first_cascade_minus_geometry_certified_actionable"
        ]
        self.assertEqual(contrast["gains"], 1)
        self.assertEqual(contrast["losses"], 0)
        self.assertTrue(result["method_gate"]["pass"])

    def test_subset_comparison_fails_on_official_mismatch(self) -> None:
        verify_subset({"a": {"b": 1}}, {"a": {"b": 1, "c": 2}})
        with self.assertRaisesRegex(RuntimeError, "a.b"):
            verify_subset({"a": {"b": 1}}, {"a": {"b": 2}})

    def test_origins_remain_frozen(self) -> None:
        self.assertEqual(GEOMETRY_ORIGIN, "lightglue_fundamental_rank_v1")
        self.assertEqual(CDEC_ORIGIN, "cdec_scene_oof_pairwise_rank_v1")


if __name__ == "__main__":
    unittest.main()
