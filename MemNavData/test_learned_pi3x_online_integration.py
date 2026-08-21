import ast
import json
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np

from MemNavData.revisit_bearing_adapter import adapt_revisit_pointgoal
from MemNavData.xnavdp_revisit_contract import pointgoal_payload


class _Response:
    def __init__(self, payload, error=None):
        self.payload = payload
        self.error = error

    def raise_for_status(self):
        if self.error is not None:
            raise self.error

    def json(self):
        return self.payload


class _Requests:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def post(self, url, **kwargs):
        self.calls.append((url, kwargs))
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def _load_route(requests):
    path = Path(__file__).with_name("eval_2leg_habitat.py")
    tree = ast.parse(path.read_text(), filename=str(path))
    function = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "srv_plan_learned_pi3x_relocalization"
    )

    def attach(nav, memnav, controller, error=None):
        return {
            **nav,
            "memnav_frame_idx": memnav.get("frame_idx"),
            "controller": controller,
            "memnav_error": error,
        }

    namespace = {
        "np": np,
        "json": json,
        "requests": requests,
        "BASE": "http://memnav",
        "NOVEL_BASE": "http://navdp",
        "args": SimpleNamespace(revisit_adapter="verified_bearing_v1"),
        "depth_png_bytes": lambda _depth: b"depth",
        "attach_memnav_diagnostics": attach,
        "adapt_revisit_pointgoal": adapt_revisit_pointgoal,
        "pointgoal_payload": pointgoal_payload,
    }
    exec(compile(ast.Module(body=[function], type_ignores=[]),
                 str(path), "exec"), namespace)
    return namespace[function.name]


class LearnedPi3XEvaluatorRouteTest(unittest.TestCase):
    probe = _Response({
        "frame_idx": 50,
        "visual_relocalization_candidates": [
            {"anchor": 12, "score": 0.95},
            {"anchor": 20, "score": 0.91},
        ],
    })

    def test_accepted_bearing_uses_mixed_navdp_at_fixed_radius(self):
        requests = _Requests([
            self.probe,
            _Response({
                "ok": True,
                "accepted": True,
                "reason": "learned_spatial_proof_accepted",
                "aux_pose": [3.0, -4.0],
                "pointgoal_units": "pi3x_current_camera_direction_only",
                "selected_anchor": 20,
                "selected_dino_rank": 2,
                "selected_overlap": 0.7,
                "ranked_candidates": [],
                "cached": False,
            }),
            _Response({"trajectory": [[0.0, 0.0, 0.0]]}),
        ])
        route = _load_route(requests)
        result = route(b"image", b"goal", np.ones((2, 2)), {}, {})

        self.assertEqual([call[0] for call in requests.calls], [
            "http://memnav/retrieval_probe_step",
            "http://memnav/learned_relocalize",
            "http://navdp/navdp_step_ip_mixgoal",
        ])
        goal_data = json.loads(requests.calls[-1][1]["data"]["goal_data"])
        point = np.asarray([
            goal_data["goal_x"][0], goal_data["goal_y"][0]
        ], dtype=float)
        self.assertAlmostEqual(float(np.linalg.norm(point)), 2.5)
        self.assertTrue(result["router_active"])
        self.assertTrue(result["revisit_adapter_takeover"])
        self.assertEqual(result["controller"], "navdp_image_point_mix")

    def test_learned_abstention_uses_native_imagegoal(self):
        requests = _Requests([
            self.probe,
            _Response({
                "ok": True,
                "accepted": False,
                "reason": "learned_spatial_proof_below_consensus_native_fallback",
                "aux_pose": None,
                "ranked_candidates": [],
                "cached": False,
            }),
            _Response({"trajectory": [[0.0, 0.0, 0.0]]}),
        ])
        result = _load_route(requests)(
            b"image", b"goal", np.ones((2, 2)), {}, {})

        self.assertEqual(requests.calls[-1][0],
                         "http://navdp/imagegoal_step")
        self.assertFalse(result["router_active"])
        self.assertFalse(result["revisit_adapter_takeover"])
        self.assertEqual(result["controller"], "navdp_image_router")

    def test_endpoint_failure_after_probe_still_calls_native_once(self):
        requests = _Requests([
            self.probe,
            RuntimeError("learned endpoint unavailable"),
            _Response({"trajectory": [[0.0, 0.0, 0.0]]}),
        ])
        result = _load_route(requests)(
            b"image", b"goal", np.ones((2, 2)), {}, {})

        self.assertEqual(requests.calls[-1][0],
                         "http://navdp/imagegoal_step")
        self.assertFalse(result["router_active"])
        self.assertEqual(result["learned_pi3x_relocalization_reason"],
                         "learned_pi3x_endpoint_failure")
        self.assertIn("RuntimeError", result["memnav_error"])


if __name__ == "__main__":
    unittest.main()
