"""NavDP-side monocular sidecar transaction cache semantics.

Regression for the 2026-08-22 fresh-DAG failure: a blocked/stationary agent
produces byte-identical JPEGs at different stream positions, so the NavDP
depth cache (keyed by image digest) collided with the OLDER append's
transaction and raised "cached monocular depth belongs to a different
transaction" (reproduced twice on 019_66seV3BWPoX_episode_0002, different
arms).  The amendment refetches by the caller's exact token instead of
failing, while every genuine mismatch still fails closed.
"""

import hashlib
import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "NavDP" / "baselines" / "navdp"))
sys.path.insert(0, str(REPO))

import navdp_server  # noqa: E402

from MemNavData.monocular_depth_runtime import (  # noqa: E402
    bind_monocular_depth_transaction,
    build_monocular_depth_payload,
    image_sha256,
)


JPEG = b"identical-blocked-frame-bytes"


def _bound_payload(frame_index):
    payload = build_monocular_depth_payload(
        relative_depth=None,
        depth_shape=(3, 4),
        image_sha256_value=image_sha256(JPEG),
        frame_index=frame_index,
        scale_receipt=None,
    )
    return bind_monocular_depth_transaction(payload)


class _FakeResponse:
    def __init__(self, payload, status=200):
        self._payload = payload
        self.ok = status < 400
        self.status_code = status
        self.text = "" if self.ok else "absent or superseded"

    def json(self):
        return self._payload


class StationaryFrameTransactionTest(unittest.TestCase):
    def setUp(self):
        navdp_server.monocular_depth_cache = {}
        self._source = navdp_server.active_depth_source
        navdp_server.active_depth_source = "monocular_sidecar"
        self._url = navdp_server.args.monocular_depth_url
        navdp_server.args.monocular_depth_url = "http://sidecar.test"
        self.transactions = {}
        self.post_calls = []

        def fake_post(url, data=None, timeout=None):
            self.post_calls.append(dict(data))
            token = data.get("monocular_depth_transaction_token")
            payload = self.transactions.get(token)
            if payload is None:
                return _FakeResponse({}, status=409)
            return _FakeResponse(payload)

        self._patcher = mock.patch.object(
            navdp_server.requests, "post", side_effect=fake_post
        )
        self._patcher.start()

    def tearDown(self):
        self._patcher.stop()
        navdp_server.active_depth_source = self._source
        navdp_server.args.monocular_depth_url = self._url
        navdp_server.monocular_depth_cache = {}

    def _resolve(self, token, frame_index):
        image_bgr = np.zeros((3, 4, 3), dtype=np.uint8)
        return navdp_server._resolve_observation_depth(
            JPEG,
            image_bgr,
            None,
            1,
            transaction_token=token,
            expected_frame_index=frame_index,
        )

    def test_identical_jpeg_new_transaction_refetches_instead_of_failing(self):
        first = _bound_payload(10)
        second = _bound_payload(11)
        tok1 = first["monocular_depth_transaction_token"]
        tok2 = second["monocular_depth_transaction_token"]
        self.assertNotEqual(tok1, tok2)
        self.transactions[tok1] = first
        self.transactions[tok2] = second

        depth1, receipt1 = self._resolve(tok1, 10)
        self.assertEqual(int(receipt1["frame_index"]), 10)
        # The stationary robot appends the SAME bytes as frame 11: this used
        # to raise "cached monocular depth belongs to a different
        # transaction"; it must now serve frame 11's own transaction.
        depth2, receipt2 = self._resolve(tok2, 11)
        self.assertEqual(int(receipt2["frame_index"]), 11)
        self.assertEqual(len(self.post_calls), 2)
        self.assertTrue(np.array_equal(depth1, depth2))

    def test_same_transaction_repeat_reads_cache_without_refetch(self):
        first = _bound_payload(10)
        tok1 = first["monocular_depth_transaction_token"]
        self.transactions[tok1] = first
        self._resolve(tok1, 10)
        self._resolve(tok1, 10)
        self.assertEqual(len(self.post_calls), 1)

    def test_genuine_mismatch_still_fails_closed(self):
        first = _bound_payload(10)
        tok1 = first["monocular_depth_transaction_token"]
        self.transactions[tok1] = first
        self._resolve(tok1, 10)
        # A token the sidecar has never bound: refetch happens and the
        # sidecar's 409 must surface as a hard error, never a silent reuse.
        with self.assertRaises(RuntimeError):
            self._resolve("f" * 64, 11)


if __name__ == "__main__":
    unittest.main()
