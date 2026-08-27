#!/usr/bin/env python3

import unittest

from build_hm3d_fullmono_lifelong_natural_v4_b_shards import build


class NaturalV4BShardTest(unittest.TestCase):
    def test_shards_partition_candidates_by_scene(self):
        manifest = {"episodes": [
            {"scene": "s0", "final14_scene_rank": 0,
             "lifelong_construction": {"recipient_episode": "a0"}},
            {"scene": "s0", "final14_scene_rank": 0,
             "lifelong_construction": {"recipient_episode": "a0"}},
            {"scene": "s0", "final14_scene_rank": 0,
             "lifelong_construction": {"recipient_episode": "a1"}},
            {"scene": "s1", "final14_scene_rank": 2,
             "lifelong_construction": {"recipient_episode": "b0"}},
        ]}
        result = build(
            manifest, manifest_sha256="a" * 64,
            maximum_histories_per_shard=2,
        )
        self.assertEqual(result["candidate_histories"], 4)
        self.assertEqual(result["source_recipient_histories"], 3)
        self.assertEqual(result["scene_clusters"], 2)
        self.assertEqual(result["shard_count"], 3)
        self.assertEqual(
            [row["history_indices"] for row in result["shards"]],
            [[0, 1], [2], [3]],
        )


if __name__ == "__main__":
    unittest.main()
