import unittest
from types import SimpleNamespace

from scripts.eval.eval_memnav_offline import _dataset_cache_contract


class MemNavOfflineEvalContractTest(unittest.TestCase):
    def test_versioned_cache_contract_is_forwarded_without_weakening(self):
        config = SimpleNamespace(il=SimpleNamespace(
            strict_feature_coverage=True,
            require_versioned_cache=True,
            expected_cache_signature='audited-signature',
            require_generated_pose_convention=True,
        ))

        self.assertEqual(_dataset_cache_contract(config), {
            'strict_feature_coverage': True,
            'require_versioned_cache': True,
            'expected_cache_signature': 'audited-signature',
            'require_generated_pose_convention': True,
        })

    def test_legacy_local_contract_remains_explicit(self):
        config = SimpleNamespace(il=SimpleNamespace())

        self.assertEqual(_dataset_cache_contract(config), {
            'strict_feature_coverage': True,
            'require_versioned_cache': False,
            'expected_cache_signature': '',
            'require_generated_pose_convention': False,
        })


if __name__ == '__main__':
    unittest.main()
