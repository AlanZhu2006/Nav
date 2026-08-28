import unittest

from cec_authority_receipt import (
    AUTHORITY_PLAN_RECEIPT_FIELDS,
    authority_plan_receipt_fields,
)


class AuthorityPlanReceiptTest(unittest.TestCase):
    def test_retains_policy_and_decision_only(self):
        response = {
            "certified_relocalization_authority": {
                "accepted": True,
                "policy": "pnp_pose_available",
            },
            "certified_relocalization_authority_policy": (
                "pnp_pose_available"),
            "trajectory": [[1.0, 2.0]],
        }
        receipt = authority_plan_receipt_fields(response)
        self.assertEqual(tuple(receipt), AUTHORITY_PLAN_RECEIPT_FIELDS)
        self.assertEqual(receipt, {
            "certified_relocalization_authority": response[
                "certified_relocalization_authority"],
            "certified_relocalization_authority_policy": (
                "pnp_pose_available"),
        })

    def test_rejects_non_mapping(self):
        with self.assertRaisesRegex(TypeError, "mapping"):
            authority_plan_receipt_fields([])


if __name__ == "__main__":
    unittest.main()
