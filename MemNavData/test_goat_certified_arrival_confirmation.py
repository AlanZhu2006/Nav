import unittest

from MemNavData.goat_certified_arrival_confirmation import (
    NUMPY_SEED_MODULUS,
    _service_reset_seed,
)


class GoatCertifiedArrivalConfirmationTest(unittest.TestCase):
    def test_service_seed_preserves_uint32_domain(self):
        self.assertEqual(_service_reset_seed(0), 0)
        self.assertEqual(
            _service_reset_seed(NUMPY_SEED_MODULUS - 1),
            NUMPY_SEED_MODULUS - 1,
        )

    def test_service_seed_reduces_frozen_int63_hash(self):
        value = (2 ** 63) - 17
        reduced = _service_reset_seed(value)
        self.assertEqual(reduced, value % NUMPY_SEED_MODULUS)
        self.assertGreaterEqual(reduced, 0)
        self.assertLess(reduced, NUMPY_SEED_MODULUS)

    def test_service_seed_rejects_ambiguous_values(self):
        with self.assertRaises(TypeError):
            _service_reset_seed(True)
        with self.assertRaises(ValueError):
            _service_reset_seed(-1)


if __name__ == "__main__":
    unittest.main()
