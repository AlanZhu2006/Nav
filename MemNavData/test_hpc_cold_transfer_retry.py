from pathlib import Path
import unittest
from resume_hpc_cold_transfer_20260909 import retryable


class ReceiptRetryTests(unittest.TestCase):
    def test_only_observed_receipt_read_error_is_retryable(self):
        root = Path("/scratch/test/receipts")
        self.assertTrue(retryable(OSError(521, "transient", str(root/"001.json")), root))
        self.assertFalse(retryable(OSError(5, "I/O error", str(root/"001.json")), root))
        self.assertFalse(retryable(OSError(521, "transient", "/archive/001.tar.gz"), root))
        self.assertFalse(retryable(OSError(521, "transient", "/scratch/other/001.json"), root))


if __name__ == "__main__":
    unittest.main()
