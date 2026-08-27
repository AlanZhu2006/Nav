#!/usr/bin/env python3

import os
import subprocess
import sys
import unittest
from pathlib import Path

import finalize_hm3d_fullmono_lifelong_ab as finalizer


class LightweightFinalizerImportTest(unittest.TestCase):
    def test_cpu_seal_does_not_import_renderer_dependencies(self):
        root = Path(__file__).resolve().parent
        environment = dict(os.environ)
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        # The immutable repair bundle intentionally contains only the changed
        # seal files.  Keep its directory first, but retain the frozen source
        # bundles supplied by the seal job for lightweight audit/contract
        # dependencies.
        inherited_path = environment.get("PYTHONPATH", "")
        environment["PYTHONPATH"] = os.pathsep.join(
            path for path in (str(root), inherited_path) if path
        )
        subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import sys; "
                    "import finalize_hm3d_fullmono_lifelong_ab; "
                    "assert 'build_final14_role_pair_scene' not in sys.modules; "
                    "assert 'generate_twoleg' not in sys.modules; "
                    "assert 'quaternion' not in sys.modules"
                ),
            ],
            cwd=root,
            env=environment,
            check=True,
        )
        self.assertTrue(callable(finalizer.finalize))


if __name__ == "__main__":
    unittest.main()
