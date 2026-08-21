#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
import unittest


ROOT = Path(__file__).resolve().parent
LAUNCHER = ROOT / "retrying_server_launcher.py"


def free_block(width: int = 4) -> int:
    for base in range(24000, 62000 - width):
        probes: list[socket.socket] = []
        try:
            for port in range(base, base + width):
                probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                probes.append(probe)
                probe.bind(("127.0.0.1", port))
            return base
        except OSError:
            pass
        finally:
            for probe in probes:
                probe.close()
    raise RuntimeError("no free contiguous port block")


class RetryingServerLauncherTest(unittest.TestCase):
    def test_retries_bind_race_and_publishes_owned_listener(self) -> None:
        base = free_block()
        server_code = (
            "import http.server,sys; "
            "port=int(sys.argv[1]); first=int(sys.argv[2]); "
            "(print('Address already in use',flush=True),sys.exit(1)) "
            "if port==first else None; "
            "http.server.ThreadingHTTPServer(('127.0.0.1',port),"
            "http.server.SimpleHTTPRequestHandler).serve_forever()"
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            port_file = root / "server.port"
            receipt_file = root / "server.json"
            process = subprocess.Popen(
                [
                    sys.executable,
                    str(LAUNCHER),
                    "--base-port",
                    str(base),
                    "--range-start",
                    str(base),
                    "--range-end",
                    str(base + 3),
                    "--stride",
                    "1",
                    "--max-attempts",
                    "4",
                    "--ready-timeout",
                    "5",
                    "--poll-interval",
                    "0.05",
                    "--port-file",
                    str(port_file),
                    "--receipt-file",
                    str(receipt_file),
                    "--log-prefix",
                    str(root / "server"),
                    "--cwd",
                    str(root),
                    "--",
                    sys.executable,
                    "-c",
                    server_code,
                    "{port}",
                    str(base),
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            try:
                deadline = time.monotonic() + 10
                while time.monotonic() < deadline and not port_file.exists():
                    if process.poll() is not None:
                        output = process.communicate()[0]
                        self.fail(f"launcher exited before ready:\n{output}")
                    time.sleep(0.05)
                self.assertTrue(port_file.exists())
                self.assertEqual(int(port_file.read_text()), base + 1)
                receipt = json.loads(receipt_file.read_text())
                self.assertEqual(receipt["attempt"], 1)
                self.assertEqual(receipt["port"], base + 1)
                with socket.create_connection(("127.0.0.1", base + 1), timeout=2):
                    pass
            finally:
                if process.poll() is None:
                    process.terminate()
                process.communicate(timeout=10)


if __name__ == "__main__":
    unittest.main()
