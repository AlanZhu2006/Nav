#!/usr/bin/env python3
"""Launch a long-lived local server with audited, collision-safe port retries.

The launcher writes the selected port only after the listening socket is owned by
the spawned process tree.  This closes the check-then-bind race created by large
models that bind their HTTP port only after loading weights.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import sys
import tempfile
import time
from typing import Iterable


ADDRESS_IN_USE = (
    "address already in use",
    "port is in use by another program",
    "errno 98",
    "eaddrinuse",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-port", type=int, required=True)
    parser.add_argument("--range-start", type=int, required=True)
    parser.add_argument("--range-end", type=int, required=True)
    parser.add_argument("--stride", type=int, default=997)
    parser.add_argument("--max-attempts", type=int, default=12)
    parser.add_argument("--ready-timeout", type=float, default=480.0)
    parser.add_argument("--poll-interval", type=float, default=1.0)
    parser.add_argument("--port-file", type=Path, required=True)
    parser.add_argument("--receipt-file", type=Path, required=True)
    parser.add_argument("--log-prefix", type=Path, required=True)
    parser.add_argument("--cwd", type=Path, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.command and args.command[0] == "--":
        args.command = args.command[1:]
    if not args.command or not any("{port}" in item for item in args.command):
        parser.error("command after -- must contain a {port} placeholder")
    if not (1 <= args.range_start <= args.base_port <= args.range_end <= 65535):
        parser.error("base port must lie inside a valid inclusive port range")
    if args.stride <= 0 or args.max_attempts <= 0:
        parser.error("stride and max-attempts must be positive")
    if args.ready_timeout <= 0 or args.poll_interval <= 0:
        parser.error("timeouts must be positive")
    return args


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def candidate_port(args: argparse.Namespace, attempt: int) -> int:
    width = args.range_end - args.range_start + 1
    offset = args.base_port - args.range_start
    return args.range_start + ((offset + attempt * args.stride) % width)


def port_is_free(port: int) -> bool:
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        probe.bind(("127.0.0.1", port))
        return True
    except OSError:
        return False
    finally:
        probe.close()


def process_tree(root_pid: int) -> set[int]:
    pending = [root_pid]
    found: set[int] = set()
    while pending:
        pid = pending.pop()
        if pid in found:
            continue
        found.add(pid)
        children = Path(f"/proc/{pid}/task/{pid}/children")
        try:
            pending.extend(int(value) for value in children.read_text().split())
        except (FileNotFoundError, PermissionError, ProcessLookupError, ValueError):
            pass
    return found


def listening_socket_inodes(port: int) -> set[str]:
    answer: set[str] = set()
    for table in (Path("/proc/net/tcp"), Path("/proc/net/tcp6")):
        try:
            lines = table.read_text().splitlines()[1:]
        except (FileNotFoundError, PermissionError):
            continue
        for line in lines:
            fields = line.split()
            if len(fields) < 10 or fields[3] != "0A":
                continue
            try:
                local_port = int(fields[1].rsplit(":", 1)[1], 16)
            except (IndexError, ValueError):
                continue
            if local_port == port:
                answer.add(fields[9])
    return answer


SOCKET_LINK = re.compile(r"socket:\[(\d+)\]")


def process_socket_inodes(pids: Iterable[int]) -> set[str]:
    answer: set[str] = set()
    for pid in pids:
        fd_root = Path(f"/proc/{pid}/fd")
        try:
            descriptors = list(fd_root.iterdir())
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        for descriptor in descriptors:
            try:
                target = os.readlink(descriptor)
            except (FileNotFoundError, PermissionError, ProcessLookupError):
                continue
            match = SOCKET_LINK.fullmatch(target)
            if match:
                answer.add(match.group(1))
    return answer


def listener_owned_by(process: subprocess.Popen[bytes], port: int) -> bool:
    listeners = listening_socket_inodes(port)
    if not listeners:
        return False
    owned = process_socket_inodes(process_tree(process.pid))
    return bool(listeners & owned)


def tail(path: Path, limit: int = 16000) -> str:
    try:
        data = path.read_bytes()
    except FileNotFoundError:
        return ""
    return data[-limit:].decode("utf-8", errors="replace")


def terminate_process_group(process: subprocess.Popen[bytes], grace: float = 8.0) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=grace)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait()


def main() -> int:
    args = parse_args()
    args.cwd.mkdir(parents=True, exist_ok=True)
    args.log_prefix.parent.mkdir(parents=True, exist_ok=True)
    args.port_file.unlink(missing_ok=True)
    args.receipt_file.unlink(missing_ok=True)
    child: subprocess.Popen[bytes] | None = None
    stopping = False

    def stop(signum: int, _frame: object) -> None:
        nonlocal stopping
        stopping = True
        print(f"launcher signal={signum}", flush=True)
        if child is not None and child.poll() is None:
            try:
                os.killpg(child.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)

    for attempt in range(args.max_attempts):
        if stopping:
            return 128 + signal.SIGTERM
        port = candidate_port(args, attempt)
        if not port_is_free(port):
            print(f"launcher skip attempt={attempt} port={port} reason=occupied", flush=True)
            continue
        rendered = [item.replace("{port}", str(port)) for item in args.command]
        attempt_log = Path(f"{args.log_prefix}.attempt_{attempt:02d}.log")
        print(
            f"launcher start attempt={attempt} port={port} log={attempt_log}",
            flush=True,
        )
        with attempt_log.open("wb") as log_handle:
            child = subprocess.Popen(
                rendered,
                cwd=args.cwd,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        deadline = time.monotonic() + args.ready_timeout
        while not stopping:
            returncode = child.poll()
            if returncode is not None:
                output_tail = tail(attempt_log)
                collision = any(token in output_tail.lower() for token in ADDRESS_IN_USE)
                if collision:
                    print(
                        f"launcher retry attempt={attempt} port={port} "
                        f"returncode={returncode} reason=address_in_use",
                        flush=True,
                    )
                    child = None
                    break
                print(output_tail, file=sys.stderr, flush=True)
                print(
                    f"launcher fatal attempt={attempt} port={port} "
                    f"returncode={returncode}",
                    file=sys.stderr,
                    flush=True,
                )
                return returncode if returncode != 0 else 2
            if listener_owned_by(child, port):
                receipt = {
                    "schema_version": "retrying_server_launcher_v1_20260814",
                    "ready_at_utc": utc_now(),
                    "attempt": attempt,
                    "port": port,
                    "launcher_pid": os.getpid(),
                    "server_pid": child.pid,
                    "log": str(attempt_log),
                    "command": rendered,
                }
                atomic_text(args.receipt_file, json.dumps(receipt, indent=2, sort_keys=True) + "\n")
                atomic_text(args.port_file, f"{port}\n")
                print(
                    f"launcher ready attempt={attempt} port={port} pid={child.pid}",
                    flush=True,
                )
                returncode = child.wait()
                print(
                    f"launcher server_exit port={port} returncode={returncode}",
                    flush=True,
                )
                if stopping:
                    return 128 + signal.SIGTERM
                return returncode
            if time.monotonic() >= deadline:
                terminate_process_group(child)
                print(tail(attempt_log), file=sys.stderr, flush=True)
                print(
                    f"launcher fatal attempt={attempt} port={port} reason=ready_timeout",
                    file=sys.stderr,
                    flush=True,
                )
                return 2
            time.sleep(args.poll_interval)
        if stopping:
            if child is not None:
                terminate_process_group(child)
            return 128 + signal.SIGTERM

    print(
        f"launcher fatal reason=all_ports_failed attempts={args.max_attempts}",
        file=sys.stderr,
        flush=True,
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
