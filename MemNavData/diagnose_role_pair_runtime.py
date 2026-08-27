#!/usr/bin/env python3
"""Run the frozen role-pair evaluator with opt-in wall-time instrumentation.

No method inputs or decisions are changed.  Wrappers only time the client-side
render, HTTP planning/memory calls, geodesic query, and pursuit update.  JSONL
events go to stderr so they survive a native Habitat abort before summaries are
written.
"""

from __future__ import annotations

import faulthandler
import functools
import json
import sys
import time
import traceback

import eval_shared_online_role_pairs as evaluation


def instrument(name: str, function):
    calls = 0

    @functools.wraps(function)
    def wrapped(*args, **kwargs):
        nonlocal calls
        call = calls
        calls += 1
        started = time.perf_counter()
        status = "ok"
        try:
            return function(*args, **kwargs)
        except BaseException:
            status = "error"
            raise
        finally:
            event = {
                "event": "runtime_timing",
                "name": name,
                "call": call,
                "elapsed_s": time.perf_counter() - started,
                "status": status,
            }
            print(json.dumps(event, sort_keys=True), file=sys.stderr, flush=True)

    return wrapped


def main() -> None:
    faulthandler.enable(all_threads=True)
    faulthandler.dump_traceback_later(60.0, repeat=True)
    base = evaluation.base
    for name in ("render", "srv_memory", "srv_plan", "pursuit_step", "geodesic"):
        setattr(base, name, instrument(name, getattr(base, name)))
    try:
        evaluation.main()
    except BaseException:
        traceback.print_exc()
        raise
    finally:
        faulthandler.cancel_dump_traceback_later()


if __name__ == "__main__":
    main()
