"""Private evaluation launcher: release only unused CUDA allocator blocks.

No model, weights, observation, KV cache, RNG or episode state is reset.
The coordinator calls this endpoint only while this worker is idle.
"""
import argparse
import runpy
import sys


def install(app, cuda):
    def usage():
        return dict(allocated_bytes=cuda.memory_allocated(), reserved_bytes=cuda.memory_reserved())

    def trim():
        cuda.synchronize()
        before = usage()
        cuda.empty_cache()
        after = usage()
        return dict(before=before, after=after, live_tensors_preserved=(
            before["allocated_bytes"] == after["allocated_bytes"]))

    app.add_url_rule("/private_cuda_usage", "private_cuda_usage", usage, methods=["GET"])
    app.add_url_rule("/private_cuda_trim", "private_cuda_trim", trim, methods=["POST"])


if __name__ == "__main__":
    from flask import Flask
    import torch
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entrypoint", required=True)
    args, rest = parser.parse_known_args()
    original = Flask.run

    def run(self, *a, **kw):
        install(self, torch.cuda)
        return original(self, *a, **kw)

    Flask.run = run
    sys.argv = [args.entrypoint, *rest]
    runpy.run_path(args.entrypoint, run_name="__main__")
