"""Public trainer exports, imported only for the selected model family.

Several legacy trainers require the optional, gitignored Long-CLIP checkout.  Eagerly
importing every trainer made a clean MemNav deployment depend on Long-CLIP even though
MemNav never uses it.  Lazy exports preserve the existing public API while keeping each
model family's dependencies isolated.
"""

from importlib import import_module


_EXPORTS = {
    "CMATrainer": (".cma_trainer", "CMATrainer"),
    "RDPTrainer": (".rdp_trainer", "RDPTrainer"),
    "NavDPTrainer": (".navdp_trainer", "NavDPTrainer"),
    "LoGoPlannerTrainer": (".logoplanner_trainer", "LoGoPlannerTrainer"),
    "MemNavTrainer": (".memnav_trainer", "MemNavTrainer"),
}

__all__ = list(_EXPORTS)


def __getattr__(name):
    try:
        module_name, attr_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    module = import_module(module_name, __name__)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value
