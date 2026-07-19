"""Trainer exports loaded on demand.

Some legacy trainers require optional LongCLIP assets. Importing an unrelated
trainer must not make those assets a prerequisite for MemNav diagnostics/tests.
"""

from importlib import import_module


_EXPORTS = {
    'CMATrainer': ('.cma_trainer', 'CMATrainer'),
    'RDPTrainer': ('.rdp_trainer', 'RDPTrainer'),
    'NavDPTrainer': ('.navdp_trainer', 'NavDPTrainer'),
    'LoGoPlannerTrainer': ('.logoplanner_trainer', 'LoGoPlannerTrainer'),
    'MemNavTrainer': ('.memnav_trainer', 'MemNavTrainer'),
}

__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    module_name, object_name = _EXPORTS[name]
    value = getattr(import_module(module_name, __name__), object_name)
    globals()[name] = value
    return value
