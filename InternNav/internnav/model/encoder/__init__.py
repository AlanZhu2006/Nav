"""Encoder exports loaded on demand.

LongCLIP is an optional, ignored external checkout. Importing NavDP/MemNav
encoders must not require it unless a LongCLIP-backed policy is selected.
"""

from importlib import import_module


_EXPORTS = {
    'PositionalEncoding': ('.bert_backbone', 'PositionalEncoding'),
    'DistanceNetwork': ('.distance_encoder', 'DistanceNetwork'),
    'ImageEncoder': ('.image_clip_encoder', 'ImageEncoder'),
    'InstructionEncoder': ('.instruction_encoder', 'InstructionEncoder'),
    'InstructionLongCLIPEncoder': (
        '.instruction_longCLIP_encoder', 'InstructionLongCLIPEncoder'
    ),
    'LanguageEncoder': ('.instruction_roberta_encoder', 'LanguageEncoder'),
    'VisionLanguageEncoder': ('.vision_language_encoder', 'VisionLanguageEncoder'),
}

__all__ = [*_EXPORTS, 'resnet_encoders']


def __getattr__(name):
    if name == 'resnet_encoders':
        value = import_module('.resnet_encoders', __name__)
    elif name in _EXPORTS:
        module_name, object_name = _EXPORTS[name]
        value = getattr(import_module(module_name, __name__), object_name)
    else:
        raise AttributeError(name)
    globals()[name] = value
    return value
