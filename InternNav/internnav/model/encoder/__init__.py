"""Public encoder exports, loaded only when the caller actually needs them.

Historically this package eagerly imported every encoder.  As a consequence importing
``encoder.navdp_backbone`` for MemNav also imported the optional, gitignored Long-CLIP
checkout through ``image_clip_encoder``.  A clean worktree then failed before MemNav was
even constructed, despite never using Long-CLIP.  Lazy exports preserve the existing
``from internnav.model.encoder import ImageEncoder`` API while keeping independent model
families' optional dependencies independent.
"""

from importlib import import_module


_EXPORTS = {
    "PositionalEncoding": (".bert_backbone", "PositionalEncoding"),
    "DistanceNetwork": (".distance_encoder", "DistanceNetwork"),
    "ImageEncoder": (".image_clip_encoder", "ImageEncoder"),
    "InstructionEncoder": (".instruction_encoder", "InstructionEncoder"),
    "InstructionLongCLIPEncoder": (".instruction_longCLIP_encoder", "InstructionLongCLIPEncoder"),
    "LanguageEncoder": (".instruction_roberta_encoder", "LanguageEncoder"),
    "VisionLanguageEncoder": (".vision_language_encoder", "VisionLanguageEncoder"),
    "resnet_encoders": (".resnet_encoders", None),
}

__all__ = list(_EXPORTS)


def __getattr__(name):
    try:
        module_name, attr_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    module = import_module(module_name, __name__)
    value = module if attr_name is None else getattr(module, attr_name)
    globals()[name] = value
    return value
