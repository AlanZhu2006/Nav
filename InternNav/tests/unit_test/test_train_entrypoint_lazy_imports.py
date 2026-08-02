import subprocess
import sys


def test_memnav_training_runtime_does_not_require_longclip_checkout():
    """A clean MemNav clone must load the production train runtime by itself."""
    code = """
import sys
from scripts.train.train import _load_runtime

runtime = _load_runtime('memnav')
assert runtime.model_class.__name__ == 'MemNavPolicy'
assert runtime.dataset_class.__name__ == 'MemNav_Dataset'
assert runtime.trainer_class.__name__ == 'MemNavTrainer'
assert not any(name.startswith('internnav.model.basemodel.LongCLIP') for name in sys.modules)
print('memnav-runtime-ok')
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        text=True,
        capture_output=True,
    )
    assert result.stdout.strip().endswith("memnav-runtime-ok")
