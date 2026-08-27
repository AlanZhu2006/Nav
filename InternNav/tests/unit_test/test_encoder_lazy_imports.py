import subprocess
import sys


def test_navdp_backbone_does_not_require_optional_longclip_checkout():
    # Use a fresh interpreter so sys.modules from other tests cannot mask an eager
    # package import.  This worktree deliberately has only the tracked LongCLIP symlink
    # and no gitignored Long-CLIP checkout.
    code = (
        "from internnav.model.encoder.navdp_backbone import TokenCompressor; "
        "print(TokenCompressor.__name__)"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        text=True,
        capture_output=True,
    )
    assert result.stdout.strip().endswith("TokenCompressor")
