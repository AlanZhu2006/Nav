import hashlib
import io
import json
import tarfile

import numpy as np
import pytest

from MemNavData.run_raw_match_verification import raw_coordinates, statistics, unpack_bundle


def test_pixel_centres_rescale_without_padding():
    xy = np.array([[0., 0.], [511., 287.]])
    expected = (xy+.5) * 480/512 - .5
    np.testing.assert_allclose(raw_coordinates(xy, (480, 270), (288, 512)), expected)


def test_square_official_crop_is_inverted():
    # Official square_ok=False retains the central 512x384, not the entire square.
    xy = np.array([[255.5, 191.5]])
    np.testing.assert_allclose(raw_coordinates(xy, (640, 640), (384, 512)), [[319.5, 319.5]])


def test_confidence_is_not_clipped_to_a_probability():
    assert statistics([1., 3., 5.])['median'] == 3.
    assert statistics([])['count'] == 0
    with pytest.raises(ValueError, match='Nonfinite'):
        statistics([float('nan')])


def bundle(path, name, data, expected):
    with tarfile.open(path, 'w:gz') as tar:
        for n, d in [(name, data), ('input_files.json', json.dumps([
            {'path': name, 'bytes': len(data), 'sha256': expected}]).encode())]:
            info = tarfile.TarInfo(n)
            info.size = len(d)
            tar.addfile(info, io.BytesIO(d))


def test_bundle_checksum_and_no_overwrite(tmp_path):
    source = tmp_path / 'bundle.tgz'
    data = b'original RGB bytes'
    bundle(source, 'pairs/a.jpg', data, hashlib.sha256(data).hexdigest())
    out = tmp_path / 'out'
    assert unpack_bundle(source, out)['verified']
    assert (out / 'pairs/a.jpg').read_bytes() == data
    with pytest.raises(FileExistsError):
        unpack_bundle(source, out)


def test_corrupt_input_is_not_used(tmp_path):
    source = tmp_path / 'bundle.tgz'
    bundle(source, 'pairs/a.jpg', b'changed', 'wrong')
    with pytest.raises(ValueError, match='Changed'):
        unpack_bundle(source, tmp_path / 'out')


def test_bundle_cannot_write_outside_run(tmp_path):
    source = tmp_path / 'bundle.tgz'
    bundle(source, '../a.jpg', b'x', hashlib.sha256(b'x').hexdigest())
    with pytest.raises(ValueError, match='Unsafe'):
        unpack_bundle(source, tmp_path / 'out')
