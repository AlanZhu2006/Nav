from copy import deepcopy
import math
import pytest

from flask import Flask

from MemNavData.table2_common_goals import witness_frames, common_band, choose_common, source_address
from MemNavData.private_gpu_cache_server import install
from MemNavData.table2_continuous_paired import publish, await_record


def test_source_witness_has_original_age_and_jitter_constraints():
    poses = [dict(x=0., y=0., z=0., yaw=0.) for _ in range(56)]
    witnesses = witness_frames(poses, [.4, 0, 0], math.radians(20))
    assert len(witnesses) == 1 and witnesses[0] == pytest.approx((39, .4, 20.))
    assert witness_frames(poses[:55], [.4, 0, 0], math.radians(20)) == []
    assert witness_frames(poses, [.1, 0, 0], math.radians(20)) == []
    assert witness_frames(poses, [.4, 0, 0], math.radians(5)) == []
    assert witness_frames(poses, [.4, 0, 0], math.radians(90)) == []


def test_distance_address_is_symmetric_but_does_not_force_same_arm_bin():
    native, gem = dict(geodesic_m=3.9), dict(geodesic_m=4.1)
    assert common_band(dict(native=native, gem=gem)) == '4_to_6'
    assert common_band(dict(native=gem, gem=native)) == '4_to_6'


def test_source_address_depends_only_on_observed_frame_not_arm():
    p = dict(x=1., y=0., z=2., yaw=.1, jpg_sha256='abc')
    assert source_address(dict(p, arm='native')) == source_address(dict(p, arm='cec'))
    assert source_address(p) != source_address(dict(p, yaw=.2))


def test_selection_preserves_candidates_and_ignores_method_outcome_fields():
    menu = {'candidates': [dict(distance_band=b, goal_rgb_sha256=str(i))
                           for i, b in enumerate(('2_to_4', '4_to_6', '6_to_9'))]}
    original = deepcopy(menu)
    choice = choose_common(menu, 'episode/B', {'2_to_4': 2, '4_to_6': 0, '6_to_9': 2})
    assert choice['distance_band'] == '4_to_6'
    menu['sr'] = {'native': 1., 'cec': 0.}
    assert choose_common(menu, 'episode/B') == choose_common(original, 'episode/B')
    assert menu['candidates'] == original['candidates']
    assert choose_common({'candidates': []}, 'empty') is None


def test_allocator_trim_does_not_reset_or_free_live_state():
    class Cuda:
        reserved = 1234
        def memory_allocated(self): return 100
        def memory_reserved(self): return self.reserved
        def synchronize(self): pass
        def empty_cache(self): self.reserved = 100
    app = Flask(__name__)
    install(app, Cuda())
    with app.test_client() as client:
        assert client.get('/private_cuda_usage').json['allocated_bytes'] == 100
        receipt = client.post('/private_cuda_trim').json
        assert receipt['live_tensors_preserved'] is True
        assert receipt['before']['reserved_bytes'] == 1234
        assert receipt['after']['reserved_bytes'] == 100


def test_stage_receipt_is_complete_and_cannot_be_overwritten(tmp_path):
    p = tmp_path / 'permit_B_native.json'
    publish(p, {'query': 'same-goal'})
    assert await_record(p, tmp_path) == {'query': 'same-goal'}
    with pytest.raises(FileExistsError):
        publish(p, {'query': 'replacement'})
    assert await_record(p, tmp_path) == {'query': 'same-goal'}


def test_missing_receipt_or_coordinator_error_is_not_navigation_failure(tmp_path):
    with pytest.raises(TimeoutError):
        await_record(tmp_path / 'absent.json', tmp_path, timeout=0)
    publish(tmp_path / 'abort.json', {'error': 'infrastructure'})
    with pytest.raises(RuntimeError, match='infrastructure'):
        await_record(tmp_path / 'absent.json', tmp_path)
