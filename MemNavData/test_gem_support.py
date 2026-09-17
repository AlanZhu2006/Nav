"""Information preservation and explicit unsupported-access boundaries."""
import hashlib
from pathlib import Path
from weakref import ref

import numpy as np
import pytest

from MemNavData.lingbot_pnp_localization import lift_reference_keypoints
from NavDP.baselines.memnav.gem.support import (
    FrameSupportArchive, OnlineSupportArchive, encode, decode, validate_points)


@pytest.mark.parametrize('confidence_case',['finite','partial_nan','all_nan'])
def test_arbitrary_future_matching_subsets_preserve_lift_and_validity(confidence_case):
    rng=np.random.default_rng(18)
    depth=rng.uniform(.1,8,(19,23)).astype(np.float32)
    confidence=rng.uniform(.05,.9,depth.shape).astype(np.float32)
    points=np.concatenate([rng.uniform([0,0],[22,18],(80,2)),
        [[0.,0.],[22.,18.],[22.8,18.9],[-.4,-.8],[5.,6.]]])
    if confidence_case=='partial_nan':
        confidence[::3,::4]=np.nan
        depth[::5,::6]=np.nan
    elif confidence_case=='all_nan':
        confidence[:]=np.nan
    # An actual confidence minimum away from the queried keypoint determines
    # the existing global statistic, and must not become an arbitrary sentinel.
    if confidence_case!='all_nan':
        confidence[9,11]=0.
    pose=np.array([0.,0.,0.,0.,0.,0.,1.,1.,1.])
    record=encode(depth,confidence,points)
    restored,conf=decode(record,2.3)
    for subset in (points,points[::3],points[[1,1,0,-1]]):
        expected,valid=lift_reference_keypoints(subset,depth*np.float32(2.3),confidence,pose,confidence_quantile=0.)
        actual,actual_valid=lift_reference_keypoints(subset,restored,conf,pose,confidence_quantile=0.)
        np.testing.assert_array_equal(actual_valid,valid)
        np.testing.assert_array_equal(actual,expected)


def test_missing_geometric_support_and_nonappend_writes_reject(tmp_path):
    archive=FrameSupportArchive(tmp_path/'geometry')
    depth=np.full((12,12),2.,dtype=np.float32)
    archive.append(0,depth,depth,1.,np.array([[2.3,3.7]]),rgb_sha256='b'*64)
    archive.validate_matches(0,np.array([[2.3,3.7]]))
    with pytest.raises(ValueError,match='outside'):
        archive.validate_matches(0,np.array([[8.2,9.3]]))
    with pytest.raises(ValueError,match='append'):
        archive.append(0,depth,depth,1.,np.array([[2.3,3.7]]),rgb_sha256='b'*64)
    restored,_=archive[0]
    assert np.isnan(restored[9,8])


def test_online_reference_rejects_rgb_replacement_and_changed_statistic(tmp_path):
    # Use the actual online guard with an already-written compact record; a
    # real detector/provider lifecycle is tested by the separate GPU run.
    path=tmp_path/'0.jpg'
    path.write_bytes(b'original-observed-rgb')
    archive=object.__new__(OnlineSupportArchive)
    FrameSupportArchive.__init__(archive,tmp_path/'geometry')
    archive.rgb_directory=tmp_path.resolve()
    archive.validated_match_calls=0
    points=np.array([[2.3,3.7]])
    depth=np.ones((12,12),dtype=np.float32)
    FrameSupportArchive.append(archive,0,depth,depth,1.,points,
        rgb_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    archive.validate_reference(0,path,points,confidence_quantile=0.)
    with pytest.raises(ValueError,match='statistic'):
        archive.validate_reference(0,path,points,confidence_quantile=.2)
    path.write_bytes(b'a-different-rgb')
    with pytest.raises(RuntimeError,match='differs'):
        archive.validate_reference(0,path,points,confidence_quantile=0.)
