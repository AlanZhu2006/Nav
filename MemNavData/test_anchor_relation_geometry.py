import math
import numpy as np
import pytest
import torch
from PIL import Image

from MemNavData.anchor_relation_decoder import AnchorRelationDecoder
from MemNavData.anchor_relation_geometry import (
    depth_to_patch_geometry, intrinsics_from_pose_fov, padded_image_mask, pooled_fraction,
)
from MemNavData.prepare_anchor_relation_geometry_inputs import prefix_requirements, array_digest


def test_pad_matches_lingbot_actual_preprocessor(tmp_path):
    import sys
    sys.path.insert(0,"/home/asus/Research/Nav/NavDP/baselines/memnav/lingbot-map")
    from lingbot_map.utils.load_fn import load_and_preprocess_images
    for height,width in [(480,640),(640,480),(512,512)]:
        path=tmp_path/f"black_{height}_{width}.png"
        Image.new("RGB",(width,height),(0,0,0)).save(path)
        result=load_and_preprocess_images([str(path)],mode="pad",image_size=518,patch_size=14)[0]
        assert torch.equal(result[0]==0,padded_image_mask((height,width)))


def test_projection_and_scale_gauge_invariance():
    depth=torch.full((518,518),2.)
    mask=padded_image_mask((480,640))
    pose=[0,0,0,0,0,0,1,math.pi/2,math.pi/2]
    k=intrinsics_from_pose_fov(pose)
    a=depth_to_patch_geometry(depth,torch.ones_like(depth),k,mask,3.)
    b=depth_to_patch_geometry(depth*7,torch.ones_like(depth),k,mask,3./7)
    np.testing.assert_allclose(a['xyz_normalized'],b['xyz_normalized'],atol=1e-6)
    assert b['normalization_m']==pytest.approx(a['normalization_m'])
    assert a['roundtrip_projection_max_px']<.001
    assert np.all(a['xyz_normalized'][a['valid_mask'],2]==1)


def test_padding_depth_does_not_change_geometry_or_scale():
    mask=padded_image_mask((480,640))
    clean=torch.full((518,518),2.)
    dirty=clean.clone();dirty[~mask]=float('nan')
    k=np.array([[200,0,259],[0,200,259],[0,0,1.]])
    a=depth_to_patch_geometry(clean,torch.ones_like(clean),k,mask,2.)
    b=depth_to_patch_geometry(dirty,torch.ones_like(clean),k,mask,2.)
    np.testing.assert_array_equal(a['xyz_normalized'],b['xyz_normalized'])
    assert a['depth_median_raw']==b['depth_median_raw']==2


def test_nonpositive_scale_and_empty_depth_rejected():
    d=torch.ones(28,28)
    k=np.eye(3)
    with pytest.raises(ValueError,match='causal metric scale'):
        depth_to_patch_geometry(d,d,k,d.bool(),0,grid_size=2)
    with pytest.raises(ValueError,match='no valid'):
        depth_to_patch_geometry(d*0,d,k,d.bool(),1,grid_size=2)


def test_masked_geometry_and_visual_values_have_no_effect():
    torch.manual_seed(3)
    model=AnchorRelationDecoder(feature_dim=16,width=16,layers=1,heads=4,
                               requires_geometry=True,dropout=0).eval()
    g=torch.randn(2,16,16);a=torch.randn_like(g);x=torch.randn(2,16,3)
    mask=torch.ones(2,16,dtype=torch.bool);mask[:,0:4]=False
    with torch.no_grad():
        original=model(g,a,x,goal_valid=mask,anchor_valid=mask)
        g[~mask]=500;a[~mask]=-500;x[~mask]=float('nan')
        changed=model(g,a,x,goal_valid=mask,anchor_valid=mask)
    torch.testing.assert_close(changed,original,rtol=0,atol=0)


def test_prefix_union_keeps_anchors_and_decision_boundaries_separate():
    def pair(anchor,decision):
        return {'candidate_relative_path':f's/e/videos/chunk-000/observation.images.rgb/{anchor}.jpg',
                'candidate_frame':anchor,'decision_frame':decision,'pair_id':str(anchor)}
    requirements=prefix_requirements([pair(10,90),pair(150,200)])['s/e']
    assert requirements['frame_count']==151
    assert requirements['minimum_decision']==90
    assert sorted(requirements['anchors'])==[10,150]
    with pytest.raises(ValueError,match='invalid historical'):
        prefix_requirements([pair(90,90)])


def test_array_hash_matches_existing_scale_contract():
    from MemNavData.external_causal_scale_contract import ndarray_sha256
    x=np.arange(64*9,dtype=np.float32).reshape(64,9)
    assert array_digest(x)==ndarray_sha256(x)
