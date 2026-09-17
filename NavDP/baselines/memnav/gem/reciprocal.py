"""Reciprocal coordinate transport between repeated observed camera sets.

The shared images have identical camera and pixel identities. Model depth is
uncertain on both sides: symmetric log-depth differences determine the scale
change, while camera rotations and centers determine the rigid relationship.
The resulting transform reverses exactly when the two contexts are swapped.
This is a structural property, not a guarantee of metric accuracy or a loop
closure detector.
"""
import numpy as np

from .bindings import immutable_similarity


def reciprocal_camera_transport(old_poses, new_poses, old_depth, new_depth, valid_pixels):
    old, new = np.asarray(old_poses), np.asarray(new_poses)
    old_z, new_z = np.asarray(old_depth), np.asarray(new_depth)
    if old.shape != new.shape or old.ndim != 3 or old.shape[1:] != (4,4) or len(old)==0:
        raise ValueError('Expected nonempty paired observed camera poses')
    if old_z.shape != new_z.shape or old_z.ndim != 2 or len(old_z)!=len(old):
        raise ValueError('Expected paired same-pixel depth observations')
    if not np.isfinite(old).all() or not np.isfinite(new).all():
        raise ValueError('Nonfinite camera poses')
    valid_pixels = np.asarray(valid_pixels,dtype=bool)
    if valid_pixels.shape != (old_z.shape[1],):
        raise ValueError('Pixel mask does not match the repeated images')
    covariance = np.sum(old[:,:3,:3] @ new[:,:3,:3].transpose(0,2,1),axis=0)
    u,singular,vt = np.linalg.svd(covariance)
    if singular[1] <= singular[0]*1e-10:
        raise ValueError('Shared camera orientations do not identify a rotation')
    rotation = (u*np.array([1.,1.,np.linalg.det(u@vt)]))@vt
    frame_log_scales,counts,scatter = [],[],[]
    for a,b in zip(old_z,new_z):
        valid=valid_pixels & np.isfinite(a) & np.isfinite(b) & (a>0) & (b>0)
        if not np.any(valid):
            raise ValueError('A shared frame has no positive finite paired depth')
        difference=np.log(a[valid].astype(np.float64))-np.log(b[valid].astype(np.float64))
        frame_log_scales.append(float(difference.mean()))
        scatter.append(float(difference.std()))
        counts.append(int(valid.sum()))
    log_scale=float(np.mean(frame_log_scales))
    scale=float(np.exp(log_scale))
    translation=np.mean(old[:,:3,3]-scale*new[:,:3,3]@rotation.T,axis=0)
    transform=np.eye(4)
    transform[:3,:3]=scale*rotation
    transform[:3,3]=translation
    transform=immutable_similarity(transform)
    return transform,dict(scale=scale,log_scale=log_scale,
        frame_log_scales=frame_log_scales,pixel_log_scale_std=scatter,
        frame_log_scale_std=float(np.std(frame_log_scales)),valid_pixels=counts,
        rotation_singular_values=singular.tolist())
