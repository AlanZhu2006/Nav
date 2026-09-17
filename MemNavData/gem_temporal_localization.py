"""Experimental generalized-camera measurement in one historical chart.

Each actual recent view has a fixed native LingBot pose in the current camera
chart. One shared rotation, translation and optional scale explains all its
2D--3D observations. No per-frame pose averaging, runtime writes or fallback.
"""
from dataclasses import dataclass
import time

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


@dataclass(frozen=True)
class TemporalView:
    frame: int
    reference_points: np.ndarray
    pixels: np.ndarray
    intrinsic: np.ndarray
    current_from_camera: np.ndarray

    def validate(self, current):
        x,u,k,l = self.reference_points,self.pixels,self.intrinsic,self.current_from_camera
        if self.frame > current or self.frame < 0:
            raise ValueError('Only already observed views are valid')
        if x.ndim!=2 or x.shape[1]!=3 or u.shape!=(len(x),2):
            raise ValueError('Aligned 3D points and 2D observations required')
        if k.shape!=(3,3) or l.shape!=(4,4) or min(k[0,0],k[1,1])<=0:
            raise ValueError('Valid intrinsics and local camera pose required')
        if not all(np.isfinite(a).all() for a in (x,u,k,l)):
            raise ValueError('Nonfinite input evidence')
        if not np.allclose(l[3],[0,0,0,1]) or not np.allclose(l[:3,:3].T@l[:3,:3],np.eye(3),atol=1e-7):
            raise ValueError('Recent camera poses must be rigid')
        if np.linalg.det(l[:3,:3])<=0:
            raise ValueError('Improper camera rotation')


def balanced_samples(counts, budget=2048):
    """Equal per-view allocation, redistributing unused capacity; no scores."""
    counts = np.asarray(counts,dtype=np.int64)
    if budget<1 or np.any(counts<0):
        raise ValueError('Positive budget and nonnegative lengths required')
    allocation = np.zeros(len(counts),dtype=np.int64)
    remaining = min(int(counts.sum()),budget)
    while remaining:
        for i in range(len(counts)):
            if allocation[i]<counts[i] and remaining:
                allocation[i]+=1
                remaining-=1
    return [np.linspace(0,n-1,k,dtype=np.int64) if k else np.empty(0,dtype=np.int64)
            for n,k in zip(counts,allocation)]


def shared_transform(views, *, current, reference_from_current, estimate_scale,
                     point_budget=2048, robust_px=3., max_nfev=100):
    """Fit X_h = s R^T X_current - R^T b from original pixel evidence.

    Projection in view j is pi(K_j R_j^T (R X_h + b - s c_j)). R_j,c_j
    come from native motion, not the pair matcher. Scale is unobservable when
    camera origins coincide. Return every finite estimate and numerical rank;
    rank is diagnostic, never a ground-truth-dependent acceptance threshold.
    """
    start = time.perf_counter()
    ordered = sorted(views,key=lambda v:v.frame)
    if len({v.frame for v in ordered})!=len(ordered):
        raise ValueError('Repeated view identities')
    for view in ordered:
        view.validate(current)
    initial = np.asarray(reference_from_current,dtype=float)
    if initial.shape!=(4,4) or not np.isfinite(initial).all():
        raise ValueError('Finite native initial transform required')
    if not np.allclose(initial[:3,:3].T@initial[:3,:3],np.eye(3),atol=1e-7):
        raise ValueError('Initialization is a rigid native pose')
    if robust_px<=0:
        raise ValueError('Positive fixed residual scale required')
    indices = balanced_samples([len(v.pixels) for v in ordered],point_budget)
    selected = [(v,i) for v,i in zip(ordered,indices) if len(i)]
    if sum(len(i) for _,i in selected)<8:
        return dict(status='insufficient_evidence',transform=None,identified=False,
                    used_points=sum(len(i) for _,i in selected),solve_ms=1000*(time.perf_counter()-start))
    X = np.concatenate([v.reference_points[i] for v,i in selected])
    pixels = np.concatenate([v.pixels[i] for v,i in selected])
    axes = np.concatenate([np.repeat(v.current_from_camera[None,:3,:3],len(i),axis=0) for v,i in selected])
    centers = np.concatenate([np.repeat(v.current_from_camera[None,:3,3],len(i),axis=0) for v,i in selected])
    cameras = np.concatenate([np.repeat(v.intrinsic[None],len(i),axis=0) for v,i in selected])
    # Normalize arbitrary native units for numerical conditioning only.
    unit = float(np.median(np.linalg.norm(X,axis=1)))
    if not unit>1e-12:
        raise ValueError('Degenerate reference geometry')
    X,centers = X/unit,centers/unit
    R0 = initial[:3,:3].T
    b0 = -R0@initial[:3,3]/unit

    def unpack(x):
        return Rotation.from_rotvec(x[:3]).as_matrix()@R0, b0+x[3:6], np.exp(x[6]) if estimate_scale else 1.

    def projection(x):
        R,b,s = unpack(x)
        delta = X@R.T+b-s*centers
        y = np.einsum('nji,nj->ni',axes,delta)
        # A signed epsilon only prevents undefined division; negative-depth
        # observations remain visible in diagnostics and are never dropped.
        z = y[:,2]
        safe_z = np.where(np.abs(z)>1e-9,z,np.where(z>=0,1e-9,-1e-9))
        homogeneous = np.einsum('nij,nj->ni',cameras,y)
        return homogeneous[:,:2]/safe_z[:,None],z

    def residual(x):
        uv,_ = projection(x)
        return (uv-pixels).reshape(-1)

    dimension = 7 if estimate_scale else 6
    lower,upper = np.full(dimension,-np.inf),np.full(dimension,np.inf)
    if estimate_scale:
        lower[-1],upper[-1] = -20.,20.  # broad numerical bounds, no scale prior
    result = least_squares(residual,np.zeros(dimension),jac='3-point',method='trf',
        loss='soft_l1',f_scale=robust_px,max_nfev=max_nfev,
        ftol=1e-8,xtol=1e-8,gtol=1e-8,bounds=(lower,upper),x_scale='jac')
    R,b,s = unpack(result.x)
    transform = np.eye(4)
    transform[:3,:3] = s*R.T
    transform[:3,3] = -R.T@b*unit
    uv,z = projection(result.x)
    residual_norm = np.linalg.norm(uv-pixels,axis=1)
    singular = np.linalg.svd(result.jac,compute_uv=False)
    rank = int(np.linalg.matrix_rank(result.jac))
    # Schur complement: scale information remaining after pose can change.
    scale_information = None
    if estimate_scale:
        pose_jac,scale_jac = result.jac[:,:6],result.jac[:,6]
        orthogonal = scale_jac-pose_jac@np.linalg.lstsq(pose_jac,scale_jac,rcond=None)[0]
        scale_information = float(orthogonal@orthogonal)
    return dict(status='converged' if result.success else 'iteration_limit',
        transform=transform.tolist(),scale=float(s),identified=rank==dimension,
        optimizer_success=bool(result.success),optimizer_message=result.message,
        nfev=int(result.nfev),numerical_rank=rank,dimension=dimension,
        jacobian_singular_values=singular.tolist(),scale_information=scale_information,
        scale_bound_active=bool(estimate_scale and result.active_mask[-1]!=0),
        used_points=len(X),used_views=[v.frame for v,_ in selected],
        points_per_view=[len(i) for _,i in selected],
        input_indices=[i.tolist() for _,i in selected],normalization_unit=unit,
        reprojection_median_px=float(np.median(residual_norm)),
        reprojection_rmse_px=float(np.sqrt(np.mean(residual_norm**2))),
        positive_depth_fraction=float(np.mean(z>0)),
        initial_robust_cost=float(robust_px**2*np.sum(np.sqrt(1+(residual(np.zeros(dimension))/robust_px)**2)-1)),
        robust_final_cost=float(result.cost),solve_ms=1000*(time.perf_counter()-start))
