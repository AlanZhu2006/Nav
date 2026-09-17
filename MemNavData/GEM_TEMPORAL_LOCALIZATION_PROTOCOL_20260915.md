# Fixed generalized-camera measurement experiment

Freeze before new matching/real-data solving. Source: completed loop observer
attempt_002. Preserve all sealed source/results, production GEM, target SP+LG,
native LingBot geometry, manuscript and navigation. No model replay or push.

Select exactly the 22 queries accepted by the previous DINO+MASt3R first-pass
inference, using no evaluator or GT. Reference identity remains fixed in all
arms; the other 14/36 queries remain outside this conditional precision test.
Add actual observations t-14 and t-7 to current t: three views on the native
7-frame cadence. Reference h is the previous causal choice h<=t-64. All three
observations exist at t; no additional views/rendering/actions or retrieval.

Reuse current correspondences/encoder artifacts. Decode only 44 new historical
reference/support pairs with official MASt3R-SLAM matching and unchanged
nonconsecutive loop gate Q>1.5, both fractions>=0.1. Verify direct upstream gate
agreement using the same decoded tensors. F filtering at 1.5 pixels on native
pad coordinates and depth lifting follow original PnP mechanics. Positive finite
native reference depth with original confidence quantile=0. Extra frames need
no individual PnP/certificate vote. Pair rejection or no valid depth means no
observations from that frame, explicitly reported.

At each fixed current/reference query, estimate one transform from reference
chart h to the recent camera rig. Native L_j=T_t^-1 T_j supplies camera axes
R_j and centers c_j; history supplies first-write 3D points X_h and each j its
first-write predicted K_j. Optimize pixel residuals

    pi(K_j R_j^T (R X_h + b - s c_j)) - u_j.

Output is Z_(h<-t)=[s R^T, -R^T b]. The scale is positive, represented by log s.
This is the generalized pose-and-scale measurement model, not a new SLAM solver
or a reproduction of the global polynomial gDLS algorithm. Native relative pose
initializes R,b and s=1; no pose prior term, covariance claim or alternate start.

Three predeclared arms: current-only rigid (6 DoF), shared three-view rigid
(6 DoF), shared three-view similarity (7 DoF, primary). Same overall maximum
2,048 correspondences in every arm: distribute evenly among nonempty views,
redistribute unfilled capacity, deterministic uniform pixel-order subsampling.
Thus multiple views do not triple the point budget. No feature-confidence
weighting or pairwise PnP averaging. Same SciPy local least-squares mechanics:
soft_l1, 3-pixel fixed residual scale inherited from PnP reprojection setting,
3-point Jacobian, trf, x_scale=jac, ftol/xtol/gtol=1e-8, max_nfev=100. Native
units normalized by median history point norm for conditioning. Broad log-scale
bounds [-20,20] prevent overflow, not an accuracy prior. No parameter sweep.

Save every finite output, including nonconvergence, poor cheirality and numerical
rank deficiency. Primary reporting scores all finite estimates on the identical
18 geometrically scorable query/reference pairs (true distance>=0.5m); separately
report full-rank/converged outputs without masking failures in the main table.
No quality-based replacement/fallback or online coordinate publication. Report
native and previous MASt3R/PnP, all three arms, per-case errors, >15/90deg and
paired improvements/worsening>5deg. Distinguish scale information and geometry
conditioning from calibrated uncertainty. GT only in post-inference evaluator.

Synthetic checks before real data: known similarity recovery, unit-gauge
equivariance, pure-rotation scale non-observability, fixed total point budget,
causal and duplicate identities. Final source/input/hash/raster integrity audit
reuses saved artifacts. Report incremental inference, archive and CPU solve
costs, not full online latency. These are consumed development traces.

References: [gDLS generalized pose-and-scale model](https://sites.cs.ucsb.edu/~mturk/pubs/gdls_eccv_14.pdf),
[SciPy least_squares](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html),
[MASt3R-SLAM official verification](https://github.com/rmurai0610/MASt3R-SLAM/blob/main/mast3r_slam/global_opt.py).
