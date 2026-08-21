# Method Draft: Certified Episodic Compass

**Status:** paper prose draft for the frozen current method. Public-benchmark
validation of the actual Revisit-bearing claim remains open. The semantic-first
variant is excluded because consumed Gate B tied geometry-first exactly and
failed the frozen strict-net-gain promotion rule.

## 1. Problem formulation

We consider continual ImageGoal navigation with a frozen policy
`pi_theta`. At decision time `t`, the agent receives its current monocular RGB
observation `I_t`, a goal image `G`, and the causal visual history

```text
H_t = {I_0, ..., I_(t-1)}.
```

The history may contain a prior observation of the goal place (Revisit), or it
may contain no usable support for the goal (Novel/unsupported). The runtime is
not told which case holds. Our objective is not to replace `pi_theta` with a
mapping or planning system. It is to decide whether a hypothesis retrieved
from `H_t` has enough evidence to receive action authority, and, if so, expose
the smallest controller input that transfers its useful information.

This leads to an open-set authorization problem:

```text
episodic hypothesis -> geometric witness -> authorized direction or abstain.
```

An abstention must reproduce the native frozen-policy decision rather than a
separately trained fallback.

## 2. Causal episodic addressing

For goal `G`, a frozen DINO encoder retrieves a temporally diverse shortlist
of at most `K=8` historical observations:

```text
Q_t(G) = (h_1, ..., h_K),  h_k < t.
```

Candidates are separated by at least four history frames and exclude the
initial unsupported prefix. Retrieval proposes where to look; its similarity
score is not treated as permission to control the robot.

For the frozen current CEC instantiation, SuperPoint/LightGlue correspondences
are computed between each candidate `I_h` and `G`. A deterministic
Fundamental-MAGSAC test measures inlier count and spatial support. Candidates
are ranked lexicographically by:

```text
(epipolar inliers,
 query grid coverage,
 query hull coverage,
 median match score,
 DINO cosine,
 earlier anchor).
```

This ranking is an implementation of hypothesis proposal, not the paper's
novel claim. The proposal/verification separation is tested explicitly in the
ablation plan because local geometric support need not equal semantic or
downstream control utility.

## 3. A geometric witness, not a metric waypoint

For the selected historical hypothesis `h`, the monocular LingBot stream
provides a reference depth image `D_h` and a camera-to-world pose
`T_h=(R_h,t_h)` in its internal per-stream scale. LightGlue keypoints in
`I_h` are lifted through `D_h` to 3-D. PnP-RANSAC then estimates the goal-camera
pose `T_G=(R_G,t_G)` from those 3-D points and their matched pixels in `G`.

CEC accepts the hypothesis only if one atomic certificate passes:

```text
PnP status is valid,
number of PnP inliers >= 16,
query inlier hull coverage >= 0.05,
reference inlier hull coverage >= 0.05,
reprojection RMSE <= 2.0 pixels.
```

The certificate uses no simulator pose, co-visibility, Novel/Revisit label, or
future frame. A rejection means only that the current causal history does not
support this hypothesis under the frozen witness; it does not semantically
prove that the goal is Novel.

Monocular scale is deliberately not exposed to navigation. Let the current
LingBot pose be `T_t=(R_t,t_t)`. The estimated goal translation in the current
camera frame is

```text
u = R_t^T (t_G - t_t).
```

Using the audited camera-to-NavDP axis convention, CEC forms

```text
v = [u_z, -u_x],          b = v / ||v||_2.
```

`b` is a scale-free `[forward,left]` bearing. PnP's arbitrary monocular norm is
never called metres and never reaches the controller.

## 4. Low-bandwidth residual control with exact fallback

When the certificate accepts, CEC projects the unit bearing to one radius
frozen before evaluation,

```text
p = rho b,     rho = 2.5 m,
```

and calls frozen NavDP through its existing mixed ImageGoal+PointGoal
interface. Thus episodic memory contributes one direction, while the goal
image and frozen diffusion policy retain local control, collision avoidance,
and replanning.

When the certificate rejects or any dependency is malformed, CEC calls the
unchanged native ImageGoal endpoint. With paired diffusion seeds and the same
NavDP observation FIFO, the unsupported branch is:

```text
Tau_t = pi_theta(I_t, G)                     if certificate rejects,
Tau_t = pi_theta(I_t, G, rho b)              if certificate accepts.
```

The memory sidecar cannot select a controller, change policy weights, expose a
metric waypoint, or infer a role label. This is the key abstraction: **the
certificate is an authorization boundary, and bearing is the only information
allowed to cross it.**

## 5. Algorithm

```text
Input: goal image G, current image I_t, causal history H_t,
       frozen ImageGoal policy pi_theta

1. Retrieve a temporally diverse DINO shortlist Q_t(G) from H_t.
2. Compute local correspondences and frozen epipolar evidence.
3. Select the frozen CEC proposal.
4. Lift its reference matches using causal monocular depth.
5. Estimate the goal-camera pose with PnP-RANSAC.
6. Apply the atomic certificate.
7. If rejected or invalid:
       return pi_theta(I_t, G).              # exact native fallback
8. Convert relative translation to unit bearing b.
9. Return pi_theta(I_t, G, 2.5 b).           # frozen mixed controller
```

Candidate membership is frozen on the first query for a goal and cached within
that goal lifecycle. All failure paths are explicit abstentions.

## 6. Why this is a method rather than a module list

DINO retrieval, local matching, monocular depth, and PnP are replaceable
instances of three abstract roles:

1. **proposal** identifies a causal episodic hypothesis;
2. **witness** determines whether that hypothesis may act;
3. **interface** restricts authorized information to a scale-free direction.

The contribution is the composition rule and the continual open-set control
contract, not a claim that hierarchical localization is new. This
factorization yields testable properties that an unconstrained engineering
pipeline does not provide:

- causal memory provenance;
- role-free acceptance/abstention;
- measurable Revisit utility and Novel interference;
- scale isolation under monocular depth;
- exact native fallback;
- compatibility with an unchanged generative navigation policy.

## 7. Training-free design choice

CEC trains no task-specific router or controller. This is evidence-driven
rather than ideological. In the current project:

- candidate-level learned ranking did not improve actionable certificate
  coverage over geometry (`115` versus `122`);
- geometry-first learned rescue added only `1/349` actionable sessions;
- removing explicit retrieval reduced long-history GCT addressing from
  `18/20` to `5/20`, paired `+0/-13`, `p=0.000244`;
- a small scene-OOF residual changed DINO from `74/80` to `76/80`, `p=0.5`,
  with gains confined to two scenes.
- changing proposal order from geometry-first to DINO-order first-certified
  changed the first anchor in `21/28` consumed Revisit histories but produced
  exactly the same closed-loop outcome (`25/28`, paired `+0/-0`, `p=1.0`);
  the first authorized bearings were all within `4.478°`.

These results isolate long-range content addressing and open-set authorization
as the failure points of the learned alternatives, while showing that fine
ranking among already supported, co-visible Revisit anchors is not the current
closed-loop bottleneck. Explicit retrieval and a geometric witness are
therefore retained until a replacement improves the same closed-loop,
role-free contract rather than only an offline AUC.

## 8. Evaluation contract

Every role-free experiment replays the exact actual-online causal prefix and
removes the construction role from runtime inputs. Novel and Revisit are
reported separately before any aggregate:

- Revisit SR and successful takeover coverage measure utility;
- Novel false takeover, paired loss, and path equality measure interference;
- native, raw always-on, geometry, and certified arms share seeds and budgets;
- exact McNemar tests and scene-cluster bootstrap intervals accompany all
  closed-loop differences.

The current evidence supports large Revisit utility and empirical fail-closed
behavior on the evaluated held-out Novel populations. It does not establish a
zero-error certificate, superior Revisit SR over every raw baseline, or a
deployable Novel direction source.

## 9. Current limitations

1. The certificate is operational and empirical, not a formal safety proof;
   train40 still contains false accepts.
2. High-support Revisit is easier than long-baseline, low-overlap
   relocalization; the paper must stratify support and temporal gap.
3. Local geometric consistency can select a semantically inferior repeated
   structure; the frozen proposal/verification audit measures this directly.
4. PnP and monocular depth add planning latency; runtime and memory scaling
   must be reported.
5. CEC intentionally falls back on unsupported Novel targets. Oracle-bearing
   recoverability is a mechanism result, not a deployable Novel method.
6. The GOAT first-ImageGoal semantic-arrival adapter failed its frozen
   confirmation (`0/20` certified successes). It did not exercise episodic
   Revisit bearing; a compatible sequential public-benchmark evaluation
   remains future work.
