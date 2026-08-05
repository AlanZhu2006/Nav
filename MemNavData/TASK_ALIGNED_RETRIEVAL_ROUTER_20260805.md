# Task-Aligned Retrieval Router Audit (2026-08-05)

## Status

This work is an offline retrieval diagnostic. It does not modify or deploy the
live MemNav/NavDP policy. The resulting linear head remains explicitly marked
`deployment_approved: false` until it passes the frozen final-scene split and a
closed-loop Novel/Revisit evaluation.

All implementation changes described here live in the `Nav-axis-uturn` child
repository. The parent checkout at `/home/asus/Research/Nav` is only used as a
read-only source for the pinned LingBot repository and weights.

## Why the previous teacher was wrong for this task

The previous router used SIFT matches plus an essential-matrix RANSAC test as
its binary teacher. That answers a generic place-recognition question:

> Can some textured pixels in these two images support a two-view geometric
> model?

The navigation target is directional and more specific:

> What fraction of the query/goal surface is visible, in-frustum, and
> depth-consistent in this memory frame?

Those questions are not equivalent. SIFT can accept a candidate because a
shared wall, grass patch, shutter, or other background is geometrically
consistent even when the goal surface is absent. It can also reject a true
low-texture revisit because too few keypoints survive the fixed match-count
threshold. This is the standard retrieval-versus-verification and perceptual
aliasing failure described in the visual place-recognition literature.

The generator already defines the task label through an occlusion-aware 3D
co-visibility curve in `generate_twoleg.py`:

- positive: co-visibility `>= 0.50`;
- negative: co-visibility `<= 0.10`;
- ambiguous: `(0.10, 0.50)`, excluded from binary supervision.

For stored goal images, the relabeler reads the generator's exact
`covis_curve`. For numeric cross-episode and return queries, it reconstructs
the same directional label from the query depth, camera-to-world pose,
candidate depth, and candidate pose. Geometry is used only while constructing
offline labels; the learned router remains RGB-only at inference time.

## Evidence

The local medium audit contains four training scenes and five completely held
out scenes. DINO first produces top-32 candidates.

### Complete relabel

- 364 retrieval sessions;
- 11,648 selected top-32 pairs;
- 2,798 positives, 5,915 negatives, and 2,935 ignored pairs;
- 139 sessions contain at least one positive inside top-32;
- 109 sessions already have a positive at DINO rank 1;
- relabel time: 10.14 seconds, or 0.87 ms per selected pair after cached I/O.

Across all supervised extremes, the old SIFT teacher has precision 0.940 but
recall only 0.659 against task co-visibility: 954 false negatives and 118 false
positives. On the harder non-goal heldout subset, precision is 0.863 and recall
is 0.530. Relaxing the SIFT match threshold increases recall but also creates
many false positives; it does not repair the target mismatch.

### Teacher-parameter sensitivity

The 11,648 pairs were relabeled under three perturbations:

| Setting | Positive | Negative | Ignored | Mean absolute score change |
| --- | ---: | ---: | ---: | ---: |
| baseline: stride 6, tolerance 0.3 m | 2,798 | 5,915 | 2,935 | 0 |
| tolerance 0.2 m | 2,733 | 5,955 | 2,960 | 0.0048 |
| tolerance 0.5 m | 2,877 | 5,824 | 2,947 | 0.0067 |
| stride 3 | 2,801 | 5,911 | 2,936 | 0.0015 |

For pairs that remain in the supervised positive/negative extremes, agreement
with the baseline is 100% for all three perturbations. Borderline pairs move
into or out of the ignore band, but explicit positives do not flip to explicit
negatives. The result is therefore not an artifact of one depth tolerance or
sampling stride.

### Refit with corrected labels

On the five heldout scenes, 27 of 85 sessions contain a positive within the
top-32 candidate set:

| Ranker | Positive selected at rank 1 | Conditional recall@1 | Mean first-positive rank | Pair AP |
| --- | ---: | ---: | ---: | ---: |
| frozen DINO cosine | 20/27 | 0.741 | 3.41 | 0.803 |
| symmetric patch | 19/27 | 0.704 | 2.74 | 0.937 |
| symmetric patch + temporal | 18/27 | 0.667 | 2.67 | 0.934 |

Correct labels remove the severe degradation seen when the same models were
trained on SIFT labels, but a pointwise logistic head still does not beat DINO
at rank 1. High pair AP and improved mean rank do not automatically imply
better top-1 ranking. The remaining likely limitations are:

1. the old feature summary intentionally erased query-versus-memory direction;
2. the objective is pointwise binary classification, not within-session
   ranking;
3. 58 of 85 heldout sessions have no positive in top-32, so no reranker can
   recover them without improving candidate recall;
4. temporal support can reinforce a visually aliased region instead of the
   goal surface.

## Implemented changes

- `covisibility_teacher.py`: exact goal-surface backprojection, reprojection,
  depth-consistency checks, thresholding, path mapping, and bounded caches.
- `relabel_router_covisibility.py`: deterministic top-K relabeling with source
  provenance, original SIFT labels retained for comparison, atomic CSV output,
  and per-kind/per-scene reports.
- `patch_temporal_router.py`: directional query/memory patch summaries plus an
  exact conversion back to the old symmetric representation.
- `diag_patch_temporal_router.py`: ignored-label support, task co-visibility
  ranking metrics, directional and exact symmetric controls from one DINO
  forward pass, and cache identity that includes the complete temporal score
  curve but excludes labels.
- `run_patch_temporal_router_multiscene.sh`: four-stage candidate, relabel,
  directional/symmetric, and scene-disjoint audit pipeline.
- `run_patch_temporal_router_long.sh`: the older 4-train/5-heldout runner now
  uses the same corrected teacher and directional control.
- `test_covisibility_teacher.py` and `test_patch_temporal_router.py`: geometry,
  occlusion, ignore-band, directionality, exact-control, and cache-identity
  tests.

The balanced CPU end-to-end smoke completed feature extraction, directional
and symmetric fitting, scene OOF, report writing, and non-deployment export.
Its 18 pairs are only a logic test and must not be reported as model quality.

## Dependency and provenance preflight

The formal runners fail closed unless all of the following pass before GPU
work starts:

- pinned repository commit, split SHA, LingBot commit, and weight SHA;
- read-only dataset overlay size and header SHA;
- clean task-file diff against the submitted commit;
- Python compilation and the router unit-test suite;
- imports for OpenCV, NumPy, Pandas, PyArrow, scikit-learn, PyTorch, and Pillow;
- CUDA availability and GPU inventory;
- disjoint train/development/final-reserved scene roles;
- complete candidate identities and finite co-visibility labels in selected
  top-K rows.

The local `base` Python lacks OpenCV, but the pinned `memnav` interpreter has
all required dependencies. Formal jobs always use the pinned interpreter.

## Full experiment and decision rule

The next full H200 run uses 40 training scenes, 10 development scenes, and
keeps four final scenes untouched. It extracts directional patch features once
and derives the exact symmetric control algebraically, ensuring identical
images, candidates, labels, and DINO forwards.

The directional model should not be promoted merely for higher pair AP. It
must improve task-aligned conditional recall@1 and selected co-visibility on
the development split without unsafe confidence-tail errors. If it does not,
the next model change should be a session-level pairwise/listwise ranking loss
with hard alias negatives and an explicit reject option. Candidate generation
must also be improved separately for sessions with no positive in top-32.

## Related primary literature

- NetVLAD introduced learned global descriptors for place recognition, but a
  global descriptor alone does not verify local overlap: Arandjelovic et al.,
  CVPR 2016, <https://arxiv.org/abs/1511.07247>.
- SeqSLAM uses temporal consistency to survive appearance change, while still
  requiring a sequence-level match hypothesis: Milford and Wyeth, ICRA 2012,
  <https://doi.org/10.1109/ICRA.2012.6224623>.
- Patch-NetVLAD reranks global candidates with local patch descriptors:
  Hausler et al., CVPR 2021, <https://arxiv.org/abs/2103.01486>.
- LoFTR and LightGlue show why learned local correspondence is a stronger
  verification primitive than fixed sparse keypoints in weak-texture cases:
  <https://arxiv.org/abs/2104.00680> and
  <https://arxiv.org/abs/2306.13643>.
- NG-RANSAC and DSAC demonstrate learned or differentiable hypothesis scoring,
  but geometric verification still needs a task-aligned target to avoid
  verifying irrelevant background: <https://arxiv.org/abs/1711.10228> and
  <https://arxiv.org/abs/1611.05705>.
