# Patch-temporal reliability router: training and audit

Date: 2026-08-05

Repository scope: this work lives only in
`/home/asus/Research/Nav-axis-uturn`.  It does not edit the parent
`/home/asus/Research/Nav` checkout.

## Objective

The current geometric router is reliable but must run SIFT matching and an
essential-matrix check for uncertain image goals.  The old learned gate uses a
single global DINO CLS descriptor and does not reliably distinguish a true
revisit from a visually similar novel goal.  This experiment asks whether a
small learned head can safely avoid some geometry calls by using:

1. symmetric local-patch correspondence from the exact frozen LingBot DINO
   trunk; and
2. temporal support around the retrieved memory frame.

This is reliability routing, not a replacement for retrieval and not a new
navigation policy.  The exported model remains marked
`deployment_approved=false` until it passes closed-loop navigation A/B tests.

## Labels and leakage controls

The teacher label is the unchanged SIFT/essential-matrix verifier.  The router
does **not** receive Habitat pose, episode success, action labels, goal phase,
or a GT gate.

The fixed split contains four training scenes and five held-out scenes:

- train: `17DRP5sb8fy`, `1LXtFkjw3qL`, `1pXnuDYAj8r`, `Uxmj2M2itWa`;
- held out: `e9zR4mvMWw7`, `rqfALeAoiTq`, `s8pcmisQ38h`, `yqstnuAEVhm`,
  `zsNo4HB9uLZ`.

Training data is expanded in two complementary ways:

- `cross_episode_train`: a frame from an independent episode queries another
  trajectory in the same training scene; these are mostly hard negatives but
  may contain genuine geometric overlap;
- `within_episode_return_train`: a post-switch return-leg frame queries only
  the pre-switch memory of the same episode; this supplies natural positive
  and negative revisit states.

Only the four training scenes are expanded.  The runner compares the base and
expanded CSV row-for-row and aborts if any held-out row changes, any new
held-out row appears, or a duplicate session/candidate pair is introduced.
Hyperparameter `C` is selected with leave-one-training-scene-out OOF
predictions.  Selective accept/reject thresholds are calibrated from training
OOF predictions only; held-out scenes are used once for reporting.

## Fixed dependencies

- base teacher SHA256:
  `7aa916080eeec15ad505ca6b8c2349ac2383a9846ee1bb20ed704c3df350c779`;
- exact DINO CLS cache SHA256:
  `5d920cf32756c26a45a3c854f1e18103cb6980cbba23684815584130db7a8d7b`;
- LingBot weight SHA256:
  `832bc82cbae0bc9bbe946ef5ee1f7226abd8c0e183ccf8beddbb3d133576f409`;
- LingBot source commit:
  `7ff6f3ed0913d4d326f8f13bbb429c4ffc0195c2`.

Every Slurm run verifies these identities, the Nav-axis-uturn commit, CUDA,
Python imports, script compilation, and all router unit tests before it starts
data generation.  Missing or mismatched dependencies fail the job rather than
silently changing the experiment.

## Local evidence before the long run

The original exact top-32 diagnostic had 64 training and 20 held-out top-1
sessions.  On held-out top-1 decisions:

| Feature family | ROC AUC | AP | zero-error coverage |
| --- | ---: | ---: | ---: |
| global cosine | 0.9011 | 0.9568 | 0% |
| patch | 0.9341 | 0.9757 | not safe: one false reject |
| patch + temporal | 0.9341 | 0.9654 | 5% |

A one-scene balance smoke generated 4,688 new pairs: 348 positive and 4,340
negative.  Cross-episode pairs were 2.7% positive; return-leg pairs were 14.2%
positive.  Therefore the expansion is not an all-negative shortcut.

The complete four-scene runner smoke then added 64 sessions and 4,412 pairs,
doubling training top-1 sessions from 64 to 128.  Its top-4/grid-4
patch-temporal result was held-out AUC 0.9341, AP 0.9757, and 10% selective
coverage with zero false accepts and zero false rejects.  This validates the
pipeline but is not a direct numerical improvement claim over top-32/grid-8.

## Eight-hour full configuration

The full task uses query stride 32, candidate stride 1, top-K 32, an 8x8 patch
grid, and all four training scenes.  From episode metadata it is expected to
add 521 training sessions and 162,271 teacher pairs.  Together with the base
data, the learned head will see about 585 training top-1 sessions while the 20
held-out top-1 sessions remain unchanged.

Entry points:

- `MemNavData/run_patch_temporal_router_long.sh`
- `MemNavData/slurm_patch_temporal_router_long.sbatch`

Expected HPC result directory:

`/scratch/yz11502/Research/Nav-axis-uturn-results/patch_temporal_router_20260805/job_<JOB_ID>`

The Slurm limit is eight hours; the task may finish earlier.  The primary
acceptance conditions are no false automatic route decisions on held-out
sessions, higher safe coverage than the 5% exact baseline, and no regression
in scene-disjoint top-1 AUC/AP.  Because the held-out set is small, passing
this audit only authorizes a larger offline/closed-loop evaluation, not live
deployment.

## Submission record

Code commit, Slurm job ID, start state, and final metrics are filled in after
the committed child checkout is synchronized to HPC.
