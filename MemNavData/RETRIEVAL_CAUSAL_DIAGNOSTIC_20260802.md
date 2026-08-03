# MemNav Retrieval-to-Action Causal Diagnostic (2026-08-02)

## Question

The archived fixed-seed 30-episode run reached B in 13/30 episodes. Several
failed episodes also had a raw-DINO retrieval anchor far from the metadata's GT
covisibility argmax. This diagnostic asks two separate questions:

1. Is raw retrieval wrong on those episodes?
2. If the anchor is made correct, does the action decoder actually change its
   waypoint in a useful way?

The second question matters because retrieval quality is not navigation quality
unless the revisit representation causally controls the diffusion action.

## Controlled interventions

All tests use `checkpoint-1500`, raw-DINO retrieval, replayed leg A, the same
episode seed, and the axis-fixed evaluator. The server exposes two optional,
evaluation-only interventions; both default to off:

- `forced_anchor`: replace only the history anchor used by LingBot goal append
  and `build_revisit`. The trained retrieval logits, predicted gate, current
  branch, novel branch, diffusion noise, and controller are unchanged.
- `forced_gate`: replace only the decoder's revisit/novel soft gate. `gate=1`
  is an oracle capacity ablation, not a deployable setting.

`diag_oracle_retrieval_firsthop.py` resets and rebuilds streaming memory for
every arm, preventing state leakage. It compares raw and oracle diffusion
trajectories generated from the same first leg-B view. The absolute direction
to the recorded path is diagnostic only: a successful policy may initially
take another collision-free route. The most reliable measurement is the paired
angular change between trajectories when only the anchor or gate changes.

## Severe-anchor-error failures

Three archived navigation failures were selected because raw retrieval was
wrong by 40-77 frames, far larger than the common 1-13-frame neighborhood
error.

| Episode | Predicted gate | Raw -> GT anchor | Anchor correction | Raw vs oracle first-hop change |
|---|---:|---:|---:|---:|
| `17DRP5sb8fy/episode_0004` | 0.238 | 119 -> 42 | 77 frames | 0.23 deg |
| `1LXtFkjw3qL/episode_0003` | 0.193 | 107 -> 65 | 42 frames | 0.65 deg |
| `1LXtFkjw3qL/episode_0007` | 0.377 | 64 -> 104 | 40 frames | 0.66 deg |

Thus retrieval is demonstrably wrong on this subset, but replacing it with the
GT covisibility anchor barely changes the first diffusion waypoint.

One known successful control, `Uxmj2M2itWa/episode_0006`, had a higher predicted
gate (0.576). Changing anchor 48 -> 40 changed its first-hop direction by 3.87
degrees. Its absolute first-hop error against the recorded trajectory was still
large, confirming that this absolute quantity is not itself a success metric.
The paired sensitivity nevertheless supports the expected trend: the decoder
reacts more to revisit changes when the gate is higher.

## Full closed-loop and gate=1 ablation

The most severe case was then rolled out end to end:

| Arm | Anchor | Decoder gate | Reached B | Steps | Travel |
|---|---:|---:|---:|---:|---:|
| Raw | 119 | 0.238 | no | 244 | 9.101 m |
| GT anchor | 42 | 0.238 | no | 244 | 9.120 m |
| GT anchor + gate oracle | 42 | 1.000 | no | 242 | 8.992 m |

For the first plan, GT anchor alone changed direction by 0.23 degrees. GT anchor
plus gate=1 increased the change to 3.42 degrees and moved the diffusion
endpoint substantially, showing that the revisit tokens are connected to the
decoder. It did not make the plan more consistent with the recorded route, and
repeated closed-loop replanning still failed.

The checkpoint explains why anchor-only intervention is weak. Decoder
cross-attention adds `log(gate)` to revisit columns and `log(1-gate)` to novel
columns. Its learned global branch bias is only `[+0.0207, -0.0096]`, too small
to counter a predicted gate of 0.19-0.38. But gate underconfidence is not the
whole issue: forcing gate=1 changes the action without making it reliable.

## Conclusion

The causal conclusion is narrower and stronger than “retrieval is bad”:

- Raw retrieval has real large-error outliers.
- Correcting those outliers is not sufficient to correct the generated
  waypoint or rescue the demonstrated full rollout.
- Low gate suppresses sensitivity to a corrected anchor.
- Maximum revisit weight increases sensitivity, but the learned
  revisit-to-action mapping is not reliably directional.

Therefore another reranker or a lower retrieval loss alone is unlikely to be
the largest SR improvement. The current larger bottleneck is the weak causal
coupling from memory pose/revisit tokens into the action decoder.

## Most useful next training change

A focused training experiment should combine three changes rather than tuning
only retrieval:

1. **GT-gate teacher forcing with scheduled mixing.** On training revisit rows,
   condition the decoder with the GT revisit gate initially, while continuing
   to train the predicted gate separately. Gradually mix in the predicted gate.
   This prevents an immature gate from suppressing gradients into the revisit
   action path.
2. **Counterfactual anchor supervision.** Decode the same revisit sample with
   its positive anchor and a hard-negative anchor. The positive-anchor action
   must fit the GT trajectory; the hard-negative action must not be
   indistinguishable. This directly trains anchor sensitivity instead of
   assuming retrieval loss will create it.
3. **Revisit branch dropout / novel suppression.** On a controlled fraction of
   revisit rows, mask the novel tokens so the decoder cannot solve every sample
   through current-goal appearance alone. Keep novel-only rows as the safety
   counterpart.

The more structural alternative is to use the recovered memory-relative pose
to produce an explicit global subgoal (or topological/A* route) and reserve
diffusion for local collision-aware execution. That cleanly separates global
memory planning from local exploration, but it is a larger architecture change.

## Evidence files

- `eval_terminal_ab/oracle_retrieval/firsthop_17DR_ep4.json`
- `eval_terminal_ab/oracle_retrieval/firsthop_1L_ep3_ep7.json`
- `eval_terminal_ab/oracle_retrieval/firsthop_success_Uxm_ep6.json`
- `eval_terminal_ab/oracle_retrieval/firsthop_17DR_ep4_gate1.json`
- `eval_terminal_ab/oracle_retrieval/17DR_ep4_raw/`
- `eval_terminal_ab/oracle_retrieval/17DR_ep4_oracle/`
- `eval_terminal_ab/oracle_retrieval/17DR_ep4_oracle_gate1/`

This is a four-episode causal diagnostic, not an aggregate SR estimate. A new
training claim must be evaluated on the full fixed episode list.

## Training-side follow-up implemented on 2026-08-02

Code audit found a precise train/inference coupling failure:

- During training, `encode_memory` already replaces the selected anchor with the
  highest-scoring GT-positive frame.
- The action decoder nevertheless uses the predicted revisit gate from its very
  first update.
- Its attention mask adds `log(gate)` to every revisit column. Therefore an
  immature low gate hides the correctly anchored revisit token from action loss;
  BCE can improve the gate classifier without ever making the revisit branch a
  useful action expert.

The first low-risk intervention is now implemented:

1. decoder gate teacher forcing starts at 1.0 and linearly decays to 0.0;
2. gate BCE always trains the predicted gate independently;
3. after the configured handoff step, training and inference use exactly the
   same predicted-gate path;
4. W&B separately reports predicted `gate_seen/unseen/sep` and actual
   `decoder_gate_seen/unseen/sep`, plus `gate_teacher_ratio`;
5. a frozen LingBot override keeps cached and live features in eval mode even
   when the outer policy is put in train mode.

The old `checkpoint-2600` was also tested on five deterministic revisit samples
from the local validation set. For each sample the anchor was a GT-positive and
40 identical noise/timestep trials were decoded with the predicted gate, a 50%
mix, and oracle gate=1:

| Metric | Result |
|---|---:|
| Mean predicted gate | 0.3718 |
| Predicted-anchor positive accuracy | 0.60 |
| Action loss, predicted gate | 0.08497 |
| Action loss, 50% GT mix | 0.08440 |
| Action loss, oracle gate=1 | 0.08524 |
| Trials where oracle gate was better | 46.0% |
| Prediction RMS change, oracle vs predicted gate | 0.1136 |

Thus the old decoder is sensitive to gate changes, but the oracle route is not
systematically better. This rules out a pure inference-threshold fix and is the
reason to retrain the revisit-to-action coupling. It does **not** yet prove the
new curriculum improves SR; that is the purpose of the new controlled run.

A real one-sample forward/backward smoke using the same checkpoint passed:
predicted gate `0.418686`, 50% mixed decoder gate `0.709343`, nonzero
action-to-gate gradient `1.56e-4`, finite gradients in retrieval, revisit,
novel, and decoder parameters, with 11.31 GiB peak allocated GPU memory.

Counterfactual anchor loss and forced novel suppression remain proposed
follow-ups, not active changes in this run. Introducing all three simultaneously
would make the causal result uninterpretable; they should be added only if the
gate curriculum improves branch use but anchor sensitivity remains weak.

## Aggregate fixed-checkpoint result (2026-08-03)

The curriculum run was evaluated after its teacher ratio had reached zero.  The
candidate was `checkpoint-600` from
`memnav_gatecurr_warm2600_alllegs_warmfix_20260803_014435`; the baseline was the
shared `memnav_mp3d_flowgate/checkpoint-2600`.  A new paired evaluator fixed all
of the following between checkpoints:

- 72 episode-unique samples, with 12 each from novel Goal A, novel covis, and
  shallow/deep leg-2/leg-3 revisit groups;
- current frame, images, labels, and live retrieval candidates;
- eight diffusion noise/timestep trials per sample;
- eval-mode inference, with neither positive-anchor teacher forcing nor decoder
  gate teacher forcing.

The preflight (`15193930`) and full job (`15193931`) both completed with exit
code zero.  The full evaluator used commit
`3543c1d6b39afa6662cc8be9c620ad8495128d22` and completed in 19m40s.

| Fixed metric | flowgate-2600 | gatecurr-600 | Paired delta (candidate - baseline) | 95% paired bootstrap CI |
|---|---:|---:|---:|---:|
| Action epsilon-MSE, predicted gate | 0.086110 | 0.087323 | +0.001214 | [-0.002714, +0.005286] |
| Retrieval set loss, revisit | 0.204852 | 0.134628 | -0.070224 | [-0.135703, -0.021704] |
| Retrieval top-1 hit, revisit | 89.58% | 87.50% | -2.08 pp | [-10.42, +4.17] pp |
| Gate BCE, all | 0.583970 | 0.505578 | -0.078392 | [-0.118246, -0.037039] |
| Gate accuracy, all | 69.44% | 80.56% | +11.11 pp | [+2.78, +19.44] pp |
| Gate accuracy, revisit | 54.17% | 72.92% | +18.75 pp | [+6.25, +31.25] pp |
| Aux x-y MSE, revisit | 4.376839 | 4.577455 | +0.200615 | [-0.110161, +0.662500] |
| Position error, revisit | 1.5487 m | 1.6070 m | +0.0583 m | [-0.0008, +0.1398] m |
| Camera-rotation error, revisit | 13.617 deg | 13.074 deg | -0.544 deg | [-2.053, +0.641] deg |

The retrieval set loss and gate classifier improved significantly.  Action
loss, top-1 retrieval, position, and rotation did not.  In particular, the
multi-positive retrieval objective rewards the total probability assigned to
the positive *set*.  It does not enforce `max_positive > max_negative`, which
is the condition used by the live `argmax` anchor.  Of 48 revisit samples, 41
were top-1 hits under both checkpoints, four were misses under both, two changed
from hit to miss, and one changed from miss to hit.  Thus the lower retrieval
loss is real but is not evidence of a better live anchor on its own.

### The gate curriculum classified revisits but did not make them useful

Ten previously incorrect gate decisions became correct, while two correct
decisions became incorrect.  Despite that clear classification improvement,
the paired action result remained flat.  The stronger within-checkpoint
intervention is even more diagnostic: on candidate revisit rows, moving the
decoder gate halfway from its prediction toward the GT value increased action
epsilon-MSE by `+0.002724` (95% CI `[+0.001082, +0.005262]`), and replacing it
with the hard GT revisit gate increased it by `+0.005091` (95% CI
`[+0.002329, +0.009156]`).  Only 16.7% of revisit rows benefited from either
intervention.

This means the curriculum succeeded at the question "is this a revisit?" but
did not solve "how should the revisit pose change the waypoint?".  The current
complementary mask also suppresses the image-goal/novel columns as the revisit
gate approaches one.  The fixed result shows that this visual branch remains
useful even on labeled revisits; a harder revisit switch is therefore not a
valid inference-only fix.

### Aux is currently a diagnostic, not an optimized auxiliary task

The default `MEMNAV_AUX_POSE_CALIBRATION=empirical` freezes
`revisit_merge.aux_pose_head`.  Its inputs come from the frozen LingBot camera
head under `no_grad`.  Consequently `w_aux_pose * aux_loss` contributes to the
reported scalar loss but has no gradient path to any trainable parameter in
this run.  W&B aux changes across random training batches must not be read as
aux learning.

The fixed evaluator confirms this mechanically: old/new position errors have
Pearson correlation 0.995 and mean absolute paired change only 0.082 m.  Four
samples contribute 77.5% of the candidate aux MSE; three of those four still
have a correct retrieval top-1.  Candidate position-error median is 0.523 m,
while the mean is 1.607 m and p90 is 5.013 m.  This is a sparse pose/scale
long-tail, not a uniform axis failure and not primarily a retrieval failure.

### Consequence for the next experiment

The next run should not merely increase the ranking weight or extend the same
gate curriculum.  The controlled evidence supports two isolated changes:

1. Add a top-1-consistent hinge margin
   `relu(max_negative - max_positive + margin)` alongside the existing
   multi-positive set loss.  This targets the actual live-anchor decision while
   preserving multiple valid positives.
2. Decouple memory availability from visual-goal suppression.  Keep the visual
   goal branch available as the base policy and inject the revisit branch as a
   gated residual (or learn a separate action-utility gate).  Gate BCE may still
   classify revisit availability, but it should not be assumed to be the
   optimal complementary attention mixture.

Counterfactual positive-vs-hard-negative action supervision remains the most
direct follow-up once this residual fusion is in place.  Making aux trainable is
a separate ablation; a global linear x-y head cannot by itself repair the few
per-episode pose/scale outliers, so it must not be bundled into the causal gate
experiment.

A local compatibility ablation changed only the persistent fusion code on the
same gatecurr-600 weights and evaluated six fixed samples with 16 paired
diffusion trials.  Residual minus complementary action epsilon-MSE was
`+0.000927` (95% CI `[-0.001478, +0.003683]`): no zero-shot gain and no
catastrophic regression.  Retrieval and gate metrics were exactly identical,
as required.  Revisit prediction-to-oracle RMS decreased from 0.02867 to
0.02236 because the visual safety branch no longer disappears at gate=1.  This
is only a wiring/initialization check; the same-origin trained A/B is required
to learn residual-memory utility and decide whether the change is useful.

Full machine-readable outputs are stored under:

`/scratch/yz11502/Research/Nav-axis-uturn-results/fixed_checkpoint_eval/fixed_ckpt2600_vs_gatecurr600_20260803`
