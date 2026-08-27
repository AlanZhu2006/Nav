# HM3D ViNT native-control / ViNT+CEC mixed-role protocol

Date frozen: 2026-08-28
Status: **protocol and controller-native interface implemented; local gate
verified; four-history pilot queued; formal outcome not yet available**

## 1. The paper question

The conference matrix needs to answer a within-controller question:

> On the same causal online RGB history and the same mixed Novel/Revisit
> queries, does CEC improve a frozen ViNT controller without borrowing NavDP as
> its reject branch?

The older proof-locked portability experiment does not answer this question.
It keeps monocular NavDP as a shared fallback and invokes ViNT only after CEC
acceptance.  That is the correct design for comparing accepted-branch
executors, but it is a heterogeneous controller mixture rather than
`ViNT Native` versus `ViNT + CEC`.

This protocol freezes a different, controller-native treatment:

```text
current RGB + original ImageGoal + causal online history
                      |
                    CEC proof
          +-----------+------------------+
          |                              |
       reject                         accept
          |                              |
same frozen ViNT                 same frozen ViNT
original ImageGoal         SHA-bound historical anchor ImageGoal
```

No branch calls NavDP for its physical action.  The only treatment is whether
an accepted CEC proof is authorized to replace ViNT's goal image for that
decision.

## 2. Frozen population

We reuse the completed outcome-blind HM3D Full-Mono mixed-role population:

- source root:
  `/scratch/yz11502/Research/Nav-axis-uturn-results/hm3d_fresh_fullmono_mixed_role_20260820/formal_20260820T143609Z_e6dd44c6`;
- benchmark:
  `benchmarks/natural_direction/manifest.json`;
- benchmark SHA-256:
  `aada40d25d01e9385df3ffdcaf37847f471b63c7be785a704eade961346a50b0`;
- source verifier SHA-256:
  `3ae4b556eef9e8144f635495f65d58b177ceee8d98301327374967415cf8d2d8`;
- 28 actual-online histories, 21 HM3D scene clusters;
- one Natural Novel and one supported Revisit query per history;
- 56 query starts in total.

The population was previously used for the NavDP rows, so the ViNT result is a
protocol-matched controller ablation on the same paper population, not a new
fresh-scene confirmation.  No ViNT outcome is used for construction,
selection, thresholding, or task scheduling.

## 3. Paired arms

Each history runs both query roles under two same-process arms:

1. **ViNT native-control** (`forced_reject_native`).  The role-free CEC
   pipeline still runs in shadow, but every possible takeover is withheld.
   Every action therefore receives current RGB and the unchanged original
   ImageGoal through the frozen ViNT proxy.
2. **ViNT + CEC** (`grant`).  A rejection executes the identical native ViNT
   request.  An acceptance projects the proof into a SHA-bound certified
   history-anchor ImageGoal and invokes the same ViNT process.

Running the proof in the baseline is deliberate: it makes the two arms differ
only in control authority and lets the same loaded MemNav/ViNT processes and
first proof be audited.  This arm is a native **control** baseline, not a
native wall-clock latency baseline.  Direct ViNT latency is reported
separately.

The following are paired and immutable:

- Habitat scene asset, start pose, goal image and success radius;
- complete online-A replay and causal history digest;
- ViNT source and checkpoint;
- CEC proposal/certificate code and thresholds;
- execution horizon 8, maximum 600 simulator steps;
- GPU, host, MemNav process, ViNT process and controller proxy process;
- first-decision proof, goal boundary and selected anchor;
- balanced arm order.

After an authorized action the physical observations may diverge.  Later proof
streams are therefore not required to remain equal.

## 4. Runtime information boundary

The evaluator retains `analysis_role` only for post-hoc stratification.  The
runtime query contains RGB, goal bytes and non-privileged asset identifiers;
it never contains Novel/Revisit role, oracle pose, ground-truth distance, or
Habitat depth.

CEC rejection is evidence abstention, not a semantic Novel classifier.  Novel
safety is evaluated from all 28 full Novel rollouts.  A first-frame reject is
not accepted as proof that no later false takeover occurs.

## 5. Exact-fallback requirement

For every query on which the grant arm never accepts, the grant and
native-control arms must have identical executed rollout traces and identical
terminal result records.  This is stronger than checking equal SR.  A mismatch
invalidates the exact-fallback claim and fails the cell audit.

If CEC accepts on a Novel query, the rollout remains a valid negative safety
outcome; it is recorded rather than removed.  Infrastructure and method
failures are not conflated.

## 6. Stages

### Stage 0: local true-stack gate

Run a consumed MP3D mixed-role history with `MAX_STEPS=8`.  It can validate
the controller-native request, proof/anchor binding, forced authority, and
exact rejection trace.  Its SR is not evidence.

### Stage 1: four-history HM3D gate

Frozen indices: `0, 7, 14, 21` (four scene clusters).  Each cell runs both
roles and both authority arms.  The gate uses only:

- completion within the frozen time limit;
- no role/depth/runtime leak;
- same-process and first-proof identity;
- valid ViNT checkpoint/source receipt;
- exact fallback on every all-reject query;
- independently verified raw files.

Pilot SR, gains, and losses are never used to select or modify the formal arm.

### Stage 2: complete formal population

All 28 histories run after the Stage-1 verifier exits successfully.  The
formal task is pre-registered before seeing pilot outcomes and is not changed
based on those outcomes.

## 7. Reported quantities

For Novel, Revisit, and balanced All:

- native-control and CEC SR;
- SPL, path length, steps, and final distance;
- paired gains/losses and exact two-sided McNemar test;
- scene-cluster bootstrap 95% interval for paired risk difference;
- queries/plans with CEC takeover;
- first-proof accept/reject counts;
- all-reject exact-fallback trace matches;
- failures by proof, controller, and infrastructure stage.

The Table-I portability claim is supported only by the within-ViNT paired
difference.  Absolute ViNT/NavDP scores are contextual because their training
and native controller interfaces differ.

## 8. Implementation

- comparison contract: `MemNavData/controller_portability_contract.py`;
- controller proxy: `MemNavData/controller_portability_proxy.py`;
- role-free hub: `MemNavData/cec_controller_portability_hub.py`;
- paired runner: `MemNavData/run_cec_controller_portability_smoke_local.sh`;
- cell auditor: `MemNavData/audit_vint_controller_native_pair.py`;
- aggregate: `MemNavData/aggregate_vint_controller_native_hm3d.py`;
- independent verifier:
  `MemNavData/independent_verify_vint_controller_native_hm3d.py`.

The older `CONTROLLER_PORTABILITY_PROOF_LOCKED_PROTOCOL_20260827.md` remains
frozen and answers its original shared-fallback executor question.  Its result
must not be silently relabeled as this within-ViNT comparison.

## 9. Frozen submission

The local true-stack gate passed on 2026-08-28: the all-reject Novel trace was
action-identical across arms, while the Revisit grant carried a verifiable
single-use handoff packet for certified anchor 121.  This eight-step gate is
infrastructure evidence only and contributes no SR observation.

The immutable HPC dependency chain is recorded in
`HM3D_VINT_CONTROLLER_NATIVE_CEC_SUBMISSION_RECEIPT_20260828.json`.  Pilot job
`16482370_[0-3%2]` was queued before reading any ViNT outcome.  The complete
28-history array is dependency-held and can start only after pilot aggregation
and independent verification both exit successfully.
