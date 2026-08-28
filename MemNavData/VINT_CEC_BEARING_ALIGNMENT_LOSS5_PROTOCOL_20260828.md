# ViNT + CEC initial-bearing alignment: Loss-5 mechanism protocol

Date frozen: 2026-08-28
Status: **consumed, outcome-aware mechanism test; never a paper SR result**

## 1. Question

The completed 28-history HM3D experiment showed that certified-anchor goal
substitution did not make ViNT consume CEC's direction.  This test asks:

> If the already-certified robot-local bearing is applied once before ViNT's
> first local horizon, does physical motion change from moving away to moving
> toward the Revisit goal?

It is not a new relocalizer, a trained adapter, an oracle-bearing arm, or a
controller-portability confirmation.

## 2. Consumed population

The frozen subset contains exactly the five formal Revisit queries for which
ViNT native succeeded and the certified-anchor treatment failed:

```text
cYkrGrCg2kB  episode_0002
QHhQZWdMpGJ  episode_0000
uSKXQ5fFg6u  episode_0003
LEFTm3JecaC  episode_0001
LEFTm3JecaC  episode_0003
```

Selection manifest:

`MemNavData/vint_cec_direction_loss5_manifest_20260828.json`

The manifest contains query identities but no runtime role field.  Selection
is explicitly outcome-aware, so any positive endpoint result requires a fresh
outcome-blind confirmation before entering a paper table.

## 3. Three same-process arms

Each query runs all arms with the same frozen history, checkpoint, simulator,
600-step budget, success radius, and loaded MemNav/ViNT processes.  Arm order
rotates across array cells.

1. `anchor_unaligned`
   - accepted CEC proof;
   - certified historical anchor replaces the ImageGoal;
   - no bearing alignment;
   - reproduces the failed treatment.

2. `native_bearing_aligned`
   - the shadow-accepted CEC packet supplies its certified bearing;
   - ViNT retains the original ImageGoal;
   - the bearing is applied once before ViNT's unchanged local trajectory.

3. `anchor_bearing_aligned`
   - accepted CEC packet supplies its certified bearing;
   - ViNT uses the certified anchor ImageGoal;
   - the same one-time alignment precedes its unchanged local trajectory.

The two aligned arms isolate whether direction alone is sufficient and whether
the historical anchor adds value after orientation is corrected.

## 4. Alignment contract

At the first accepted or shadow-accepted proof only:

```text
theta = atan2(direction_left, direction_forward)
yaw_after = wrap(yaw_before + theta)
```

The packet envelope, proof digest, units, and absence of privileged role fields
must verify before the turn.  The mechanism uses an idealized zero-translation
yaw followed by the controller's unchanged robot-local trajectory.  This is
the same kind of capability isolation as the earlier oracle-yaw mechanism
gate, but the direction here comes from deployable CEC evidence rather than
Habitat.

This turn is intentionally not yet a deployable actuator result.  A passing
mechanism must later replace it with bounded physical turn steps and fresh
observations.

## 5. Frozen interpretation gates

Primary direction-consumption gate:

- exactly one certified alignment in every aligned rollout;
- executed first-horizon heading error to the sealed bearing no greater than
  30 degrees in at least 4/5 queries;
- goal distance decreases over the first horizon in at least 4/5 queries.

Exploratory endpoint gate:

- an aligned arm recovers at least 3/5 of the selected failures.

If the first gate fails, stop ViNT portability work and inspect the common
executor.  If the first gate passes but endpoint recovery fails, orientation
was necessary but insufficient; the next diagnosis is route/topomap context,
not more CEC retrieval training.  Only if both gates pass may a fresh,
outcome-blind bounded-turn evaluation be designed.

## 6. Non-claims

This test cannot establish:

- a paper-level ViNT improvement;
- deployment-ready turning;
- Novel safety;
- superiority over full official ViNT topological navigation;
- controller-agnostic portability.
