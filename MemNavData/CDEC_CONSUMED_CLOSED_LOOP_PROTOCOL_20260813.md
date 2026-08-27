# CDEC consumed-pool closed-loop protocol — frozen before PnP outcome

## Purpose and authority

This protocol is written before `report_repeatability_v2.json` exists.  It
prevents the train-only PnP result from changing the downstream comparison.
It is authorized only when both the official dual-proposal report and the
independent raw-CSV verifier agree that the frozen method gate passed.

The question is deliberately narrow:

> On the already consumed 20-scene / 160-episode Revisit benchmark, does a
> scene-OOF learned proposal, consulted only after the geometry proposal's
> atomic PnP certificate rejects, safely recover additional closed-loop
> successes over the same geometry-only certified bearing system?

This is not fresh-scene evidence and never authorizes threshold, radius,
certificate, feature, or controller tuning on the pool.

The executable preparation path is
`prepare_cdec_consumed_closed_loop.py`.  It binds the old Goal-A source run,
manifest, dependencies, and each trace by SHA256 while treating trace payloads
as opaque bytes.  Its regression fixture deliberately uses invalid JSON trace
files: successful preparation therefore cannot depend on reading a target or
outcome.  Full trace semantics are validated only during causal replay.

## Frozen system

Both arms run in one persistent MemNav/LingBot process and one persistent
NavDP process per scene.  They replay the same immutable online Goal-A trace,
episode seed, and deterministic per-plan diffusion seed.

```text
causal DINO top-8
       |
geometry proposal -> unchanged atomic PnP certificate
       | pass                         | reject
       v                              v
scale-free bearing             baseline: native ImageGoal
                                      |
                               CDEC top-1 proposal
                                      |
                              same atomic certificate
                               pass / reject->native
```

Arms:

1. `geometry_certificate`: the server has the CDEC artifact loaded, but every
   request explicitly sends `learned_rescue=0`;
2. `cdec_cascade`: the same server sends `learned_rescue=1`; learned ranking
   cannot run before geometry rejection and cannot override a geometry pass.

The artifact is frozen at SHA256
`eea77098531c6e5865516d49540374edca7bb87a419613ef40d56cc2c85add31`.
The research-only loader override permits this pre-approval experiment; it
does not silently change `deployment_approved=false`.  Both arms retain the
same `verified_bearing_v1` fixed 2.5 m residual, frozen NavDP, PnP thresholds,
success radius, execution horizon, and 500-step budget.  Stagnation graph
rescue is disabled so the experiment isolates proposal coverage.

## Pre-run gate

No closed-loop task may start unless all are true:

- the dual collector contains 480 paired sessions / 960 rows from exactly
  train40, with no development/blind read;
- geometry-first CDEC adds at least one certified-actionable session, loses
  none, and adds no certificate false positive;
- every repeated same-anchor certificate decision agrees;
- the independent verifier reproduces every authoritative policy, paired,
  proposal-identity, and gate field;
- the runtime parity artifact remains exact on all 480 shortlist sessions and
  the production-DINO parity receipt remains pinned.

If this gate fails, the closed-loop run is not submitted and CDEC is retained
as a negative learned-proposal result.

## Causal and safety audits

For every paired episode the summary must verify:

- identical Goal-A trace SHA, Goal-A outcome/steps/path/final distance, and
  exact `legA_memory_trace`;
- identical episode/geodesic/seed contracts and counterbalanced arm order;
- baseline has zero learned requests/invocations/selections;
- CDEC requests learned rescue, but its proposal is absent whenever geometry
  accepts;
- geometry's first proposal anchor and certificate decision match across arms;
- every learned takeover follows a recorded geometry rejection and carries a
  newly accepted copy of the unchanged certificate;
- every reject executes native ImageGoal; there is no metric-scale leakage;
- zero CDEC runtime failure and no gain/loss without an actual learned
  certified takeover.

## Frozen statistics and decision

Report `SR_A`, `SR_B|A`, joint SR, paired gains/losses, exact two-sided
McNemar, and a 100,000-resample scene-cluster bootstrap interval.  Also report
learned invocation, learned certificate acceptance, takeover, same-anchor
reuse, and runtime-failure counts.

Promotion beyond this consumed pool requires all of:

1. at least one gain in at least two scene clusters;
2. zero paired losses;
3. exact McNemar `p < 0.05`;
4. scene-cluster 95% interval lower bound strictly above zero;
5. every gain contains a learned certified takeover and every safety/causal
   audit passes.

Otherwise the branch is `do_not_promote_cdec`; no retuning on this pool is
allowed.  Even a pass authorizes only inclusion in an already frozen one-shot
held-out system comparison; it is not itself a paper-final generalization
claim.

After the primary report, a dependency-chained verifier
`independent_verify_cdec_consumed_closed_loop.py` reconstructs the raw 160
pairs without importing the primary summarizer.  It independently recomputes
the success counts, McNemar test, scene bootstrap, learned-takeover attribution,
no-treatment equality, and exact frozen promotion decision.  A primary report
without this second report is not an accepted result.

## Relationship to 3-leg

The learned proposal is not added to 3-leg before this 2-leg causal test.  The
currently frozen 3-leg method remains:

`geometry certificate -> direct bearing -> stagnation certificate -> causal
history-arc rescue`.

Only a consumed-pool CDEC promotion may insert learned-on-reject before that
same direct-bearing stage.  The graph rescue and CDEC are not jointly tuned.
