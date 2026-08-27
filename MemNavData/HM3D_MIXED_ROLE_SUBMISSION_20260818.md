# HM3D mixed Novel/Revisit role-unknown safety submission

Date: 2026-08-18

## Question

The completed HM3D val9 evaluation established cross-dataset Revisit utility,
but every executed query was Revisit.  This extension asks whether the same
frozen CEC stack can retain that utility when unsupported Novel and supported
Revisit queries are interleaved without exposing their role to the policy.

This is training-free and uses the same nine constructible HM3D scenes and the
byte-identical saved native Goal-A traces.  Because the scenes were previously
evaluated, it is explicitly a same-scene mixed-role safety extension, not a
new scene-disjoint confirmation.

## Frozen construction

- Novel: maximum co-visibility over the complete online-A history `< 0.10`.
- Revisit: maximum co-visibility in `[0.55, 0.90]`.
- Query distance: `2–9 m` from the exact online-A endpoint.
- One independently rendered Novel/Revisit pair per retained history.
- Each query independently resets and exactly replays the same online-A trace.
- Runtime projection removes role, co-visibility, and construction diagnostics.
- At most three histories per scene are retained in frozen source order.

Five paired arms are evaluated: native, raw direct, raw fixed bearing, geometry
fixed, and certified.  The main estimands are Revisit utility, Novel
interference, CEC versus raw-fixed utility/interference, false takeover, and
exact fallback.

## Submission

```text
RUN_ROOT=/scratch/yz11502/Research/Nav-axis-uturn-results/
  hm3d_mixed_role_20260818/hm3d_mixed_role_20260818T105403Z

TASK_ROOT=/scratch/yz11502/Research/Nav-axis-uturn-source-bundles/
  hm3d_mixed_role_119279411da41afb
```

Jobs:

| Stage | Job | Limit | Dependency |
|---|---:|---:|---|
| 9-scene construction array | 15947671 | 1 h/task | none |
| population audit and seal | 15947673 | 30 min | construction |
| paired evaluation array | 15947675 | 1 h/task | seal |
| summary | 15947677 | 30 min | evaluation |
| independent raw-CSV verification | 15947678 | 30 min | summary |

At submission, construction was pending for `Priority`; every downstream job
was correctly held by `afterok`.  The submission receipt records
`query_policy_outcomes_read_at_submission=false`.

## Interpretation boundary

Passing this test would add cross-domain evidence for role-free open-set
authorization: supported Revisit utility plus unsupported Novel interference
control in HM3D.  It must not be reported as a new-scene generalization result,
because the HM3D scenes have already contributed to the Revisit-only table.
