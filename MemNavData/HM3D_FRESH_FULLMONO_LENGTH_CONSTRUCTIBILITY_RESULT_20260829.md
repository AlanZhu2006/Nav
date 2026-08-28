# Fresh HM3D trajectory-length constructibility audit

## Question

Can the already verified fresh Full-Mono HM3D population fill the meeting's
`0--20 m / 20--30 m / 30--50 m` trajectory-length table without constructing
or running a new benchmark?

This audit reads only the sealed role-pair construction manifest. It does not
read native, raw-memory, or CEC navigation outcomes.

## Frozen source

- 28 histories from 21 fresh HM3D scene clusters;
- one Natural Novel and one Revisit query per history;
- 56 total queries;
- manifest SHA-256:
  `aada40d25d01e9385df3ffdcaf37847f471b63c7be785a704eade961346a50b0`;
- construction contract: query geodesic distance in `[2,9]` m.

## Result

| Role | n | minimum | median | maximum | 0--20 m | 20--30 m | 30--50 m |
|---|---:|---:|---:|---:|---:|---:|---:|
| Novel | 28 | 2.250 | 4.802 | 9.000 | 28 | 0 | 0 |
| Revisit | 28 | 2.040 | 2.677 | 3.776 | 28 | 0 | 0 |
| All | 56 | 2.040 | 3.169 | 9.000 | 56 | 0 | 0 |

The existing population therefore cannot support the requested long-distance
bins. Computing SR inside the empty bins, widening boundaries after seeing the
data, or substituting executed path length for the frozen shortest-path
distance would all answer a different question.

## Decision

Do not submit another policy evaluation on this population for Table 3. A
valid length study requires a new result-blind HM3D construction contract that
explicitly samples 20--30 m and 30--50 m shortest paths, verifies scene-level
constructibility and role support before reading policy outcomes, and only
then runs paired NavDP/CEC evaluation. Because the meeting labels Table 3 low
priority, this new benchmark must not preempt the active HM3D continual-memory
chain, cross-controller population construction, or operator-attended robot
trials.

Machine-readable result:
`HM3D_FRESH_FULLMONO_LENGTH_CONSTRUCTIBILITY_20260829.json`.

Reproducer:
`audit_role_pair_length_constructibility.py`.
