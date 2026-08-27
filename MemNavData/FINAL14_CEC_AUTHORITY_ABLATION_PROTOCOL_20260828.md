# Final14 CEC authority-only ablation

This consumed-population experiment fills the conference mechanism row that
cannot be answered by raw-DINO versus CEC.  Both arms replay the same Final14
causal history and use the same monocular query controller, candidate
shortlist, local matcher, Fundamental-MAGSAC ranking, historical depth, PnP,
fixed 2.5 m bearing adapter, and frozen NavDP.  The only changed variable is
whether operational certificate thresholds control intervention.

| Arm | Pose computation | Intervention authority |
|---|---|---|
| `mono_cec` | identical DINO + geometry + PnP | full v2 certificate |
| `mono_unthresholded_witness` | identical DINO + geometry + PnP | any finite PnP pose |

The second arm is **not** retrieval-only and is **not** geometry-free.  It is
an intentionally unsafe diagnostic that removes the four operational
thresholds only after a PnP witness can be formed.  Fewer than eight local
matches still cannot produce a PnP pose; this is an algorithmic existence
condition, not an authorization threshold.

The primary estimands are paired strict-CEC minus unthresholded-witness SR for
Novel, Revisit, and all 42 queries.  Secondary outputs are Novel authorization
count, Revisit rejection count, and the number of first-decision authority
discordances.  Runtime never receives the Novel/Revisit role.

Every history runs both arms in one process pair with rotated order.  The
verifier requires the first DINO order, geometry-ranked order, selected anchor,
and selected DINO rank to match exactly before accepting an
authorization-only interpretation.
