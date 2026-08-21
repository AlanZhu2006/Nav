# HPC repair protocol (2026-08-21)

This repair resumes two jobs that stopped before any new navigation outcome
was produced.

## HM3D Full-Mono mixed-role chain

The original source generation array `16080301` completed all 54 scene tasks.
It produced 49 complete four-episode scenes (196 episodes) and five explicit
scene-level constructibility attritions.  The parent job `16080319` then
stopped on `AYpsNQsWncn/episode_0003`, whose legacy two-leg carrier has a
22-frame recall gap rather than 32 frames.

That carrier is not consumed by the formal query.  Goal-A collection invokes
`eval_2leg_habitat.py --stop_after_leg1`; Natural Novel/Revisit queries are
then constructed independently from the actual online-A trace and retain all
original support and gap checks.  The repair therefore records the carrier
gap as provenance instead of using it as Goal-A eligibility.  It does not
regenerate, filter, or modify any source episode and does not inspect query
outcomes.

The original source-generation directory, protocol, assets, task bundle and
base source bundle remain immutable.  A new parent-manifest job uses only the
patched parent builder from a content-addressed repair bundle, then starts a
new downstream dependency chain in the original result root.

## Controller-portability pilot

Job `16085641_0` stopped before model startup with the message
`base source receipt changed`.  The frozen receipt itself was unchanged.  The
submitted environment value contained a 65-character typo: an extra trailing
`a` after the real 64-character SHA-256.  The repair uses the independently
recomputed 64-character value and does not relax the hash gate.  It also logs
the label, path, expected value, actual value, and file metadata on a
persistent mismatch.  All bundle-content checks remain exact.

The pilot remains a latency/failure-mode smoke only.  It runs history index 0
for NavDP, ViNT, iPlanner and ViPlanner with the previously frozen benchmark,
checkpoints, maximum 80 steps, and role-blind CEC contract.
