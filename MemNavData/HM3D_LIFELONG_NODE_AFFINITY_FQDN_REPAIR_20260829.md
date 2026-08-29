# HM3D lifelong node-affinity FQDN repair

Date frozen: 2026-08-29  
Status: infrastructure-only launcher repair; no B2 navigation job was submitted

## Incident

After all 22 factual-C items passed the integrity barrier and the population was
sealed, resume launcher `16521679` failed in
`hm3d_lifelong_node_affinity.py`. Every hash-bound `compute_identity.json`
records the NYU compute host in fully qualified form, for example
`gh005.hpc.nyu.edu`, while the launcher accepted only the equivalent Slurm
short form `gh005`.

The failure occurred while constructing the node-affinity plan, before the B2
smoke or any B2 formal evaluation was submitted. No factual-C success, B2
success, SPL, distance, or action outcome was read to diagnose or design this
repair.

## Minimal repair

The parser now accepts exactly two receipt forms:

1. a valid Slurm short name matching `ghNNN` or `gaNNN`;
2. that same short name followed by the exact suffix `.hpc.nyu.edu`.

It strips only that suffix, validates the remaining short name, maps `gh` to
`h100_tandon` and `ga` to `a100_tandon`, and passes the short name to both
`--nodelist` and `EXPECTED_REPLAY_NODE`. Any other hostname or suffix still
fails closed.

This preserves the scientific requirement that each B2 replay runs on the
physical node that produced its factual-C prefix. It changes no population,
history, RGB bytes, checkpoint, policy, arm, order, threshold, seed, budget,
success criterion, or analysis rule. The sealed 22-history/15-scene population
is reused without collection or materialization reruns.

## Frozen continuation

- amendment bundle:
  `hm3d_lifelong_node_affinity_repair_ddd01842308dfa37`;
- bundle receipt SHA-256:
  `ddd01842308dfa372d08551b00e1434a0049703625137a4c3e94f6340ee22b1e`;
- repaired resume launcher: `16540396`, completed `0:0`;
- emitted B2 true-stack smoke: `16540468`;
- emitted node-affine formal launcher: `16540469`, dependent on the smoke;
- the sealed downstream B2 population contains 17 accepted histories. The
  upstream factual-C integrity barrier still covers all 22 histories from 15
  scenes; the two counts refer to different stages and must not be conflated.

At submission time the smoke was pending on `QOSMaxGRESPerUser`, not on an
invalid dependency. No B2 navigation outcome had been produced or inspected.
