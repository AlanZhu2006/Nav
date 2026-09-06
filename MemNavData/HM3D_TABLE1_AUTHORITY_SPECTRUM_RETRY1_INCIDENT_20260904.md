# HM3D Table-I authority-spectrum retry incident

Date: 2026-09-04  
Scope: infrastructure compatibility only; no policy arm was evaluated before the retry

## Incident

Initial jobs `16923257` (smoke), `16923258` (formal array), and `16923259`
(analysis) used immutable task bundle
`hm3d_table1_authority_spectrum_ee8f978f78f66769`. The smoke stopped before
an episode was evaluated because the common launcher passed
`--certified_route_motion_model fundamental_then_pnp` to an older, previously
validated `memnav_server.py` that does not expose that later route-motion CLI.
The authority-spectrum experiment does not use route motion.

The formal and analysis jobs were dependency-cancelled. The failed run produced
no navigation success result, so it did not reveal either previously absent
arm outcome.

## Repair

`run_hm3d_fullmono_server_scene.sh` now checks whether the selected server
advertises `--certified_route_motion_model`:

- a supporting server receives the route-motion arguments as before;
- a legacy server omits them only when all requested values are the frozen
  unused defaults;
- any non-default route request against a legacy server fails closed.

This repair changes launcher capability negotiation, not the four policy arms,
population, thresholds, checkpoints, controller budget, or estimands.

## Verification and retry

- local shell syntax, unit tests, port-pair test, and diff checks passed;
- the retry bundle passed its full `SOURCE_BUNDLE.sha256` verification and was
  made read-only;
- remote authority-spectrum tests: `2 passed`;
- remote import gate passed;
- all three Slurm scripts passed `sbatch --test-only`.

Retry chain:

- smoke: `16924414`;
- formal array: `16924419` (`0-27%4`, dependent on smoke);
- independent analysis: `16924421` (dependent on the formal array).

Immutable retry bundle:
`/scratch/yz11502/Research/Nav-axis-uturn-source-bundles/hm3d_table1_authority_spectrum_805cfb9813046489`

Formal result root:
`/scratch/yz11502/Research/Nav-axis-uturn-results/hm3d_table1_authority_spectrum_20260904/formal_805cfb9813046489`

Scientific disclosure remains unchanged: this is a sealed fixed-population
retrospective authority spectrum, not a fresh confirmation.
