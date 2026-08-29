# HM3D Table-1 NavDP canonical-stratum analysis repair

The frozen role-pair writer stores `assigned_direction_stratum` on the Novel
query. The original NavDP Table-1 aggregator instead indexed a duplicated
history-level field that exists only in its unit fixture, not in the verified
formal manifest. Consequently the original aggregate job can fail after all
rollouts are complete with a `KeyError`; it does not change or invalidate any
raw navigation outcome.

This repair changes one read path only:

```text
history["selected_direction_stratum"]
    -> unique Novel query["assigned_direction_stratum"]
```

The replacement analysis:

- waits for the original aggregate attempt with `afterany`, preventing a write
  race;
- reads the same 28-history / 21-scene verified benchmark and the same raw
  NavDP rollout directories;
- changes no population, checkpoint, observation, action, seed, threshold,
  success rule, bootstrap rule, or claim scope;
- writes a new summary only if the original summary is absent;
- runs the existing independent raw-file verifier afterward;
- creates a replacement cross-controller seal only after both the repaired
  NavDP verifier and the retained ViNT verifier pass.

No partial SR or success outcome was read to identify this issue; it follows
directly from comparing the immutable manifest writer and aggregator schemas.
