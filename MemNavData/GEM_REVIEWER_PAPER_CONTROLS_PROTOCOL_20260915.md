# Main-table reviewer controls, frozen 2026-09-15

Purpose: distinguish memory-guided initial orientation, continued bearing, and explicit turning on queries with old historical support and weak recent visual support.

Population: all 13 original main-table NavDP histories selected by old eligible maximum co-visibility >= 0.5, maximum among the last seven historical decision observations < 0.1, and the current-query visibility upper bound < 0.1. Query outcomes do not enter selection. This is a metadata subgroup of an existing population, not an independent test set. Selection and current-view projection receipts are stored in `.diagnostics/gem_reviewer_p0_20260915`.

Five arms per history: native ImageGoal; unconditional positive 180 degree turn then native ImageGoal; GEM bearing with no explicit in-place turn; GEM initial rearward alignment only then native ImageGoal; full GEM. Native and unconditional-turn arms retain the same current monocular depth sidecar, but cannot invoke episodic retrieval, PnP or mixed point-goal control. No goal-role labels or simulator poses enter the model. Turns consume the shared 600-action budget. Same original RGB goal, seed, frozen NavDP, eight-action planning interval, and bounded execution law in every arm.

The paper-matching geometry (`legacy`, window 32, canonical reference depth) and strict certificate are held fixed. These results are reported separately from the consumed four-history W64/support-storage local pilot. This study tests the contribution of the control interface without changing the geometry condition behind Table I.

Each history owns one Slurm array element and all five arms run sequentially, resetting the same model servers from the same original RGB history. Actual physical GPU UUID and node identity are recorded. Arm order is cyclically shifted by frozen cell index. The longest eligible history (564 frames) is the environment/timing gate; its results remain part of the total 65 rollouts.

Primary contrasts: full GEM versus fixed turn, initial-only, and no-turn. Report paired wins/losses and exact McNemar tests with Holm adjustment across these three contrasts. All outcomes and failures are retained; small sample size cannot establish equivalence. Also report actions, explicit-turn actions, path length and SPL. The shared 1 m automatic arrival criterion is retained: these runs do not test autonomous STOP or tighter thresholds.

Independent checks reconstruct every action from saved trajectories and navigation meshes, verify actual travelled distance, success and SPL, inspect actual RGB/monocular-depth transactions, verify memory-free endpoints, compare native/fixed first samples, and compare first localization and guided-turn prefixes across memory arms. Infrastructure or evidence-verification failures cannot be counted as navigation failures or silently replaced. Source and input hashes bind every result to the frozen plan.

The nearest-history retrieval control, positional-offset closed-loop test, image substitution for ViNT/NoMaD, and stricter success criteria remain separate experiments. This bounded study does not claim to complete all reviewer suggestions.
