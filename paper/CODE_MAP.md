# Exact code boundary

This document distinguishes the paper method from experimental branches that coexist in the research runtime.

## Primary runtime

### 1. Streaming integration

- [`NavDP/baselines/memnav/memnav_server.py`](../NavDP/baselines/memnav/memnav_server.py)
  - enables the frozen endpoint with `--certified_relocalization`;
  - creates the pinned SuperPoint/LightGlue matcher;
  - exposes `POST /certified_relocalize`;
  - optional Phase-B, CDEC and Pi3X flags are disabled in the paper method.
- [`NavDP/baselines/memnav/policy_agent.py`](../NavDP/baselines/memnav/policy_agent.py)
  - `MemNavAgent.certified_relocalize`: proposal, PnP, certificate, cache and abstention lifecycle;
  - `_certified_reference_depth`: causal historical-frame depth replay;
  - `_certified_bearing_vector`: discards monocular metric scale;
  - graph rescue, CDEC rescue and learned Pi3X routes are not part of CEC v1.
- [`NavDP/baselines/memnav/reverse_memory_graph.py`](../NavDP/baselines/memnav/reverse_memory_graph.py) remains an import-compatible support module; graph rescue is disabled in CEC v1.
- [`NavDP/baselines/navdp/policy_agent.py`](../NavDP/baselines/navdp/policy_agent.py), [`policy_network.py`](../NavDP/baselines/navdp/policy_network.py), and [`navdp_server.py`](../NavDP/baselines/navdp/navdp_server.py) are the frozen execution stack used by both native fallback and accepted residuals.

### 2. Deterministic proof boundary

- [`MemNavData/certified_relocalization_runtime.py`](../MemNavData/certified_relocalization_runtime.py)
  - `fundamental_support`;
  - `rank_candidates` and `candidate_rank_key`;
  - `fundamental_can_reach_certificate`;
  - `certificate_decision`;
  - `scale_free_relative_xy`;
  - `runtime_contract`.
- [`MemNavData/lingbot_pnp_localization.py`](../MemNavData/lingbot_pnp_localization.py)
  - `LightGluePointMatcher`;
  - `correspondence_pnp_localize`.
- [`MemNavData/lingbot_colored_registration.py`](../MemNavData/lingbot_colored_registration.py)
  - deterministic pose/depth registration primitives used by PnP.
- [`MemNavData/revisit_bearing_adapter.py`](../MemNavData/revisit_bearing_adapter.py)
  - `adapt_revisit_pointgoal`;
  - `verified_bearing_v1` is the primary interface;
  - `raw_fixed_bearing_v1` is an ablation only.

### 3. Closed-loop evaluator

- [`MemNavData/eval_2leg_habitat.py`](../MemNavData/eval_2leg_habitat.py): shared two-leg controller/evaluator.
- [`MemNavData/eval_3leg_habitat.py`](../MemNavData/eval_3leg_habitat.py): sequential multi-leg extension.
- [`MemNavData/generate_twoleg.py`](../MemNavData/generate_twoleg.py): rendering, geodesic and episode construction primitives.
- [`MemNavData/deterministic_eval_protocol.py`](../MemNavData/deterministic_eval_protocol.py): shared seeds and trace identity.
- [`MemNavData/revisit_action_shadow.py`](../MemNavData/revisit_action_shadow.py): action-path equality auditing.
- [`MemNavData/navdp_goal_switch.py`](../MemNavData/navdp_goal_switch.py): policy-state reset/carry rules.
- [`MemNavData/goat_terminal_alignment.py`](../MemNavData/goat_terminal_alignment.py): a pure optical-frame alignment helper reused by the CEC runtime; the GOAT benchmark route itself is excluded.

The evaluator contains compatibility code for older and secondary arms. A paper CEC run must satisfy:

```text
hybrid_route = certified_relocalization
revisit_adapter = verified_bearing_v1
certified_cdec_rescue = off
graph rescue = off
Pi3X learned route = off
X-NavDP controller = off
runtime role visibility = none
```

The authoritative machine-readable contract is [`paper/configs/cec_v1.json`](configs/cec_v1.json).

## Final14 population and analysis

- [`MemNavData/final14_role_pair_contract.py`](../MemNavData/final14_role_pair_contract.py): Natural/Hard support definitions and deterministic yaw/direction strata.
- [`MemNavData/freeze_final14_source_manifest.py`](../MemNavData/freeze_final14_source_manifest.py): source freeze before policy outcomes.
- [`MemNavData/build_final14_role_pair_scene.py`](../MemNavData/build_final14_role_pair_scene.py): role-pair construction from actual-online histories.
- [`MemNavData/audit_final14_role_pairs.py`](../MemNavData/audit_final14_role_pairs.py): independent constructibility/identity audit.
- [`MemNavData/finalize_final14_role_pairs.py`](../MemNavData/finalize_final14_role_pairs.py): population finalization and receipts.
- [`MemNavData/materialize_online_a_traces.py`](../MemNavData/materialize_online_a_traces.py): actual-online Goal-A materialization.
- [`MemNavData/eval_shared_online_role_pairs.py`](../MemNavData/eval_shared_online_role_pairs.py): paired query execution.
- [`MemNavData/summarize_paper_role_pair_eval.py`](../MemNavData/summarize_paper_role_pair_eval.py): primary summary.
- [`MemNavData/independent_verify_paper_role_pair_eval.py`](../MemNavData/independent_verify_paper_role_pair_eval.py): independent raw-file recount.

## External HM3D analysis

- [`MemNavData/build_hm3d_heldout_val10_revisit_manifest.py`](../MemNavData/build_hm3d_heldout_val10_revisit_manifest.py)
- [`MemNavData/summarize_hm3d_heldout_val10_revisit.py`](../MemNavData/summarize_hm3d_heldout_val10_revisit.py)
- [`MemNavData/verify_hm3d_heldout_val10_revisit.py`](../MemNavData/verify_hm3d_heldout_val10_revisit.py)
- [`MemNavData/hm3d_heldout_val10_revisit_protocol_20260816.json`](../MemNavData/hm3d_heldout_val10_revisit_protocol_20260816.json)

## Secondary learned relocalizer

These files reproduce the learned Pi3X arm but are not the primary paper method:

- [`MemNavData/pi3x_online_relocalizer.py`](../MemNavData/pi3x_online_relocalizer.py)
- [`MemNavData/pi3x_spatial_reliability_model.py`](../MemNavData/pi3x_spatial_reliability_model.py)
- [`MemNavData/pi3x_spatial_proof_runtime.py`](../MemNavData/pi3x_spatial_proof_runtime.py)
- [`MemNavData/train_pi3x_spatial_reliability_crossfit_oof.py`](../MemNavData/train_pi3x_spatial_reliability_crossfit_oof.py)
- [`MemNavData/fit_pi3x_spatial_reliability_deployment.py`](../MemNavData/fit_pi3x_spatial_reliability_deployment.py)
- [`MemNavData/compare_pi3x_spatial_proof_to_certificate.py`](../MemNavData/compare_pi3x_spatial_proof_to_certificate.py)

It retains DINO addressing, uses a b16 causal visual bridge and four small task-trained spatial proof heads. It passed utility gate L1 but failed non-inferiority/safety gates L2/L3.

## Deliberately excluded from the primary path

- graph rescue;
- CDEC learned rescue;
- Phase-B learned activation;
- X-NavDP controller replacement;
- oracle/frontier Novel bearing;
- active glance or original-position scanning;
- GOAT semantic arrival adapters;
- candidate-free long-history transformer.

These may remain elsewhere in the research history, but none should be enabled in a CEC-v1 result.
