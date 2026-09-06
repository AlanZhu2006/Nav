#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/asus/Research/Nav-graph-blind
MIRROR_ROOT="$ROOT/.diagnostics/long_range_path_field_20260902/local_mirror"
MANIFEST="$ROOT/.diagnostics/long_range_path_field_20260902/source/population/manifest.json"
PROTOCOL="$ROOT/MemNavData/MONOCULAR_ROUTE_TANGENT_COMPASS_PROTOCOL_20260903.md"
FREEZE="$ROOT/.diagnostics/mono_adjacent_motion_20260903/route_tangent_confirmation_freeze_v1/confirmation_freeze.json"
QUERY_ROOT="$ROOT/.diagnostics/mono_adjacent_motion_20260903/dense_reverse_index39_v2"
QUERY_SET="$QUERY_ROOT/query_set.json"
QUERY_AUDIT_ROOT="$ROOT/.diagnostics/mono_adjacent_motion_20260903/index39_dense_v2_confirmation_attempt1"
QUERY_AUDIT="$QUERY_AUDIT_ROOT/adjacent_motion_audit.json"
ROUTE_AUDIT_ROOT="$ROOT/.diagnostics/mono_adjacent_motion_20260903/route_compass_index39_confirmation_attempt1"
ROUTE_AUDIT="$ROUTE_AUDIT_ROOT/monocular_route_compass_audit.json"
TANGENT_ROOT="$ROOT/.diagnostics/mono_adjacent_motion_20260903/tangent_path_budgeted_index39_confirmation_v1"
TANGENT="$TANGENT_ROOT/path_budgeted_route_replay.json"
GATE_ROOT="$ROOT/.diagnostics/mono_adjacent_motion_20260903/route_tangent_index39_gate_attempt1"
SERVER_ROOT="$ROOT/.diagnostics/mono_adjacent_motion_20260903/server_index39_depthonly_v1"
SERVER_LOG="$SERVER_ROOT/server.log"
PORT=22675

HABITAT_PY=/home/asus/miniconda3/envs/habitat/bin/python3.9
RUNTIME_PY=/home/asus/miniconda3/envs/memnav-realworld/bin/python
LIGHTGLUE_REPO="$ROOT/.diagnostics/dependencies/LightGlue"
LIGHTGLUE_DEPS="$ROOT/.diagnostics/dependencies/python"
CHECKPOINT=/home/asus/Research/Nav-axis-uturn/.diagnostics/unseen_scene_eval_20260803/checkpoints/gatecurr600.memnav.ckpt

EXPECTED_PROTOCOL_SHA=4200787139e73c9d27354292eccd25a8337282a31a8f1c49ae78fceba7c9a3d4
EXPECTED_FREEZE_SHA=3fb0c2d1401107e43230bf077f3fa8d383f4b41c99da80db59fd7c56059ba3cc

cd "$ROOT"
test "$(sha256sum "$PROTOCOL" | awk '{print $1}')" = "$EXPECTED_PROTOCOL_SHA"
test "$(sha256sum "$FREEZE" | awk '{print $1}')" = "$EXPECTED_FREEZE_SHA"
test -x "$HABITAT_PY"
test -x "$RUNTIME_PY"
test -d "$LIGHTGLUE_REPO"
test -d "$LIGHTGLUE_DEPS"
test -f "$CHECKPOINT"

if [[ ! -f "$QUERY_SET" ]]; then
    test ! -e "$QUERY_ROOT"
    PYTHONPATH="$ROOT" "$HABITAT_PY" \
        MemNavData/generate_hm3d_dense_reverse_route_queries.py \
        --mirror-root "$MIRROR_ROOT" \
        --manifest "$MANIFEST" \
        --history-index 39 \
        --spacing-m 0.30 \
        --max-turn-step-deg 15 \
        --out "$QUERY_ROOT"
fi
(cd "$QUERY_ROOT" && sha256sum -c query_set.json.sha256)
QUERY_SHA=$(sha256sum "$QUERY_SET" | awk '{print $1}')

for path in "$QUERY_AUDIT_ROOT" "$ROUTE_AUDIT_ROOT" "$TANGENT_ROOT" "$GATE_ROOT" "$SERVER_ROOT"; do
    test ! -e "$path"
done
if ss -ltn | awk '{print $4}' | grep -Eq "(^|:)$PORT$"; then
    echo "port $PORT is already in use" >&2
    exit 1
fi

mkdir -p "$SERVER_ROOT"
PYTHONPATH="$ROOT:$ROOT/InternNav" "$RUNTIME_PY" -u \
    NavDP/baselines/memnav/memnav_server.py \
    --host 127.0.0.1 \
    --port "$PORT" \
    --checkpoint "$CHECKPOINT" \
    --internnav_root "$ROOT/InternNav" \
    --device cuda:0 \
    --num_samples 1 \
    --exclude_recent 32 \
    --retrieval raw \
    --retrieval_candidate_top_k 32 \
    --retrieval_candidate_min_gap 16 \
    --graph_subgoal_spacing_m 0.0 \
    --graph_subgoal_arrival_m 0.60 \
    --flow_gate auto \
    --depth_observation_only \
    --buffer_root "$SERVER_ROOT/buffer" \
    >"$SERVER_LOG" 2>&1 &
SERVER_PID=$!

cleanup() {
    if kill -0 "$SERVER_PID" 2>/dev/null; then
        kill "$SERVER_PID"
        wait "$SERVER_PID" || true
    fi
}
trap cleanup EXIT INT TERM

ready=0
for _ in $(seq 1 180); do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        tail -n 100 "$SERVER_LOG" >&2
        exit 1
    fi
    if "$RUNTIME_PY" - "$PORT" <<'PY'
import socket
import sys
with socket.socket() as sock:
    sock.settimeout(0.2)
    sock.connect(("127.0.0.1", int(sys.argv[1])))
PY
    then
        ready=1
        break
    fi
    sleep 1
done
test "$ready" -eq 1

PYTHONPATH="$ROOT" "$RUNTIME_PY" \
    MemNavData/run_local_monocular_adjacent_motion_audit.py \
    --mirror-root "$MIRROR_ROOT" \
    --manifest "$MANIFEST" \
    --query-set "$QUERY_SET" \
    --history-index 39 \
    --port "$PORT" \
    --lightglue-repo "$LIGHTGLUE_REPO" \
    --lightglue-dependency-root "$LIGHTGLUE_DEPS" \
    --device cuda:0 \
    --out "$QUERY_AUDIT_ROOT"
QUERY_AUDIT_SHA=$(sha256sum "$QUERY_AUDIT" | awk '{print $1}')

PYTHONPATH="$ROOT" "$RUNTIME_PY" \
    MemNavData/run_local_monocular_route_compass_audit.py \
    --mirror-root "$MIRROR_ROOT" \
    --manifest "$MANIFEST" \
    --query-set "$QUERY_SET" \
    --query-motion-audit "$QUERY_AUDIT" \
    --expected-query-motion-sha "$QUERY_AUDIT_SHA" \
    --authorization-receipt "$FREEZE" \
    --expected-authorization-sha "$EXPECTED_FREEZE_SHA" \
    --history-index 39 \
    --port "$PORT" \
    --lightglue-repo "$LIGHTGLUE_REPO" \
    --lightglue-dependency-root "$LIGHTGLUE_DEPS" \
    --device cuda:0 \
    --out "$ROUTE_AUDIT_ROOT"
ROUTE_AUDIT_SHA=$(sha256sum "$ROUTE_AUDIT" | awk '{print $1}')

PYTHONPATH="$ROOT" "$RUNTIME_PY" \
    MemNavData/analyze_path_budgeted_monocular_route.py \
    --route-audit "$ROUTE_AUDIT" \
    --expected-route-audit-sha "$ROUTE_AUDIT_SHA" \
    --query-motion-audit "$QUERY_AUDIT" \
    --expected-query-motion-sha "$QUERY_AUDIT_SHA" \
    --guidance-mode route-tangent \
    --out "$TANGENT_ROOT"
TANGENT_SHA=$(sha256sum "$TANGENT" | awk '{print $1}')

set +e
PYTHONPATH="$ROOT" "$RUNTIME_PY" \
    MemNavData/verify_route_tangent_confirmation.py \
    --protocol "$PROTOCOL" \
    --freeze "$FREEZE" \
    --expected-freeze-sha "$EXPECTED_FREEZE_SHA" \
    --query-set "$QUERY_SET" \
    --expected-query-set-sha "$QUERY_SHA" \
    --query-motion "$QUERY_AUDIT" \
    --expected-query-motion-sha "$QUERY_AUDIT_SHA" \
    --history-route "$ROUTE_AUDIT" \
    --expected-history-route-sha "$ROUTE_AUDIT_SHA" \
    --tangent-replay "$TANGENT" \
    --expected-tangent-replay-sha "$TANGENT_SHA" \
    --out "$GATE_ROOT"
GATE_STATUS=$?
set -e

cleanup
trap - EXIT INT TERM
exit "$GATE_STATUS"
