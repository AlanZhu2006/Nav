#!/usr/bin/env bash
# Continue the local pairwise-only comparison after the official checkpoint download.
set -euo pipefail
task_root=${1:?usage: run_raw_match_verification_local.sh ABSOLUTE_RUN_ROOT}
task_python="$task_root/venv/bin/python"
task_checkpoint="$task_root/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth"
task_repository=/home/asus/Research/Nav-graph-blind/.diagnostics/dependencies/MASt3R_official_20260910
cd /home/asus/Research/Nav-graph-blind

# Content-Length from the official checkpoint endpoint. Never load a partial download.
task_expected_bytes=2754910614
task_deadline=$((SECONDS + 1800))
while [ "$(stat -c %s "$task_checkpoint")" -ne "$task_expected_bytes" ]; do
    if [ "$SECONDS" -ge "$task_deadline" ]; then
        echo 'STOP: checkpoint download is incomplete; no MASt3R result.'
        exit 1
    fi
    sleep 10
done
echo 'CHECKPOINT_COMPLETE: starting isolated two-pair smoke'
"$task_python" -m MemNavData.run_raw_match_verification run \
    --inputs "$task_root/inputs" --out "$task_root/mast3r_smoke_v1" \
    --matcher mast3r --repository "$task_repository" --checkpoint "$task_checkpoint" --limit 2
echo 'SMOKE_COMPLETE: starting all 159 fixed pairs, no navigation'
"$task_python" -m MemNavData.run_raw_match_verification run \
    --inputs "$task_root/inputs" --out "$task_root/mast3r_v1" \
    --matcher mast3r --repository "$task_repository" --checkpoint "$task_checkpoint"
"$task_python" -m MemNavData.score_raw_match_verification \
    --inputs "$task_root/inputs" \
    --results "$task_root/lightglue_v1" "$task_root/mast3r_v1" \
    --out "$task_root/comparison_v1.json"
echo 'COMPARISON_COMPLETE: matching evidence only, no new navigation SR'
