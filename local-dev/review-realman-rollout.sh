#!/usr/bin/env bash
# This recording only. Add --free for preparation without any model call.
set +x
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/argus-env.sh"
input="$PWD/data/inputs/realman/uploads_realman_rollout_01_20261015_165739_262467_rollout.mcap"
output="$PWD/data/review/realman_rollout_01_deepseek_live"
for arg in "$@"; do
    if [[ "$arg" == --free ]]; then
        output="$PWD/data/review/realman_rollout_01_deepseek_dry"
    fi
done
if [[ ! -f "$input" ]]; then
    printf 'Input MCAP not found: %s\n' "$input" >&2
    exit 2
fi
printf 'Input: %s\nOutput: %s\n' "$input" "$output"
exec bash local-dev/review-deepseek.sh \
  --data "$input" \
  --rig teleop_arms \
  --dataset-note 'Realman dual-arm teleoperation rollout. Infer actions from recorded evidence; task success is not assumed.' \
  --out "$output" \
  --cap 1 --jobs 1 --concurrency 1 "$@"
