#!/usr/bin/env bash
# Cooking recording. Add --free to prepare without model calls.
set +x
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/argus-env.sh"
input="$PWD/data/inputs/realman/uploads_realman01_20261009_180214_697738_Kitchen_Cooking_the_vegetables_20261009_180214_48_b0_2d_dd_ec_4e_0.mcap"
output="$PWD/data/review/realman_cooking_20261009_180214_deepseek_live"
for arg in "$@"; do
    if [[ "$arg" == --free ]]; then
        output="$PWD/data/review/realman_cooking_20261009_180214_deepseek_dry"
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
  --dataset-note 'Realman dual-arm recording. The filename identifies a kitchen cooking-vegetables task; verify actual actions and completion from recorded evidence rather than assuming success from the filename.' \
  --out "$output" \
  --cap 1 --jobs 1 --concurrency 1 "$@"
