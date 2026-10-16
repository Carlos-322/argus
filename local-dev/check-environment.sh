#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/argus-env.sh"

mkdir -p "$PWD/data/environment-checks"
argus_check_dir="$(mktemp -d "$PWD/data/environment-checks/check-$(date +%Y%m%d-%H%M%S)-XXXXXX")"

ffmpeg -hide_banner -loglevel error -f lavfi -i testsrc2=size=320x240:rate=30 -t 3 -c:v libx264 -pix_fmt yuv420p "$argus_check_dir/synthetic.mp4"

uv run --frozen python -m review --data "$argus_check_dir/synthetic.mp4" --rig teleop_arms --dataset-note 'Synthetic environment check; no robot task.' --out "$argus_check_dir/review" --free --jobs 1 --concurrency 1

printf '\nEnvironment check saved to: %s\n' "$argus_check_dir"
