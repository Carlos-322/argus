#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/argus-env.sh"
exec bash local-dev/review-deepseek.sh --rig teleop_arms --annotation-spec heating-v2-complete --preparation-cache data/cache/prepared_mcap "$@"
