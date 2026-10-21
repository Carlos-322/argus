#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/argus-env.sh"
unset DEEPSEEK_API_KEY OPENAI_API_KEY OPENROUTER_API_KEYS
exec python local-dev/view-results.py "$@"
