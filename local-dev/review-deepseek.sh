#!/usr/bin/env bash
# Run with bash, not source: the entered key belongs only to this process.
set +x
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/argus-env.sh"
export ARGUS_PROVIDER=deepseek
unset OPENAI_API_KEY OPENROUTER_API_KEYS
_argus_needs_key=1
for _argus_arg in "$@"; do
    case "$_argus_arg" in --free|--help|-h) _argus_needs_key=0 ;; esac
done
if [[ $_argus_needs_key == 1 && -z "${DEEPSEEK_API_KEY:-}" ]]; then
    read -r -s -p "DeepSeek API Key (hidden, not saved): " DEEPSEEK_API_KEY
    printf '\n'
    if [[ -z "$DEEPSEEK_API_KEY" ]]; then
        printf 'No key entered. Nothing was sent.\n' >&2
        exit 2
    fi
    export DEEPSEEK_API_KEY
fi
if [[ $_argus_needs_key == 0 ]]; then
    unset DEEPSEEK_API_KEY
fi
exec uv run --frozen python -m review --model deepseek/deepseek-flash --reasoning high "$@"
