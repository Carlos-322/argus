#!/usr/bin/env bash
set -uo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/argus-env.sh" || exit 1

while true; do
    printf '\n========== Argus ==========\n'
    printf '1. 选择加热数据，开始 dry run 或 API 标注\n'
    printf '2. 查看已有结果\n'
    printf '3. 进入命令行（手动输入命令）\n'
    printf '0. 退出\n'
    read -r -p '请输入选项：' choice || exit 0
    case "$choice" in
        1) python local-dev/heating-samples.py --interactive ;;
        2) bash local-dev/view-results.sh ;;
        3) exec bash --rcfile ./local-dev/argus-env.sh -i ;;
        0|q|Q) exit 0 ;;
        *) printf '请输入 1、2、3 或 0。\n' ;;
    esac
done
