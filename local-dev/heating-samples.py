"""Run the downloaded heating samples with fixed six-subtask rules, one MCAP per job."""
import argparse
from datetime import datetime
import getpass
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parent.parent


def select_samples(values, samples):
    """Resolve IDs and inclusive ranges against the current manifest."""
    tokens = re.split(r'[\s,，、]+', ' '.join(values).strip().lower())
    known = {s['sample_id'] for s in samples}
    if tokens == ['all'] or tokens == ['全部']:
        return list(samples)
    wanted = set()
    for token in tokens:
        if re.fullmatch(r'[0-9]+', token):
            wanted.add(token.zfill(3))
        elif re.fullmatch(r'[0-9]+-[0-9]+', token):
            first, last = map(int, token.split('-'))
            if first > last or last - first + 1 > len(known):
                raise ValueError(f'编号范围无效：{token}')
            wanted.update(str(i).zfill(3) for i in range(first, last + 1))
        else:
            raise ValueError('请输入编号（如 1 3 5）、范围（如 1-5）或 all。')
    if wanted - known:
        raise ValueError('未知样本编号：' + ', '.join(sorted(wanted - known)))
    chosen = [s for s in samples if s['sample_id'] in wanted]
    if not chosen:
        raise ValueError('没有选中样本')
    return chosen


def print_samples(samples):
    print('\n编号  录制编号                     时长')
    for sample in samples:
        print(f"{sample['sample_id']}   {sample['episode_id']}   {sample['duration_seconds']:.1f} 秒")


def interactive_options(args, samples):
    print_samples(samples)
    print('\n使用加热工位完整六段模式：H1至H6必须齐全，边界不清保留候选值待复核。')
    print('单条：17；多条：1 3 5（也可用逗号）；连续：1-5；全部：all；返回：q。')
    while True:
        answer = input('想处理哪些条目：').strip()
        if answer.lower() == 'q':
            return False
        try:
            chosen = select_samples([answer], samples)
            break
        except ValueError as exc:
            print(exc)
    args.samples = [s['sample_id'] for s in chosen]
    print(f'已选择 {len(chosen)} 条：' + '、'.join(args.samples))
    print('1. dry run：准备请求，不调用模型、不产生正式标注（默认）')
    print('2. API 标注：调用 DeepSeek，生成标注结果，会产生费用')
    while True:
        mode = input('选择运行方式 [1]，q 返回：').strip().lower()
        if mode == 'q':
            return False
        if mode in ('', '1', '2'):
            args.run_api = mode == '2'
            break
        print('请输入 1 或 2。')
    if args.run_api:
        while True:
            value = input(f'每条预算上限（美元）[{args.cap:g}]，q 返回：').strip().lower()
            if value == 'q':
                return False
            try:
                cap = float(value) if value else args.cap
                if not math.isfinite(cap) or cap <= 0:
                    raise ValueError
                args.cap = cap
                break
            except ValueError:
                print('请输入大于 0 的有限金额，例如 1 或 0.5。')
        print(f'即将正式标注 {len(chosen)} 条，每条预算 ${args.cap:g}，合计 ${args.cap * len(chosen):g}。')
        if input('输入 y 开始，其他输入返回：').strip().lower() != 'y':
            return False
    return True


def main():
    parser = argparse.ArgumentParser(description='20条加热样本：默认仅dry run，--run-api才调用模型。')
    parser.add_argument('--samples', nargs='+', default=['001'], help='样本编号，例如001 002；all表示全部')
    parser.add_argument('--manifest', type=Path,
                        default=ROOT.parent / 'dataset/realman_heating_20_20261021/manifest.json')
    parser.add_argument('--out', type=Path, help='批次输出目录；默认data/review下新建带时间的目录')
    parser.add_argument('--preparation-cache', type=Path, default=ROOT / 'data/cache/prepared_mcap',
                        help='转换缓存目录；不同标注批次共用视频和时间映射')
    parser.add_argument('--no-preparation-cache', action='store_true', help='本次重新转换，不读取或写入共享缓存')
    parser.add_argument('--model', default='deepseek/deepseek-flash', help='本次标注模型（需符合当前DeepSeek接入支持范围）')
    parser.add_argument('--sample-fps', type=float, default=None, help='每秒抽帧数；改变时复用转换视频')
    parser.add_argument('--annotation-spec', choices=('heating-v2', 'heating-v2-complete'),
                        default='heating-v2-complete', help='本次采用的标注规则')
    parser.add_argument('--run-api', action='store_true', help='实际调用DeepSeek；不加时只构建请求')
    parser.add_argument('--cap', type=float, default=1.0, help='每条MCAP的费用上限（美元，默认1）')
    parser.add_argument('--list', action='store_true', help='仅列出样本编号')
    parser.add_argument('--interactive', action='store_true', help='交互选择样本、运行方式和预算')
    args = parser.parse_args()
    if not math.isfinite(args.cap) or args.cap <= 0:
        parser.error('--cap must be finite and positive')
    if args.sample_fps is not None and (not math.isfinite(args.sample_fps) or args.sample_fps <= 0):
        parser.error('--sample-fps must be finite and positive')
    manifest = args.manifest.resolve()
    samples = json.loads(manifest.read_text(encoding='utf-8'))['samples']
    if args.list:
        print_samples(samples)
        return 0
    if args.interactive and not interactive_options(args, samples):
        print('已返回，未开始处理。')
        return 0
    try:
        chosen = select_samples(args.samples, samples)
    except ValueError as exc:
        parser.error(str(exc))
    inputs = []
    for sample in chosen:
        source = (manifest.parent / sample['mcap_file']).resolve()
        if not source.is_relative_to(manifest.parent) or source.suffix.lower() != '.mcap' or not source.is_file():
            parser.error(f"样本{sample['sample_id']}的MCAP不存在或路径无效")
        inputs.append((sample, source))
    mode = 'api' if args.run_api else 'dry'
    output = (args.out or ROOT / 'data/review' / f'heating_v2_{mode}_{datetime.now():%Y%m%d_%H%M%S_%f}').resolve()
    env = dict(os.environ, ARGUS_PROVIDER='deepseek')
    env.pop('OPENAI_API_KEY', None)
    env.pop('OPENROUTER_API_KEYS', None)
    print(f'加热六子任务规范；{len(inputs)}条；模式：{mode}；输出：{output}', flush=True)
    if args.run_api:
        print(f'每条上限 ${args.cap:g}，本批最多 ${args.cap * len(inputs):g}。', flush=True)
        env['DEEPSEEK_API_KEY'] = env.get('DEEPSEEK_API_KEY') or getpass.getpass('DeepSeek API Key（隐藏输入）：')
        if not env['DEEPSEEK_API_KEY'].strip():
            parser.error('未输入Key，未发送请求')
    else:
        env.pop('DEEPSEEK_API_KEY', None)
    output.mkdir(parents=True, exist_ok=True)
    failed = []
    for index, (sample, source) in enumerate(inputs, 1):
        print(f'\n[{index}/{len(inputs)}] 开始处理 {sample["sample_id"]}：{sample["episode_id"]}', flush=True)
        job = output / f"{sample['sample_id']}_{sample['episode_id']}"
        # Give the reader only the MCAP, never the neighboring human annotations or HTML answers.
        cmd = [sys.executable, '-m', 'review', '--data', str(source), '--out', str(job),
               '--rig', 'teleop_arms', '--annotation-spec', args.annotation_spec, '--model', args.model,
               '--cap', str(args.cap), '--jobs', '1', '--concurrency', '1']
        if not args.no_preparation_cache:
            cmd += ['--preparation-cache', str(args.preparation_cache.resolve())]
        if args.sample_fps is not None:
            cmd += ['--sample-fps', str(args.sample_fps)]
        if not args.run_api:
            cmd.append('--free')
        if subprocess.run(cmd, cwd=ROOT, env=env).returncode:
            failed.append(sample['sample_id'])
            for receipt in sorted((job / 'run/out').glob('*.json')):
                try:
                    record = json.loads(receipt.read_text(encoding='utf-8'))
                except (OSError, ValueError):
                    continue
                reason = (record.get('labels') or {}).get('_parse_error') or record.get('no_reply')
                if reason:
                    print(f'样本 {sample["sample_id"]} 失败原因：{str(reason)[:500]}', flush=True)
    print(f'完成，输出：{output}；失败编号：{", ".join(failed) or "无"}', flush=True)
    if args.run_api:
        print('已生成的结果页面可在“查看已有结果”中选择；已有列表输入 r 刷新。')
    else:
        print('dry run 只保存请求准备记录，不产生正式标注，也不会出现在已有结果列表。')
    return int(bool(failed))


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (EOFError, KeyboardInterrupt):
        print('\n已停止；不会开始后续条目，已保存的内容保留。')
        raise SystemExit(130)
