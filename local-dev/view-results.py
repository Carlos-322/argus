"""Open an existing local board without running preparation or annotation."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parent.parent


def heating_result_info(path):
    """Identify a saved sample and its batch time without modifying the result."""
    sample = re.fullmatch(r'(\d{3})_(\d{8}_\d{6}_\d+)', path.name)
    batch = re.fullmatch(r'heating_v2_api_(\d{8}_\d{6}_\d+)', path.parent.name)
    if not sample or not batch:
        return None
    try:
        created = datetime.strptime(batch[1], '%Y%m%d_%H%M%S_%f')
    except ValueError:
        return None
    return sample[1], sample[2], created


def result_titles(results):
    """Number saved results per recording, not by menu position or rebuild time."""
    titles, groups = {}, {}
    for path in results:
        info = heating_result_info(path)
        if info:
            sample, recording, created = info
            groups.setdefault(recording, []).append((created, str(path), sample, path))
        else:
            titles[path] = ('炒菜任务' if path.name == 'realman_cooking_20261009_180214_deepseek_live' else
                            'rollout 任务' if path.name == 'realman_rollout_01_deepseek_live' else path.name)
    for rows in groups.values():
        for number, (created, _, sample, path) in enumerate(sorted(rows), 1):
            try:
                policy = json.loads((path / 'annotation_policy.json').read_text(encoding='utf-8'))
                policies = [policy] if 'annotation_spec' in policy else list(policy.values())
                specs = {(p.get('annotation_spec') or {}).get('id') for p in policies if isinstance(p, dict)}
                spec = next(iter(specs)) if len(specs) == 1 else None
            except (OSError, ValueError, AttributeError):
                spec = None
            rule = {'heating-v2': '旧规则', 'heating-v2-complete': '完整六段'}.get(spec)
            titles[path] = f'加热 {sample}｜第 {number} 次标注｜{created:%Y-%m-%d %H:%M:%S}'
            if rule:
                titles[path] += f'｜{rule}'
    return titles


def discover_results():
    """Read the output folders again on every refresh, including grouped jobs."""
    results = []
    for directory, folders, files in os.walk(ROOT / "data" / "review"):
        path = Path(directory)
        if "manifest.json" in files and "clips" in folders and "qa" in folders:
            if any((path / "qa").glob("*.json")):
                results.append(path)
            # These are job internals, not more independent review results.
            folders[:] = []
    return sorted(results, key=lambda p: ((p / "manifest.json").stat().st_mtime, str(p)), reverse=True)


def show_results(results):
    print("\n查看已有结果（不调用模型，不需要 API Key）\n", flush=True)
    print("已重新读取 data/review，最近生成的结果排在前面。", flush=True)
    titles = result_titles(results)
    for i, path in enumerate(results, 1):
        name = titles[path]
        print(f"  {i}. {name}\n     {path.relative_to(ROOT)}", flush=True)
    if not results:
        print("暂时没有可查看的结果。新标注生成网页数据后，输入 r 刷新即可。")
    print("\n新标注请保存到 data/review/任务名；只有 dry/ 的免费检查不会列入。", flush=True)


def main():
    parser = argparse.ArgumentParser(description="查看已有结果，不运行标注。")
    parser.add_argument("--list", action="store_true", help="只列出可以查看的结果")
    args = parser.parse_args()
    results = discover_results()
    show_results(results)
    if args.list:
        return 0
    while True:
        choice = input("\n输入编号查看，r 刷新列表，q 退出：").strip()
        if choice.lower() == "q":
            return 0
        if choice.lower() == "r":
            results = discover_results()
            show_results(results)
            continue
        if choice.isdecimal() and 1 <= int(choice) <= len(results):
            selected = results[int(choice) - 1]
            break
        print("请输入列表中的数字，或 r 刷新、q 退出。")
    # Leave any other running viewers alone and choose an unused local port.
    for port in range(8896, 8996):
        with socket.socket() as probe:
            try:
                probe.bind(("127.0.0.1", port))
            except OSError:
                continue
            break
    else:
        print("8896–8995 端口都被占用，请关闭不用的结果窗口后重试。")
        return 1
    url = f"http://localhost:{port}"
    server = subprocess.Popen([sys.executable, "-m", "board", "serve", "--board", str(selected),
                               "--clips", str(selected / "clips"), "--port", str(port),
                               "--title", result_titles(results)[selected]], cwd=ROOT,
                              start_new_session=True)
    try:
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        for _ in range(50):
            if server.poll() is not None:
                print("网页服务启动失败，请查看上方提示。")
                return 1
            try:
                with opener.open(f"http://127.0.0.1:{port}/", timeout=1):
                    break
            except (OSError, urllib.error.URLError):
                time.sleep(0.2)
        else:
            print("网页服务未及时就绪，请重新打开此入口。")
            return 1
        print(f"\n查看地址：{url}\n请保持此窗口打开；按 Ctrl+C 停止查看。\n", flush=True)
        try:
            subprocess.run(["/mnt/c/Windows/System32/cmd.exe", "/c", "start", "", url],
                           cwd="/mnt/c/Windows", stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                           timeout=10, check=True)
        except (OSError, subprocess.SubprocessError):
            print(f"浏览器未自动打开，请手动访问：{url}", flush=True)
        return server.wait()
    finally:
        if server.poll() is None:
            server.terminate()
            try:
                server.wait(timeout=5)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (KeyboardInterrupt, EOFError):
        print("\n已停止查看。")
