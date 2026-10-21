# Argus 使用 DeepSeek：本地运行说明

适用于本项目的 WSL Ubuntu 环境。

**只想查看以前的结果：双击项目根目录的 [查看已有结果.cmd](../查看已有结果.cmd)。不需要 API Key，不要重新执行下面的标注命令。** 详细操作见 [本机使用说明](../Argus环境使用说明.md)。

下面的步骤用于处理新的数据。

## 1. 你需要准备什么

- 一把 DeepSeek 官方平台的 API Key，以及可用的 API 余额。
- 一段短 MCAP 或视频，用于第一轮测试。
- 不需要 OpenAI API Key，也不需要将你的 Key 发到聊天中。

当前模型是 `deepseek-flash`（官方文档对应 DeepSeek-V4.1-Flash，支持图片）。主标注、字段词典生成、分辨率选择以及证据检查均走 DeepSeek。模型名会在 Argus 内记录为 `deepseek/deepseek-flash`，发送到官方接口时为 `deepseek-flash`。

## 2. 先做免费检查

在 Argus 项目目录的 WSL 终端运行，替换实际文件路径：

```bash
bash local-dev/review-deepseek.sh \
  --data "/mnt/d/data/你的录制.mcap" \
  --rig teleop_arms \
  --dataset-note "Realman双臂遥操录制，请根据实际证据分析操作过程。" \
  --out "./data/review/my_mcap_deepseek_dry_01" \
  --free --jobs 1 --concurrency 1
```

`--free` 表示只检查数据准备和请求构建，不输入 Key，不调用模型，也不产生 AI 标注。Windows 的 `D:\data\你的录制.mcap` 在 WSL 下是 `/mnt/d/data/你的录制.mcap`。

之前 environment-checks 中的 synthetic.mp4 是自动生成的测试图案，不是公司的机器人录制。它适合检查环境，不适合判断 AI 标注效果。

## 3. 正式调用 DeepSeek

建议先用几十秒的短录制；使用新的输出目录：

```bash
bash local-dev/review-deepseek.sh \
  --data "/mnt/d/data/你的录制.mcap" \
  --rig teleop_arms \
  --dataset-note "Realman双臂遥操录制，请根据实际证据分析操作过程。" \
  --out "./data/review/my_mcap_deepseek_live_01" \
  --cap 1 --jobs 1 --concurrency 1
```

终端显示 `DeepSeek API Key (hidden, not saved):` 后，在本地粘贴 Key 并按回车。屏幕不会显示 Key 或星号，这是正常现象。

**让新结果自动出现在查看菜单：** 每次把 `--out` 保存到 `./data/review/新的任务名`。流程生成结果页面后，重新打开 `查看已有结果.cmd`，或在选择菜单输入 `r` 刷新，就能看到新任务，无需手动登记。目录名可以用工位、日期和批次命名；保存到 `data/review/` 之外不会被这个入口自动发现。

脚本通过隐藏输入读取密钥，仅传给当前运行的程序，不把它写进文件、Git 或命令历史。每次新运行可能需要重新输入。请使用 `bash local-dev/review-deepseek.sh`，不要 `source` 此脚本，也不要把 Key 写到命令参数中。

去掉 `--free` 后，会把抽取的图片和相应上下文发送至 DeepSeek，并消耗 API 余额。ChatGPT Pro 订阅不参与这次调用。

`--cap 1` 的单位是美元，不是人民币。本适配按 DeepSeek 官方英文价格表的高峰单价估算：每百万 token 输入未命中 $0.30、缓存命中 $0.006、输出 $1.20（核对日期 2026-10-16）。为避免时段、节假日和跨时段请求的歧义，统一按高峰价格预留。结果中的 `price_basis` / `peak_list_estimate` 标明这是估算，不是实际账单。实际扣款以 DeepSeek 控制台为准；价格变更后应更新代码中的费率。预算预留不足时程序可能停止，不能保证一个预算一定完成整段录制。

## 4. 结果在哪里

上述正式运行的输出根目录：

```text
D:\code\元节智能\ai_Quality_Inspection_Division\argus\data\review\my_mcap_deepseek_live_01
```

`dry/` 保存准备发送的请求；`run/out/` 保存模型阶段的输出或失败记录；其他目录用于预处理、片段及汇总。流程成功不代表 AI 判断一定正确，需要人工查看关键动作及时间边界。

可在已进入 Argus 环境的终端启动结果页面：

```bash
uv run --frozen python -m board serve \
  --board ./data/review/my_mcap_deepseek_live_01 \
  --clips ./data/review/my_mcap_deepseek_live_01/clips
```

切换模型、提示词或数据后使用新的 `--out` 目录，避免混用旧的请求和计费记录。原有脚本 `local-dev/check-environment.sh` 仍然只是环境检查；DeepSeek 专用入口是本页的 `review-deepseek.sh`。

## 5. 配置原理（不必先学会才能使用）

本次使用环境变量和启动脚本，没有新增 YAML 读取器。脚本自动设置：

| 配置 | 值 |
| --- | --- |
| 服务商选择 | `ARGUS_PROVIDER=deepseek` |
| 官方接口 | `https://api.deepseek.com/chat/completions` |
| 主模型与辅助模型 | `deepseek-flash` |
| Key 来源 | 本地隐藏输入后传入 `DEEPSEEK_API_KEY` |
| 主标注思考强度 | `high`，可通过 `--reasoning low` 或 `--reasoning none` 调整 |

辅助词典和分辨率选择保持 `low`。API Key 不放在 YAML 中。普通入口未设置 `ARGUS_PROVIDER` 时仍保留原来的 OpenAI/OpenRouter 选择方式；只填写 DeepSeek Key 不会自动切换服务商。

当前接入遵循 DeepSeek 的 JSON 输出、图片输入和请求体限制。超出 48 MiB 的请求在发送前拒绝，需要缩短输入或调整采样。这里只接入 Flash 视觉模型，未接入不支持图片的 V4-Pro。

## 6. 当前验证边界

本次接入不需要真实 Key 即可做离线验证。未使用你的账号调用 API，因此账号权限、真实扣费、网络连通性及机器人录制的标注质量仍需首次短样本运行确认。

代码位于本地 `platform` 分支工作区；没有自动提交、推送或修改 upstream/main，也没有修改公司的 data_label_platform 项目。此前的独立 `python -m label` 命令仍保留上游要求的干净 Git 提交检查，本页使用的是 `python -m review` 流程。

参考：

- [DeepSeek 首次调用 API](https://api-docs.deepseek.com/zh-cn/)
- [DeepSeek 图像理解](https://api-docs.deepseek.com/zh-cn/guides/vision/)
- [DeepSeek 美元价格表](https://api-docs.deepseek.com/quick_start/pricing/)
