# Realman rollout 首次运行

## 已整理的数据

原位置：`D:\Downloads\uploads_realman_rollout_01_20261015_165739_262467_rollout.mcap`

已移动至项目内：`data/inputs/realman/uploads_realman_rollout_01_20261015_165739_262467_rollout.mcap`

文件大小：440700459 字节（约 420 MiB）。移动前后 SHA-256 一致：

```text
4094EDEE2203586C6BE9B65AF20ED5E81B46D2E16F65C49D3D9F2B1FFFB0D3B4
```

数据放在 Git 忽略的 data 目录，不会随普通代码提交上传。

## 当前检查结果

2026-10-16 已执行 `bash local-dev/review-realman-rollout.sh --free`，正常退出。

- 识别为 MCAP，时长约 227.85 秒。
- 识别到 3 路视频：exo（left_head_wide_cam_compressed）、left、right。
- 数据转换、字段词典的免费步骤、采集检查、传感器检查、视频预览和请求构建均执行完成。
- 请求模型：`deepseek/deepseek-flash`，思考强度 `high`。
- 未使用 API Key，未调用模型，尚无 AI 标注结果。
- 限制：report.json 中 state_kind 为 none，运动通道 `/inference_control/handover_chunks` 没有匹配上游标准的每臂六关节加夹爪格式；标准关节运动检查没有运行。这不代表原始数据没有运动记录。当前可基于视频和已保留证据继续标注。

免费检查输出：`data/review/realman_rollout_01_deepseek_dry/`

## 下一步：正式调用

在 Argus 项目目录的 WSL 终端运行：

```bash
bash local-dev/review-realman-rollout.sh
```

文件路径、模型和输出位置均已填好。不需要复制长路径。

出现 `DeepSeek API Key (hidden, not saved):` 后，在自己的终端粘贴 Key 并按回车。输入不会显示，也不会保存到文件。这次会发送图片和上下文到 DeepSeek 并消耗 API 余额。

脚本默认预算为 1 美元（保守估算），并发为 1。正式结果保存到：

```text
data/review/realman_rollout_01_deepseek_live/
```

需要再次做免费检查时运行：

```bash
bash local-dev/review-realman-rollout.sh --free
```

两种模式的输出目录分开，避免将免费准备结果误认为正式标注结果。预算不足时不要直接删除旧的费用记录，请先查看运行输出。更详细的配置说明见 [DeepSeek使用说明.md](DeepSeek使用说明.md)。