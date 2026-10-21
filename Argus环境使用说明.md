# Argus 本机使用说明

## 1. 只查看过去的结果：双击这个文件

在项目根目录双击 **[查看已有结果.cmd](查看已有结果.cmd)**。

1. 窗口会列出已经可以查看的结果，例如“炒菜任务”和“rollout 任务”。
2. 输入前面的数字，按回车，浏览器会自动打开对应的结果网页。
3. 查看期间保持终端窗口打开。看完后，在终端按 `Ctrl+C` 停止服务。

**这个入口只读取已有标注和视频，不重新分段、不重新标注、不调用模型，也不需要 API Key。**

如果浏览器没有自动打开，把窗口显示的 `http://localhost:端口号` 复制到浏览器。重启电脑后，需要重新双击入口；旧网址本身不会启动服务。已有端口被占用时，入口会自动换一个端口，以窗口显示的地址为准。

完整项目位置：

```text
D:\code\元节智能\ai_Quality_Inspection_Division\argus
```

## 2. 已有结果保存在哪里

以下目录都在项目的 `data/review/` 下：

| 结果 | 目录名 |
| --- | --- |
| 炒菜任务 | `realman_cooking_20261009_180214_deepseek_live` |
| rollout 任务 | `realman_rollout_01_deepseek_live` |

网页需要同时读取标注数据和视频，并不是双击一个 HTML 文件就能完整查看。请保留整个结果目录，尤其是 `manifest.json`、`qa/` 和 `clips/`。

### 新的 API 标注结果会自动出现在列表里吗？

**会。列表每次打开都重新读取文件夹，不是写死的两条记录，不需要手动登记。**

以后正式标注时，把 `--out` 指向 `data/review/` 下面的新目录，例如：

```bash
--out ./data/review/加热工位_20261022_第1次
```

标注流程生成可查看的网页数据后：

- 再次双击 `查看已有结果.cmd`，新结果会自动列出，最近生成的排在前面。
- 如果已经停留在选择菜单，输入 **`r`** 并回车，就会重新扫描并显示新结果。
- 也可以按日期或工位分子文件夹，例如 `data/review/加热工位/第1次/`，入口会一起查找。
- 新结果默认显示输出文件夹的名字，所以建议目录名包含工位、日期或批次，方便选择。

自动查找范围是本项目的 `data/review/`。保存到其他位置的结果不会自动列出。只有 `dry/` 的免费检查，以及还没有生成网页数据的任务，也不会列入。列表表示“有结果页面可看”，不代表所有片段都已成功标注，具体状态仍要看页面。

## 3. 几个入口有什么区别

| 操作 | 会做什么 | 会不会调用模型 |
| --- | --- | --- |
| 双击 `查看已有结果.cmd` | 选择并查看已有结果 | 不会 |
| 双击 `启动Argus.cmd` | 打开运行环境，等待你输入命令 | 不会 |
| 执行 `review-deepseek.sh ... --free` | 读取数据、检查并构建请求 | 不会，也不生成正式标注 |
| 执行 `review-deepseek.sh ...`，不加 `--free` | 运行包含模型标注的流程 | 会，需要 Key 和余额 |

**双击 CMD 不会自动把文件夹里的所有文件重新标注。** 处理范围取决于正式运行命令的 `--data`：指定单个文件就处理该文件；指定文件夹才按文件夹内容读取。现有炒菜和 rollout 脚本各自指定一个 MCAP。

重新执行标注流程时，程序会检查旧结果是否可复用；不能把它当作单纯的查看入口。只想看结果时，用第一节的入口即可。

## 4. 想开始新的标注

先双击 `启动Argus.cmd`，再按 **[DeepSeek 使用说明](local-dev/DeepSeek使用说明.md)** 操作。那里有免费检查和正式标注的完整命令。

新任务使用新的 `--out` 目录。不要覆盖需要保留的旧结果。

## 5. 不用双击入口，怎样手动打开网页

双击 `启动Argus.cmd` 后，粘贴下面一条命令即可查看炒菜结果：

```bash
python -m board serve --board data/review/realman_cooking_20261009_180214_deepseek_live --clips data/review/realman_cooking_20261009_180214_deepseek_live/clips --port 8896
```

然后在浏览器打开 <http://localhost:8896>。保持终端打开，按 `Ctrl+C` 停止。

查看 rollout 结果则使用：

```bash
python -m board serve --board data/review/realman_rollout_01_deepseek_live --clips data/review/realman_rollout_01_deepseek_live/clips --port 8897
```

对应地址是 <http://localhost:8897>。若提示端口被占用，可以改 `--port`，或直接使用第一节的选择入口。

## 6. 环境维护时才需要看

程序在 WSL Ubuntu-22.04 中运行；源码和数据仍在 Windows 的项目目录。`D:\` 在 WSL 中对应 `/mnt/d/`。

- 环境配置：[local-dev/argus-env.sh](local-dev/argus-env.sh)。
- Linux Python 环境：`/mnt/d/applications/uv/environments/argus-linux`。项目里的 Windows `.venv` 不用于完整运行。
- 在已有 WSL 终端进入项目后，用 `source local-dev/argus-env.sh` 加载环境。
- 需要同步依赖时，在该环境执行 `uv sync --frozen`。
- `bash local-dev/check-environment.sh` 会生成测试视频并运行免费检查；不是查看旧结果的命令。
- `data/` 不进入 Git。备份源码不会自动备份原始数据和标注结果。
