# Argus 本机开发与运行指南

配置日期：2026-10-16。整理目标：以本目录作为 Argus 的统一开发工作区。

## 1. 日常从哪里开始

工作区：

```text
D:\code\元节智能\ai_Quality_Inspection_Division\argus
```

双击项目根目录的 [启动Argus.cmd](启动Argus.cmd)。它会进入已有的 WSL2 Ubuntu-22.04，激活专用 Linux 环境，并将当前目录切到本项目。

在该终端中执行：

```bash
pwd
python --version
uv run --frozen python -m review --help
```

`pwd` 应显示 `/mnt/d/code/元节智能/ai_Quality_Inspection_Division/argus`，Python 应为 3.11.15。

项目的 [README.md](README.md) 是上游功能与安装文档；本文是本机路径和操作指南，两份文档用途不同。

## 2. 源码在 D 盘，程序在 Linux 中运行

本地 Argus 的 `label/route.py` 导入 Unix 的 `fcntl` 模块，完整运行需要 Linux 环境。本机使用已有 WSL2 Ubuntu，没有重新安装 Ubuntu。

Windows 的 `D:\...` 在 WSL 中对应 `/mnt/d/...`，两者访问同一批文件。

本项目中的 `.venv` 是此前创建的 Windows 环境，保留但不用于完整运行 Argus。当前专用 Linux 环境放在项目外部；源码和虚拟环境不要求在同一目录。

## 3. 工作区如何组织

```text
argus/
├─ README.md                  上游文档
├─ Argus环境使用说明.md        本机开发与运行指南
├─ 启动Argus.cmd               日常启动入口
├─ local-dev/
│  ├─ argus-env.sh             本机环境配置
│  └─ check-environment.sh     合成视频免费检查脚本
├─ data/
│  ├─ input/                  建议存放测试用 MCAP 等输入
│  ├─ review/                 建议存放正式分析结果
│  └─ environment-checks/     检查脚本生成的测试数据与结果
├─ prepare/、label/、board/    项目源码
└─ .venv/                     已保留的 Windows 环境
```

`data` 中的建议目录按实际需要建立。项目已有规则忽略 `data/`，原始数据和模型输出不应当作为源码提交。本文和启动脚本是本机定制文件；本次没有提交或推送 Git。

若使用 VS Code 开发，建议通过 WSL 打开本工作区，选择解释器：
`/mnt/d/applications/uv/environments/argus-linux/bin/python`。
新终端运行前可执行：

```bash
source local-dev/argus-env.sh
```

只选择编辑器解释器并不保证终端已获得全部环境变量；启动入口或上面的 source 命令会同时配置 uv、虚拟环境和 ffmpeg。

## 4. 通用工具与环境仍放在哪里

| 内容                            | Windows 路径                                                |
| ------------------------------- | ----------------------------------------------------------- |
| Windows uv 0.13.1               | `D:\applications\uv\uv.exe`                               |
| Linux uv 0.13.1                 | `D:\applications\uv\linux\uv`                             |
| Linux Python 3.11.15            | `D:\applications\uv\python-linux`                         |
| Argus 实际使用的 Linux 虚拟环境 | `D:\applications\uv\environments\argus-linux`             |
| Linux 下载缓存                  | `D:\applications\uv\cache-linux`                          |
| Linux ffmpeg/ffprobe            | `D:\applications\uv\tools\ffmpeg-linux\bin`               |
| 已保留的 Windows Python／缓存   | `D:\applications\uv\python`、`D:\applications\uv\cache` |

保留这些目录可以避免重新下载以及虚拟环境路径失效。它们提供运行能力，开发、查看文档和管理本项目数据都从 Argus 工作区开始。

环境配置的唯一维护位置是 [local-dev/argus-env.sh](local-dev/argus-env.sh)。它设置 Python、缓存、专用虚拟环境及工具搜索路径，并进入项目目录。项目根目录的启动入口负责加载它。

uv 目录中的旧启动入口和旧环境脚本只是兼容转接，避免已有使用方式失效；旧说明文档也仅指向本文。无需在两处维护相同内容。

## 5. 后续安装依赖与检查

在项目启动入口打开的终端中同步已有锁定依赖：

```bash
uv sync --frozen
```

查看实际解释器：

```bash
python -c "import sys; print(sys.executable)"
```

需要重新验证免费处理链路时：

```bash
bash local-dev/check-environment.sh
```

该脚本创建一段三秒合成视频，再运行免费预处理；不读取公司数据，不调用模型。每次结果保存在 `data/environment-checks` 下的独立目录，避免覆盖旧结果。

最初环境安装时的历史验证材料仍在：
`D:\applications\uv\checks\review-smoke-20261016`。
其中的记录含原有路径，因此本次没有直接搬动。之后的验证统一写入项目 `data` 目录。

## 6. 开始处理 MCAP 前

先使用 `--free` 检查实际相机、时间轴和数据字段，再按需要配置模型 API。输入路径使用 Linux 形式，例如 `D:\data\example.mcap` 对应 `/mnt/d/data/example.mcap`。

已有验证：Linux Python 3.11.15、35 个依赖的一致性检查、查看器入口、ffmpeg/ffprobe 及三秒合成视频免费流程通过。尚未据此证明真实 MCAP 自定义字段兼容或 API 可用。

API 密钥只在使用时单独配置，不写入本机指南、启动脚本或源码。WSL 若显示 localhost 代理未映射提示，应单独检查 API 连通性；本次整理不修改代理或 Ubuntu 全局配置。
