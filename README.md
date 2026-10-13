# Argus

[![Python 3.11](https://img.shields.io/badge/python-3.11-455468)](pyproject.toml)
[![Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-455468)](LICENSE)

### Update v2: Midcentury on WGO-Bench

Recently, [Midcentury AI](https://www.midcentury.xyz/blog/optimizing-argus) pushed an adaptation of Argus to SOTA (**50.2% semantic F1**) on WGO-Bench. However, this uses **Sol 6.1 High**, meaning comparisons with published results from older models do not isolate the harness's contribution.

### Update v1: Format and sensor support

This release extends annotation to depth, tactile and other sensor data, and consolidates teleop, UMI and human ego under one harness. Format readers preserve source fields and timestamps; long recordings are annotated in chunks and rejoined on the original timeline. Behavioral and sensor annotations are available in both the viewer and JSON exports.

## Overview

Argus is a multimodal annotation harness for robot learning data. The VLM jointly interprets multi-view RGB, proprioception, actions, depth, tactile and other episode data to produce dense annotations of **actions, subgoals, outcomes, operator mistakes and recoveries**. A shared pipeline handles ingestion, frame sampling and sensor inspection across teleop, UMI and human ego datasets, preserving source timing and measurement semantics without dataset-specific prompts. We believe this combination of broad format support and joint visual/sensor reasoning distinguishes Argus among open-source robotics annotation tools.

<p align="center">
  <img src="media/architecture.svg" width="720" alt="Annotation pipeline: format ingestion, field interpretation, visual and sensor evidence selection, VLM annotation, temporal stitching, viewer and JSON exports.">
</p>

**Data:** teleop, UMI and human ego. **Formats:** LeRobot v2.0/v2.1/v3.0, MCAP, HDF5, video files and folders, and ZIP/TAR archives, with JSON/CSV/Parquet sidecars. Nine public dataset adapters provide download and sample-selection entrypoints.

| Available evidence | How it contributes |
|---|---|
| RGB | Timestamped multi-view frames provide visual context for task execution, object interactions, progress, failures and recoveries. |
| Proprioception and actions | Joint states, end-effector poses, gripper states and action trajectories provide motion context. Training targets retain their declared semantics rather than being assumed to be executed commands. |
| Depth | Metric depth supports camera-relative distance and separation when units, calibration and temporal alignment are known. Relative depth remains relative. |
| Tactile | Pressure, normal/shear force, contact arrays and optical tactile images support analysis of grip, contact distribution and surface motion. Raw sensor response alone does not establish physical force or slip. |
| Other episode fields | Numeric channels, event flags, waveform packets and structured metadata are accessible through the same inspection interface. Spectral analysis of eligible waveforms requires a recorded sample rate. |
| Task context | Instructions, annotations and calibration metadata provide context. The VLM checks publisher claims against the recording rather than assuming their correctness. |

## Install and review

You need Python 3.11, [uv](https://docs.astral.sh/uv/), Node.js for viewer checks, and `ffmpeg`/`ffprobe` 5.1 or newer.

```bash
git clone https://github.com/Pantheon-Industries-Inc/argus.git
cd argus
uv sync --frozen
```

Paid annotation requires `OPENROUTER_API_KEYS` (comma-separated keys) or `OPENAI_API_KEY` in your shell or secret manager. Gated Hugging Face datasets require accepting their terms and setting `HF_TOKEN`. Keep inputs and outputs under the ignored `data/` directory.

```bash
# Prepare and inspect base requests without model calls.
uv run python -m review --data path/to/dataset --rig teleop_arms \
  --dataset-note "Two arms arrange blocks on a tray." --out data/review/mine --free

# Annotate the same job under a spending cap, then open its board.
uv run python -m review --data path/to/dataset --rig teleop_arms \
  --dataset-note "Two arms arrange blocks on a tray." --out data/review/mine --cap 20
uv run python -m board serve --board data/review/mine --clips data/review/mine/clips
```

`--data` accepts a folder, file, archive or HTTP(S) URL. Set `--rig` to `teleop_arms`, `handheld_gripper` or `ego_head` for teleop, UMI or human ego, respectively. Argus detects the format and available evidence; RGB-only data follows the same pipeline without requiring sensor inputs. [Data Review](https://data.pantheon.inc/review) uses the same default policy, though its deployed version and upload limits can differ.

| Default or optional setting | Behavior |
|---|---|
| Astra, the default | `openai/gpt-6-astra`, medium reasoning; a regular sample every 1.5s for teleop, 1s for UMI and 0.5s for human ego, plus eligible boundaries and first/last frames. |
| Sol 6.1 High | `--model openai/gpt-6.1-sol --reasoning high`; defaults to 1 FPS on all recording setups and requires at least 1 FPS. |
| Faster actions | `--sample-fps 2` is recommended for subsecond fumbles and rapid recoveries. More images shorten the available request window and can increase cost. |
| Task scope | Episode/session scope follows source metadata; `--task-mode episode` or `session` overrides it. A missing instruction does not automatically imply a session. Human ego uses the activities schema. |

Each recording produces one public episode. Long recordings are split to fit API request limits and rejoined on the original timeline; internal chunk boundaries are not treated as recording truncation. Session mode annotates activities within the recording. Preparation limits, missing chunks and processing failures are reported alongside any usable results.

Resume preserves the original settings, input verification and evidence selection. Changes to paid request semantics require a new run. `--free` skips paid field interpretation, image-detail routing and evidence selection, so it previews base requests rather than the full paid evidence path. Spending caps account for all applicable calls and outstanding reservations; unresolved charges stay reserved.

## Results and further use

The output schema includes action phases (advancing, wasteful or idle), subgoals, key events, outcomes, operator mistakes, recoveries, scene changes and recording quality. Sensor findings include evidence, confidence and limitations; inspection can yield no additional finding when the signal is inconclusive.

A review job contains prepared `episodes/`, base requests in `dry/`, receipts in `run/`, rejoined labels in `run_final/`, and the board. Episode JSON, filtered JSON Lines and static exports retain annotations and supporting evidence while omitting internal access metadata. Human field overrides take precedence, including a blank role; source or interpretation changes invalidate stale findings.

Live and static boards use the same [viewer](board/index.html), with behavioral and sensor annotations overlaid on the footage and supporting measurements available alongside it. Shared footage remains subject to its source license and access policy.

![Two-camera UMI annotation walkthrough, with action phases, subgoals and goal reasoning over the footage.](media/board_demo.webp)

Animated WebP from the standalone demonstration. [Browse example annotations](https://pantheon.inc/data-board).

<details>
<summary>Individual stages, dataset adapters and custom boards</summary>

```bash
uv run python -m prepare folder prepare --root path/to/data --rig teleop_arms --out data/episodes/mine/all
uv run python -m checks data/episodes/mine/all
uv run python -m label --dataset mine --episodes data/episodes/mine/all --kind dry
uv run python -m label --dataset mine --episodes data/episodes/mine/all --kind full --cap 20
```

Public adapters are `molmo`, `abc130k`, `galaxea`, `habit`, `fastumi`, `realomin`, `egocentric100k`, `genhumanego` and `openaoe`; `configs/slices/` records published selections. Generic readers preserve source fields instead of applying corpus-release cropping or task selection.

For a custom board, create `data/boards/mine/manifest.json` with the dataset, prepared episodes and run paths. [configs/quickstart/board.json](configs/quickstart/board.json) shows the structure with published-dataset severity rules. Use `python -m board build`, `clips`, `serve`, and `static media`/`static site`; each takes `--help`. Refresh the board after dictionary edits. Owner editing requires `board serve --dictionary-job JOB`.

Paid staged runs require a clean tracked checkout. `--resume RUN --why "reason"` uses saved settings and remaining budget; `python -m label.reparse RUN` reparses stored replies without another call. Receipts survive failures. `board grids` reconstructs base request grids only when original input hashes can be verified; detail/sensor images are counted but not reconstructed there.

`checks.timebase scan` proposes clock corrections; inspect them before `apply`. Capture, source and sensor check failures are distinct from clean results. Optional hand pose uses `board/hand_pose/modal_app.py` and `board/hands.py`; it is separate from ordinary annotation and has the licenses below.

</details>

<details>
<summary>Model comparisons, gate cases and offline verification</summary>

```bash
uv run python -m compare prepare --selection configs/compare/main.json
uv run python -m compare label --selection configs/compare/main.json --models astra,sol61_high --kind full --cap 20
uv run python -m compare board --entries data/runs/compare/main.json --out data/boards/compare
uv run python -m gate prepare
uv run python -m gate label --kind full --cap 20
```

Comparison caps apply per model run and gate caps per recording setup. Sampling may differ by model: compare saved requests before claiming identical inputs. `--with-example` supplies a reference annotation; `--routes` reuses resolution choices. [gate/selection.json](gate/selection.json) and [gate/cases.json](gate/cases.json) define the selected cases; `gate score` evaluates the supplied runs.

`uv run --frozen --offline pytest -q -ra` runs offline tests. CI uses the locked environment, Node, ffmpeg and ffprobe without provider credentials or dataset access. Secret scans cover the committed tree and reachable history; they do not grant permission to publish uploaded data.

</details>

## Credits

The capture checks in `checks/vendor/public_dataset_adapter_qc.py` are Sambhav Gupta's, from public-dataset-adapter (commit 99d0a9e), with the changes listed in that file's header. The hand pose model is ACE-Ego-Hand (Yufei Liu et al., [arXiv:2608.20308](https://arxiv.org/abs/2608.20308)), built on Wan2.2, VideoX-Fun and MANO (Romero, Tzionas and Black, 2017).

## Licenses

The code is Apache-2.0 (`LICENSE`), and the few files that carry material under another license are listed in `THIRD_PARTY_NOTICES.txt`. The components below are downloaded at run time and are not included here. The labels Pantheon publishes, on the data dashboard, in its downloads and in the release, are licensed [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), so anyone may use them for any purpose with credit to Pantheon. The footage they describe keeps its dataset's license.

| Component | License | Source |
|---|---|---|
| ACE-Ego-Hand code (commit 9757868) | MIT | [github.com/ggxxii/ACE-Ego-Hand](https://github.com/ggxxii/ACE-Ego-Hand/blob/97578680931b3f1c8111396c100d595b98857fb8/LICENSE) |
| ACE-Ego-Hand checkpoints (`ace_ego_hand_k.pt`, `ace_ego_hand_kfree.pt`) | CC BY-NC 4.0, non-commercial | [huggingface.co/acerobotics2025/ACE-Ego-Hand](https://huggingface.co/acerobotics2025/ACE-Ego-Hand) |
| Wan2.2-Fun-5B-Control (weights, VAE, umT5-XXL text encoder and tokenizer) | Apache-2.0 | [huggingface.co/alibaba-pai/Wan2.2-Fun-5B-Control](https://huggingface.co/alibaba-pai/Wan2.2-Fun-5B-Control) |
| VideoX-Fun (commit 968f0e2) | Apache-2.0 | [github.com/aigc-apps/VideoX-Fun](https://github.com/aigc-apps/VideoX-Fun) |
| smplx 0.1.28 | Max Planck license for non-commercial scientific research | [github.com/vchoutas/smplx](https://github.com/vchoutas/smplx/blob/main/LICENSE) |
| MANO (you register and download it yourself) | Max Planck license for non-commercial scientific research, no redistribution | [mano.is.tue.mpg.de/license.html](https://mano.is.tue.mpg.de/license.html) |
| Python packages in `uv.lock` | Each package retains its own license | Package metadata and source distributions |
| MolmoAct2-BimanualYAM-Dataset (Ai2) | Apache-2.0 | [allenai/MolmoAct2-BimanualYAM-Dataset](https://huggingface.co/datasets/allenai/MolmoAct2-BimanualYAM-Dataset) |
| ABC-130k (XDOF) | Apache-2.0 | [XDOF/ABC-130k](https://huggingface.co/datasets/XDOF/ABC-130k) |
| Galaxea Open-World Dataset (Galaxea) | CC BY-NC-SA 4.0, non-commercial | [OpenGalaxea/Galaxea-Open-World-Dataset](https://huggingface.co/datasets/OpenGalaxea/Galaxea-Open-World-Dataset); the adapter reads the copy at `RogersPyke/Galaxea-Open-World-Dataset_10K_20260123`; accept Galaxea's terms on its page before using it |
| HABIT (Config) | CC BY 4.0 | [configinc/HABIT](https://huggingface.co/datasets/configinc/HABIT) |
| FastUMI-100K | Apache-2.0 | [IPEC-COMMUNITY/FastUMI_100k_lerobot](https://huggingface.co/datasets/IPEC-COMMUNITY/FastUMI_100k_lerobot) |
| 10Kh-RealOmin-OpenData (GenRobot) | CC BY-SA 4.0 | [genrobot2025/10Kh-RealOmin-OpenData](https://huggingface.co/datasets/genrobot2025/10Kh-RealOmin-OpenData) |
| Egocentric-100K (Build AI) | Apache-2.0 | [builddotai/Egocentric-100K](https://huggingface.co/datasets/builddotai/Egocentric-100K) |
| Gen-HumanEgo (GenRobot) | CC BY-SA 4.0 | [genrobot2025/Gen-HumanEgo](https://huggingface.co/datasets/genrobot2025/Gen-HumanEgo) |
| OpenAoE-2000h (inclusionAI) | Open-AoE Dataset License, attribution required; its MANO-derived files, which the adapter never reads, also fall under MANO's license | [inclusionAI/OpenAoE-2000h](https://huggingface.co/datasets/inclusionAI/OpenAoE-2000h/blob/main/LICENSE) |

Running a dataset through this pipeline does not change its license, so credit each dataset as its license asks when you publish its footage or labels. The hand keypoints come from the ACE-Ego-Hand checkpoints (CC BY-NC 4.0) and a model built on MANO (non-commercial research only), so use them only for non-commercial purposes and credit ACE-Ego-Hand.
