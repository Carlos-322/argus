# Argus

Argus annotates robot-learning recordings and checks their data quality. Give it a dataset and a short description, then review the timeline, task outcomes, mistakes and supporting evidence beside the footage.

It reads LeRobot, MCAP, HDF5, video files and archives for teleoperated arms, handheld UMI grippers and human ego video. Recorded instructions, state and sensors provide evidence where their ownership, units and clocks support it. Video-only inputs are supported too.

The project accompanies [Argus: An Open-Source Annotator for Robotics Data](https://pantheon.inc/research/argus). See the [published dashboard](https://pantheon.inc/data-board) for example annotations or [Data Review](https://data.pantheon.inc/review) for hosted uploads. Hosted deployments have their own version and upload limits.

![An episode playing with its cameras, timeline and outcome](media/board.gif)

## Install

You need Python 3.11, [uv](https://docs.astral.sh/uv/), Node.js for viewer checks, and `ffmpeg` with `ffprobe` 5.1 or newer.

```bash
git clone https://github.com/Pantheon-Industries-Inc/argus.git
cd argus
uv sync --frozen
uv run --frozen --offline pytest -q -ra
```

The tests make no model calls. Paid annotation uses `OPENROUTER_API_KEYS` (comma-separated keys) or `OPENAI_API_KEY` for OpenAI models directly. Set credentials in your shell or secret manager. Downloading gated datasets also requires accepting their terms and setting `HF_TOKEN`. Runtime data and outputs belong under the ignored `data/` directory.

## Review your data

First build the base requests without making model calls:

```bash
uv run python -m review --data path/to/dataset --rig teleop_arms \
  --dataset-note "Two arms arrange blocks on a tray." --out data/review/mine --free
```

Then annotate under a spending cap and open the result:

```bash
uv run python -m review --data path/to/dataset --rig teleop_arms \
  --dataset-note "Two arms arrange blocks on a tray." --out data/review/mine --cap 20
uv run python -m board serve --board data/review/mine --clips data/review/mine/clips
```

`--data` accepts a folder, file, archive or HTTP(S) URL. Choose `--rig teleop_arms`, `handheld_gripper` or `ego_head`. The description supplies context; recorded instructions and metadata stay attributed to their source.

| Choice | Behavior |
|---|---|
| Astra, the default | `openai/gpt-6-astra`, medium reasoning; one regular sample every 1.5 s for teleop, 1 s for UMI and 0.5 s for ego. First/last frames and eligible evidence boundaries are also included. |
| Sol 6.1 High | Add `--model openai/gpt-6.1-sol --reasoning high`. Its default is 1 FPS on every rig, with a minimum of 1 FPS. |
| Denser sampling | Add `--sample-fps 2` for subsecond fumbles or rapid recoveries. More images reduce the available request window and increase cost. |
| Task scope | `--task-mode episode` evaluates one goal. `--task-mode session` annotates separate task spans and outcomes within the same recording. Reliable source declarations can establish session scope; an absent instruction alone does not. Ego retains its activities schema. |

One recording stays one public episode. Long recordings use bounded internal request windows, then rejoin their labels on the original clock. Session tasks are annotations within that recording; they do not physically cut it into new task episodes. Explicit preparation or hosted footage limits are separate and report any trimming. Missing or unparsed parts stay visible at their spans, and an unlabelled recording keeps its usable footage, checks and sensors.

The resolved model, reasoning, cadence, task scope and description are saved with the job and reused on resume. Changing the paid request identity requires a new run. Request records show effective settings, selected frames and omitted or qualified evidence.

`--free` builds base requests and skips paid routing, field interpretation and evidence selection. During a paid run, eligible extra sensors or depth may trigger bounded interpretation and inspection choices before the final request. Their selections, receipts and costs are retained. A dry request therefore does not promise every later paid evidence choice. The cap covers applicable calls and outstanding reservations; an unknown charge remains reserved rather than being reported as free.

## Results and evidence

A review job contains prepared `episodes/`, base requests in `dry/`, paid receipts in `run/`, the rejoined outputs in `run_final/`, and the board. Staged label runs write `data/runs/<dataset>/<run_id>/`, with `run.json` and per-episode outputs.

Labels include action phases marked advancing, wasteful or idle; key events, progress and instruction alignment; task outcomes and goal frames; operator mistakes, failed attempts and recoveries; scene changes; and recording issues. Robot episode completion and session/ego task outcomes use their respective schemas. Examples live in `configs/examples/`.

Prepared episodes retain native arrays, clocks, calibration and attributed publisher claims. Unresolved ownership or conflicting annotations remain qualified. Motion checks require declared physical roles, suitable units and usable clocks; a vector's width alone does not establish joints or poses. Pressure, optical motion, jaw movement and relative depth can support observations, but force, slip, physical distance and contact timing require suitable calibration and timing evidence.

The board keeps the episode navigation and existing action, progress, recovery, depth and hand-pose controls. It distinguishes recording faults from model or processing failures. Useful sensor conclusions appear with their readings, reasoning, confidence, alternatives and limitations. An inspection can return no additional finding.

Episode JSON, filtered JSON Lines and static exports retain annotations and substantive evidence. Public projection withholds private access descriptors and infrastructure locations. Human field interpretations take precedence, including an explicitly blank role; stale conclusions are withheld when the source or interpretation no longer supports them. Keep uploaded footage under its own access policy and license when sharing a board.

## Advanced stages

The normal entrypoint is `review`. Individual stages remain available for dataset downloads, existing prepared episodes and custom boards:

```bash
uv run python -m prepare folder prepare --root path/to/data --rig teleop_arms --out data/episodes/mine/all
uv run python -m checks data/episodes/mine/all
uv run python -m label --dataset mine --episodes data/episodes/mine/all --kind dry
uv run python -m label --dataset mine --episodes data/episodes/mine/all --kind full --cap 20
```

Dataset adapters keep their download and sampling entrypoints. For example:

```bash
uv run python -m prepare molmo prepare --episodes configs/quickstart/teleop.txt --out data/episodes/molmo/quickstart
uv run python -m prepare lerobot prepare --root path/to/lerobot --rig teleop_arms --out data/episodes/mine/all
uv run python -m prepare videos prepare --root path/to/videos --rig ego_head --instructions instructions.json --out data/episodes/ego/all
```

The public adapters are `molmo`, `abc130k`, `galaxea`, `habit`, `fastumi`, `realomin`, `egocentric100k`, `genhumanego` and `openaoe`. `configs/slices/` records the published sample selections. Generic upload readers preserve source fields rather than applying corpus-specific release cropping or task selection.

`checks` writes deterministic results into each episode's `context.json`, including clock, camera pairing, signal, capture and supported sensor checks. It reports unreadable inputs and check failures separately from a clear result. Timebase review remains available through `python -m checks.timebase scan` and `apply`; inspect the proposed correction before applying it.

Paid `label` runs require a clean tracked checkout. `--resume RUN --why "reason"` continues an interrupted staged run with its saved settings and remaining budget. Stored replies and billing receipts survive failures. `python -m label.reparse RUN` reparses saved replies without another model call. A retained failure is distinct from authorization to dispatch or charge again.

Frame selection includes the declared cadence, first/last frames and supported boundaries. Teleop resolution routing, calibration, retained camera coverage and provider limits affect the actual request. Its record retains image order, sampled times and effective resolution. Model outputs are stochastic; neither repeated labels nor image bytes across different decoding environments are promised to be identical.

## Boards, comparisons and gate

For a custom board, copy `configs/quickstart/board.json`, then set its dataset, run and prepared-episode paths:

```bash
mkdir -p data/boards/mine
cp configs/quickstart/board.json data/boards/mine/manifest.json
uv run python -m board build data/boards/mine
uv run python -m board clips --episodes data/episodes/mine/all --out data/clips
uv run python -m board serve --board data/boards/mine --clips data/clips
uv run python -m board static media --board data/boards/mine --clips data/clips
uv run python -m board static site --board data/boards/mine --clips data/clips
```

Live and static pages use the canonical `board/index.html` through `board.serve.render_index`. Refresh a board after changing its field dictionary before producing a static export. Local owner editing is enabled explicitly with `board serve --dictionary-job JOB`.

Requests record ordered image hashes without storing duplicate image archives. `uv run python -m board grids data/boards/mine` reconstructs base frame grids only when their original inputs and image bytes can be verified. The viewer distinguishes answered requests from dry or unverified dispatches, keeps request-part clocks, and identifies unavailable parts. Detail and sensor images are counted in the receipt but are not reconstructed by this command. Legacy labels without input hashes remain unavailable.

Model comparisons and frame-verified gate cases remain separate paid workflows:

```bash
uv run python -m compare prepare --selection configs/compare/main.json
uv run python -m compare label --selection configs/compare/main.json --models astra,sol61_high --kind full --cap 20
uv run python -m compare board --entries data/runs/compare/main.json --out data/boards/compare
uv run python -m gate prepare
uv run python -m gate label --kind full --cap 20
uv run python -m gate score data/runs/gate_teleop/RUN data/runs/gate_handheld/RUN data/runs/gate_ego/RUN
```

Comparison caps apply per model run; gate caps apply per rig run. Effective sampling may differ by model, so compare saved requests before treating a result as an identical-input comparison. `--with-example` supplies a reference annotation, and `--routes` reuses recorded resolution choices. `python -m compare.metrics BOARD` summarizes a comparison board. Gate selections and expected facts are in `gate/selection.json` and `gate/cases.json`; current scores come from the runs you score, not a number pinned in this README.

The optional ACE-Ego-Hand overlay uses `board/hand_pose/modal_app.py` on Modal GPUs and `board/hands.py` to build and verify overlay files. Its module docstring contains the setup commands, including MANO registration and download. It is separate from ordinary annotation and has the model licenses below.

Every entrypoint takes `--help`. Offline CI runs the locked Python environment and viewer checks with Node, ffmpeg and ffprobe, without model credentials. Secret scanning covers the committed tree and reachable Git history; it does not establish permission to publish uploaded data.

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
