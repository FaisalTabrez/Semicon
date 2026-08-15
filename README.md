# SemiRestore — KLA PS01

SemiRestore restores noisy, 2× downsampled grayscale semiconductor-inspection
arrays with a statistics-conditioned NAF-SR model. This repository is the code
submission for **AI-Based Restoration of Degraded Images for Semiconductor
Inspection** in the SEMICON India Hackathon 2026.

The packaged checkpoint is frozen and self-describing:

- model: conditioned NAF-SR, real paired training plus D4 augmentation;
- parameters: 9,111,684;
- checkpoint: `weights/model.pt` (34.87 MiB);
- SHA-256: `273abd9d6dcfa9bdee71ac15016994962304b6c9d902898b4f4d503bed158c28`;
- default inference: eager PyTorch FP32, batch 16; and
- output: float32 `.npy`, 2× spatial dimensions, clipped to `[0,1]`.

## Judge quick start

```bash
git clone https://github.com/FaisalTabrez/Semicon.git
cd Semicon
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements-runtime.txt
```

Run the required standalone evaluation script with exactly the two mandatory
paths:

```bash
python evaluation.py /path/to/NoisyLR /path/to/restored_outputs
```

`evaluation.py` resolves repository assets independently of the current working
directory, automatically loads `weights/model.pt`, uses CUDA when available,
and processes the complete directory without source edits. Useful options:

```bash
python evaluation.py INPUT_DIR OUTPUT_DIR --device cuda --batch-size 16
python evaluation.py INPUT_DIR OUTPUT_DIR --device cpu
python evaluation.py INPUT_DIR OUTPUT_DIR --overwrite \
  --report-json /path/to/inference_report.json
```

The default remains eager FP32 because it has no compilation delay and is
portable across the evaluator's PyTorch/CUDA environment. Batch 16 was selected
by A100 calibration for directory throughput. Mixed spatial sizes are grouped
safely within each requested chunk.

## Official submission requirement mapping

| Organizer requirement | Repository location |
|---|---|
| Public source repository | This GitHub repository |
| Standalone evaluation `.py` | `evaluation.py` |
| Training script or notebook | `train.py` and optional Colab notebook |
| Trained weights | `weights/model.pt` plus `weights/model.sha256` |
| Restored public-test arrays | `restored_test_outputs/` |
| Complete training freeze (final A100 capture) | `requirements.txt` |
| Portable evaluator install | `requirements-runtime.txt` |
| Clone-and-run instructions | Judge quick start and input/output contract below |

The notebook is supplementary. The organizer's evaluator can invoke
`evaluation.py INPUT_DIR OUTPUT_DIR` as-is, without opening Jupyter or changing
hard-coded paths.

## Input/output contract

- Input: recursively discovered 2D grayscale `.npy` arrays.
- Input dtype: any real NumPy numeric dtype, converted to contiguous float32.
- Input range: preserved, including negative values and values above `1.0`.
- Output: identical relative `.npy` paths with shape `(2H, 2W)`.
- Output dtype/range: float32, finite, clipped to `[0,1]`.
- Loading: `allow_pickle=False`; `__MACOSX` and `._*` metadata are ignored.
- Invalid input, unsafe reuse, shape mismatch, NaN, or infinity returns non-zero.

`--overwrite` replaces matching outputs but never deletes an output directory.
`--report-json` must be outside the restored-output tree.

## Verify the packaged submission

The repository includes all 400 restored public-test outputs. Given the
organizer's extracted inputs, verify paths, shapes, dtype, range, and checkpoint:

```bash
python scripts/verify_submission.py \
  --input-dir /path/to/test/NoisyLR \
  --output-dir restored_test_outputs \
  --weights weights/model.pt \
  --sha256 weights/model.sha256 \
  --expected-count 400
```

Regenerate them through the submission path:

```bash
python scripts/generate_test_outputs.py \
  /path/to/test/NoisyLR restored_test_outputs \
  --weights weights/model.pt --device cuda --batch-size 16 --overwrite
```

## One-notebook Colab walkthrough

Open [`notebooks/SemiRestore_KLA_PS01_Colab.ipynb`](notebooks/SemiRestore_KLA_PS01_Colab.ipynb)
in Colab. The quick path clones the repository, preserves Colab's CUDA PyTorch,
downloads the public test archive, calls the standalone `evaluation.py`, verifies
all 400 outputs, and displays a sample.

The optional training section rebuilds the locked split, runs a 200-step
calibration without shortening the declared schedule, and resumes the final
configuration to 5,000 steps. The notebook is only orchestration; KLA inference
does not depend on it.

## Reproduce training from Python

Organizer data is excluded. Extract `train.zip` as:

```text
/path/to/train/
├── GT/000000.npy
└── NoisyLR/000000.npy
```

Build the paired manifest and deterministic texture-held-out split:

```bash
python scripts/build_manifest.py \
  --input-dir /path/to/train/NoisyLR \
  --target-dir /path/to/train/GT \
  --dataset-root /path/to/train \
  --manifest data/splits/manifest.csv \
  --audit reports/dataset_audit.json \
  --expected-pairs 3200 --overwrite

python scripts/assign_texture_ood_split.py \
  --manifest data/splits/manifest.csv \
  --dataset-root /path/to/train \
  --output data/splits/manifest_texture_ood.csv \
  --audit reports/texture_split_audit.json \
  --clusters 12 --validation-id-fraction 0.15 \
  --validation-ood-fraction 0.15 --seed 2026 --overwrite
```

Expected texture-manifest SHA-256:
`5c95b6353112e1d1ffe87f091c47af4528aff139b09fe40de9b1ffb2f030afae`.

Calibrate and resume full training:

```bash
python train.py \
  --config configs/final_conditioned.yaml \
  --manifest data/splits/manifest_texture_ood.csv \
  --dataset-root /path/to/train \
  --run-dir runs/final_calibration_200 \
  --device cuda --batch-size 16 --num-workers 2 \
  --stop-after-step 200 --overwrite

python train.py \
  --config configs/final_conditioned.yaml \
  --manifest data/splits/manifest_texture_ood.csv \
  --dataset-root /path/to/train \
  --run-dir runs/final_conditioned_5000 \
  --device cuda --batch-size 16 --num-workers 2 \
  --resume runs/final_calibration_200/last.pt --overwrite
```

Runs write compact `best.pt`, resumable `last.pt`, `history.csv`, `summary.json`,
`resolved_config.yaml`, and `environment.json`. Checkpoints record architecture,
manifest hash, augmentation/loss policy, environment, and raw/EMA selection.

## Locked quality evidence

The final split has 2,181 train, 480 validation-ID, and 539 texture-cluster
pseudo-OOD pairs. All choices were made without organizer test labels.

| Split | Method | PSNR (dB) | SSIM | LPIPS-Alex ↓ |
|---|---|---:|---:|---:|
| Validation-ID | Bicubic | 23.796180 | 0.552271 | 0.408522 |
| Validation-ID | SemiRestore | 29.226261 | 0.776577 | 0.271403 |
| Pseudo-OOD | Bicubic | 20.227942 | 0.425777 | 0.495734 |
| Pseudo-OOD | SemiRestore | 25.251258 | 0.666080 | 0.353022 |

The 15% synthetic-degradation ablation was rejected. Statistics conditioning
passed the predeclared OOD gate. A 4.11M-parameter distilled student improved
over its supervised control but failed the fine-detail and batch-1 latency gates,
so it did not replace the teacher. See
[`reports/student32_ablation.md`](reports/student32_ablation.md).

## A100 calibration and deployment modes

Synchronized model-only measurements on NVIDIA A100-SXM4-40GB, Python 3.12.13,
PyTorch 2.11.0+cu128, and CUDA 12.8:

| Execution mode | Batch/shape | Result | Decision |
|---|---|---:|---|
| Eager FP32 | 1×128×128 | 14.871 ms median | Portable fallback |
| Eager FP32 | 16×128×128 | 542.61 images/s | Directory default |
| Compiled FP32, channels-last | 1×128×128 | 1.890 ms median | Quality-safe |
| Compiled FP16, channels-last | 16×128×128 | 1,958.49 images/s | Quality-safe |

Compilation costs approximately 20–23 seconds on first use for each new shape,
so compiled modes are persistent-service options, not the cold-start default.
Locked compiled FP32/FP16 changes stayed below 0.0003 dB PSNR on both splits.

## Cold external-domain validation

The frozen model was tested without tuning on all 4,591 images in the Carinthia
SEM Defect Dataset using deterministic 2× controlled degradations:

| Condition | NAF-SR minus bicubic PSNR | SSIM change | PSNR/SSIM wins |
|---|---:|---:|---:|
| Organizer-profile low noise | +8.15598 dB | +0.347664 | 100% / 100% |
| Organizer-profile high noise | +4.09757 dB | +0.232346 | 100% / 100% |
| Downsample only | -2.17861 dB | -0.033734 | 0% / 0% |

This is controlled-degradation evidence, not native paired LR/HR evidence. The
clean-input failure shows that the model can over-smooth already-clean texture.
See [`reports/external_carinthia_evidence.md`](reports/external_carinthia_evidence.md).

## Requirements and tests

- `requirements.txt`: organizer-mandated complete A100 training-environment
  `pip freeze` provenance (703 entries, including PyTorch 2.11.0+cu128).
- `requirements-runtime.txt`: minimal pinned standalone CPU/CI installation.
- `requirements-colab.txt`: pinned non-PyTorch Colab packages; preserves the
  CUDA wheel supplied by Colab.
- `requirements-dev.txt`: test dependencies.

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

Tests cover clean-checkout CLI execution, checkpoint loading, input safety,
manifest determinism, metrics, training/resume, distillation provenance,
external evaluation, and submission verification.

## Repository map

```text
evaluation.py                 Required standalone inference entry point
train.py                      Reproducible training and resume engine
evaluate_metrics.py           PSNR/SSIM/LPIPS evaluator
weights/                      Frozen model, checksum, model card
restored_test_outputs/        400 packaged public-test restorations
configs/                      Final, baseline, ablation, student configs
src/semirestore/              Models, data, metrics, inference
scripts/                      Verification, benchmark, split, external tools
notebooks/                    Optional Colab orchestration
reports/                      Machine-readable and written evidence
docs/hackathon-build/         Design, roadmap, and build log
tests/                        Automated fresh-environment checks
```

## Safety and limitations

- Pseudo-OOD clusters are a robustness proxy, not hidden-test labels.
- Carinthia has no native aligned degraded/clean pairs.
- Restoration can suppress genuine texture on cleaner-than-training inputs.
- Outputs support inspection review, not a manufacturing disposition.
- Never replace `weights/model.pt` without regenerating its checksum, all 400
  outputs, quality evidence, and clean-clone verification.
