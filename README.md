# SemiRestore — KLA PS01

Reproducible grayscale image restoration for the SEMICON India Hackathon 2026 problem **AI-Based Restoration of Degraded Images for Semiconductor Inspection**.

## Current status

The repository contains the compliance-first inference path, deterministic paired-data audit, locked labeled-metric tooling, and the EDSR-lite learned-baseline training path. Bicubic is deliberately a runnable lower bound, **not the final learned model**. EDSR-lite GPU verification is in progress; NAF-SR follows after it clears the validation gate.

## Supported data

- input: one 2D grayscale `.npy` array per sample;
- dtype: any real numeric dtype accepted by NumPy, converted to contiguous float32;
- range: preserved exactly at input, including negative values and values above `1.0`;
- output: one float32 `.npy` per input, same relative filename, shape `2H×2W`;
- final output range: clipped to `[0,1]`; and
- metadata paths named `__MACOSX` or files prefixed `._` are ignored.

NumPy files are loaded with `allow_pickle=False`. Invalid dimensionality, non-finite values, empty inputs, missing directories, and unsafe output reuse fail with a non-zero exit code.

## Quick start

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Run directory-to-directory restoration:

```powershell
python evaluation.py C:\path\to\NoisyLR C:\path\to\restored_outputs
```

The command is intentionally independent of the current working directory. Until learned weights are added, it uses bicubic interpolation and prints a concise count/device/latency summary.

Useful options:

```powershell
python evaluation.py INPUT_DIR OUTPUT_DIR --batch-size 8 --report-json C:\path\to\run.json
python evaluation.py INPUT_DIR OUTPUT_DIR --overwrite
```

`--overwrite` only permits replacing outputs that correspond to discovered inputs; it never deletes the output directory.

## Tests

```powershell
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

Tests cover metadata filtering, range preservation, validation failures, output shape/dtype/range, unsafe overwrite protection, and executing `evaluation.py` from outside the repository.

## Dataset layout

Raw organizer data is intentionally excluded from Git:

```text
data/raw/train/
  GT/000000.npy
  NoisyLR/000000.npy

data/raw/test/
  NoisyLR/000000.npy
```

See [`data/README.md`](data/README.md) and the design documents under [`docs/hackathon-build/`](docs/hackathon-build/).

## Build the paired training manifest

After extracting `train.zip`, validate every pair and produce the deterministic training manifest and audit:

```powershell
python scripts/build_manifest.py `
  --input-dir C:\path\to\train\NoisyLR `
  --target-dir C:\path\to\train\GT `
  --dataset-root C:\path\to\train
```

The command defaults to the current-release gate of 3,200 pairs and writes:

- `data/splits/manifest.csv` with relative paths, per-image statistics, and SHA-256 hashes; and
- `reports/dataset_audit.json` with counts, shapes, dtypes, global ranges, ignored metadata, byte totals, and the manifest hash.

Use `--expected-pairs 0` only for small development fixtures. Existing artifacts are protected unless `--overwrite` is explicit.

## Build the provisional validation baseline

Before the texture/source-aware OOD split is available, create an explicitly provisional 15% validation-ID holdout. Membership is determined by `SHA-256(seed:stem)`, so input row order cannot change the split:

```powershell
python scripts/assign_provisional_split.py `
  --manifest data/splits/manifest.csv
```

This writes `data/splits/manifest_provisional.csv` and `reports/provisional_split_audit.json`. It labels only `train` and `val_id`; it never claims that a random/hash holdout is OOD.

Run the labeled bicubic baseline:

```powershell
python evaluate_metrics.py `
  C:\path\to\train\NoisyLR `
  C:\path\to\train\GT `
  --manifest data/splits/manifest_provisional.csv `
  --split val_id `
  --device cuda `
  --batch-size 32
```

The evaluator writes a per-image CSV and aggregate JSON with PSNR, SSIM, LPIPS-Alex, Sobel-gradient L1, mean-intensity bias, pre-clamp out-of-range rate, worst-decile results, and deterministic 95% bootstrap confidence intervals. Metric preprocessing is fixed in `src/semirestore/metrics.py`: PSNR uses `data_range=1`, SSIM uses an 11×11 Gaussian window, and LPIPS repeats grayscale into RGB then maps `[0,1]` to `[-1,1]` once with package normalization disabled.

`--no-lpips` exists only for fast CPU development tests; do not use it for reported hackathon evidence.

### Measured bicubic lower bound

The first real-data evidence run used the deterministic provisional `val_id` holdout (480 pairs) on a Colab Tesla T4. These are preliminary validation-ID values, not OOD or final-model results:

| Metric | Mean | Deterministic 95% bootstrap CI |
|---|---:|---:|
| PSNR | 23.063744 dB | 22.779444–23.378360 |
| SSIM | 0.541088 | 0.525420–0.557836 |
| LPIPS-Alex | 0.419660 | 0.405365–0.433218 |

## Train EDSR-lite

The EDSR-lite baseline has 16 residual blocks, width 64, a 2× pixel-shuffle head, and a bicubic global skip. It has 1,367,553 parameters and trains against unclipped model output with Charbonnier loss.

First prove the complete pipeline by overfitting eight samples:

```powershell
python train.py `
  --config configs/baseline_edsr.yaml `
  --manifest C:\path\to\manifest_provisional.csv `
  --dataset-root C:\path\to\train `
  --run-dir runs/edsr_overfit8 `
  --device cuda `
  --overfit-samples 8 `
  --batch-size 8 `
  --num-workers 0 `
  --max-steps 300
```

Then run the real provisional train/validation baseline. The checked-in config declares 5,000 steps, validation every 250 steps, AdamW at `2e-4`, 100-step warm-up plus cosine decay, CUDA AMP, and gradient clipping at `1.0`:

```powershell
python train.py `
  --config configs/baseline_edsr.yaml `
  --manifest C:\path\to\manifest_provisional.csv `
  --dataset-root C:\path\to\train `
  --run-dir runs/edsr_lite_baseline `
  --device cuda
```

Every run writes `best.pt`, `last.pt`, `history.csv`, `summary.json`, `resolved_config.yaml`, and `environment.json`. Checkpoints embed the architecture, data policy, manifest hash, loss, step, and environment and are loaded with PyTorch's safe `weights_only=True` path.

Restore a directory with a learned checkpoint:

```powershell
python evaluation.py INPUT_DIR OUTPUT_DIR `
  --model edsr_lite `
  --weights runs/edsr_lite_baseline/best.pt `
  --device cuda
```

Run the identical labeled evidence suite used for bicubic:

```powershell
python evaluate_metrics.py C:\path\to\train\NoisyLR C:\path\to\train\GT `
  --model edsr_lite `
  --weights runs/edsr_lite_baseline/best.pt `
  --manifest C:\path\to\manifest_provisional.csv `
  --split val_id `
  --device cuda
```

The thin Colab launcher is [`notebooks/01_train_edsr_lite_colab.ipynb`](notebooks/01_train_edsr_lite_colab.ipynb); it only invokes the same version-controlled scripts.

## Reproducibility note

`requirements.txt` currently pins the tested baseline runtime. It will be replaced by the complete frozen CUDA training environment after the final clean-environment verification, as required by the challenge.

GPU work will run in Google Colab. The planned notebook is only an orchestration layer: model, data, training, evaluation, and checkpoint behavior remain in version-controlled Python modules and CLI scripts so the notebook is never required by KLA's evaluator. Each run will record the actual Colab GPU, CUDA, Python, and PyTorch environment rather than assuming a particular accelerator tier.
