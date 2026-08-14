# SemiRestore — KLA PS01

Reproducible grayscale image restoration for the SEMICON India Hackathon 2026 problem **AI-Based Restoration of Degraded Images for Semiconductor Inspection**.

## Current status

The final candidate is a statistics-conditioned NAF-SR trained on real pairs with paired D4 geometry. It is selected by the declared validation-ID/pseudo-OOD rank policy. Native PyTorch FP32 is the default: it is faster than BF16 on the measured Tesla T4 runtime.

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

`requirements.txt` is the clean CPU/CI environment. GPU training and benchmark evidence used the Colab-provided CUDA runtime recorded in `requirements-colab.txt` and `reports/benchmark_fp32_batch1.json`; the measured core was Python 3.12.13, PyTorch 2.11.0+cu128, CUDA 12.8, cuDNN 91900, on a Tesla T4.

Run directory-to-directory restoration:

```powershell
python evaluation.py C:\path\to\NoisyLR C:\path\to\restored_outputs
```

The command is intentionally independent of the current working directory and loads `weights/model.pt` by default. It prints a concise count/device/latency summary.

Useful options:

```powershell
python evaluation.py INPUT_DIR OUTPUT_DIR --batch-size 8 --report-json C:\path\to\run.json
python evaluation.py INPUT_DIR OUTPUT_DIR --overwrite
python evaluation.py INPUT_DIR OUTPUT_DIR --model bicubic
```

`--overwrite` only permits replacing outputs that correspond to discovered inputs; it never deletes the output directory.

## Generate and verify the 400 public-test outputs

After extracting the organizer public test archive, generate the required outputs with the frozen final checkpoint:

```powershell
python scripts/generate_test_outputs.py `
  C:\path\to\test\NoisyLR `
  restored_test_outputs `
  --device cuda

python scripts/verify_submission.py `
  --input-dir C:\path\to\test\NoisyLR
```

The verifier requires exactly 400 matching float32 outputs with 2× spatial dimensions, finite `[0,1]` values, and a checkpoint matching `weights/model.sha256`.

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

### Measured EDSR-lite baseline

The 5,000-step EDSR-lite run used the identical 480-pair provisional `val_id` split and metric implementation as bicubic:

| Metric | Bicubic | EDSR-lite | Change |
|---|---:|---:|---:|
| PSNR | 23.063744 dB | 27.791390 dB | +4.727646 dB |
| SSIM | 0.541088 | 0.749483 | +0.208395 |
| LPIPS-Alex ↓ | 0.419660 | 0.305228 | -0.114432 |

On a Colab Tesla T4, the training process recorded `1354.33 s` and `1.4646 GiB` peak allocated CUDA memory. These remain provisional validation-ID results, not texture-held-out OOD results.

## Train NAF-SR

The unconditioned primary model uses a width-48, three-stage NAF encoder/decoder with block layout `[2,2,4] / 6 / [2,2,2]`, additive skips, a 2× pixel-shuffle residual head, and a bicubic global skip. It has 8,974,084 parameters; its FP32 model state is approximately 34.23 MiB.

Start with the required 200-step T4 calibration before choosing the full-run batch size and duration:

```powershell
python train.py `
  --config configs/naf_sr.yaml `
  --manifest C:\path\to\manifest_provisional.csv `
  --dataset-root C:\path\to\train `
  --run-dir runs/naf_sr_calibration_200 `
  --device cuda `
  --batch-size 4 `
  --max-steps 200
```

`best.pt` is a compact self-describing inference checkpoint without optimizer moments. `last.pt` retains the raw model, EMA, best model, optimizer, scheduler, AMP scaler, and DataLoader RNG state. The thin launcher is [`notebooks/02_train_naf_sr_colab.ipynb`](notebooks/02_train_naf_sr_colab.ipynb).

### Measured NAF-SR baseline

The identical-budget NAF-SR run used batch 16, 5,000 steps, and the same split, seed, loss, and locked metric pipeline as EDSR-lite:

| Metric | EDSR-lite | NAF-SR | Change |
|---|---:|---:|---:|
| PSNR | 27.791390 dB | 28.178510 dB | +0.387120 dB |
| SSIM | 0.749483 | 0.760657 | +0.011174 |
| LPIPS-Alex ↓ | 0.305228 | 0.276804 | -0.028424 |

On a Colab Tesla T4, training recorded `1871.64 s`, `374.33 ms/step`, and `3.7012 GiB` peak allocated CUDA memory. The compact inference checkpoint is `34.36 MiB`. NAF-SR is the selected primary architecture; EDSR-lite remains the fallback.

## Deterministic debug and resume

The checked-in training configs enable EMA with decay `0.999`. Validation scores both raw and EMA weights, records both in `history.csv`, and writes whichever has the best PSNR to the compact `best.pt`. Every run also records the resolved config, manifest SHA-256, environment, metrics, seed mode, and resume provenance.

For a short deterministic verification, run the same CPU command twice and compare the selected checkpoint tensors. The required tolerance is exact (`atol=0`) in the tested CPU runtime:

```powershell
python train.py --config configs/naf_sr.yaml --device cpu --deterministic `
  --num-workers 0 --stop-after-step 2 --run-dir runs/debug_a
python train.py --config configs/naf_sr.yaml --device cpu --deterministic `
  --num-workers 0 --stop-after-step 2 --run-dir runs/debug_b
python scripts/compare_training_runs.py runs/debug_a runs/debug_b --atol 0
```

`--stop-after-step` stops at an absolute step without changing the configured full learning-rate schedule. Resume from the full-state checkpoint into the same or a new run directory:

```powershell
python train.py --config configs/naf_sr.yaml `
  --resume runs/naf_interrupted/last.pt `
  --run-dir runs/naf_resumed
```

On resume, the engine validates model metadata, planned step count, manifest hash, and sample policy before loading any state. It then begins a fresh deterministic epoch from the saved DataLoader RNG state; this policy is written into `resolved_config.yaml` and `summary.json` rather than implying bitwise continuation from the middle of an epoch.

The updated engine was calibrated at batch 16 for 200 steps on a Colab Tesla T4. It recorded `390.27 ms/step`, `3.7333 GiB` peak allocated CUDA memory, and a successful step `200 -> 220` resume in a new run directory. At this short horizon raw weights scored `26.763529 dB` versus EMA's `23.401285 dB`, so the automatic selector correctly retained raw weights.

## Build the texture-held-out split

The provisional hash holdout is not used for OOD claims. Build the final development split from the organizer's labeled training data only:

```powershell
python scripts/assign_texture_ood_split.py `
  --manifest data/splits/manifest.csv `
  --dataset-root C:\path\to\train `
  --output data/splits/manifest_texture_ood.csv `
  --audit reports/texture_split_audit.json
```

The splitter computes eleven deterministic GT descriptors covering intensity, entropy, gradients, Laplacian response, and radial FFT energy. Fixed-seed k-means forms 12 texture clusters; complete outlying clusters become `val_ood`, `val_id` is hash-stratified inside the remaining clusters, and the remainder is `train`. The audit records descriptor scaling, cluster sizes, held-out membership, hashes, and an explicit `public_test_used: false` assertion.

Fit synthetic degradation ranges strictly from the resulting `train` rows:

```powershell
python scripts/fit_degradation_profile.py `
  --manifest data/splits/manifest_texture_ood.csv `
  --dataset-root C:\path\to\train `
  --output reports/degradation_profile.json
```

Three same-budget configurations isolate the robustness variables:

- `configs/naf_sr_ood_baseline.yaml`: paired D4 geometry, 100% real pairs;
- `configs/naf_sr_synthetic15.yaml`: the baseline plus 15% fitted synthetic samples; and
- `configs/naf_sr_conditioned.yaml`: the baseline plus internal `[mean,std,min,max]` FiLM conditioning.

Synthetic operations—blur, area/bicubic downsampling, Gaussian noise, and multiplicative speckle—are randomly ordered per selected training sample and never clamp the generated input. Validation and public-test samples are never augmented. Metric reports include per-texture-cluster aggregates.

After scoring each checkpoint separately on `val_id` and `val_ood`, use `scripts/record_ablation.py` to write the controlled decision table. The predeclared gate keeps a change only when at least two of three OOD metrics improve and validation-ID loses no more than `0.15 dB` PSNR or `0.002` SSIM.

The measured 12-cluster split contains 2,181 train, 480 validation-ID, and 539 pseudo-OOD pairs. Bicubic scores `23.796180 dB / 0.552271 / 0.408522` PSNR/SSIM/LPIPS on ID and `20.227942 dB / 0.425777 / 0.495734` on pseudo-OOD. The fitted profile found Gaussian-noise standard deviation `[0.032383,0.080666]`, speckle `[0.045366,0.063229]`, near-zero additive bias, and no evidence for extra blur beyond the downsampling base; an unsupported blur range was not fabricated.

## Reproducibility note

`requirements.txt` currently pins the tested baseline runtime. It will be replaced by the complete frozen CUDA training environment after the final clean-environment verification, as required by the challenge.

GPU work will run in Google Colab. The planned notebook is only an orchestration layer: model, data, training, evaluation, and checkpoint behavior remain in version-controlled Python modules and CLI scripts so the notebook is never required by KLA's evaluator. Each run will record the actual Colab GPU, CUDA, Python, and PyTorch environment rather than assuming a particular accelerator tier.
