# SemiRestore — KLA PS01

Reproducible grayscale image restoration for the SEMICON India Hackathon 2026 problem **AI-Based Restoration of Degraded Images for Semiconductor Inspection**.

## Current status

The repository currently contains the first compliance-first vertical slice: safe NumPy input handling and a standalone bicubic 2× baseline. Bicubic is deliberately a runnable lower bound, **not the final learned model**. EDSR-lite and NAF-SR training are the next checklist items.

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

## Reproducibility note

`requirements.txt` currently pins the tested baseline runtime. It will be replaced by the complete frozen CUDA training environment after the final clean-environment verification, as required by the challenge.

GPU work will run in Google Colab. The planned notebook is only an orchestration layer: model, data, training, evaluation, and checkpoint behavior remain in version-controlled Python modules and CLI scripts so the notebook is never required by KLA's evaluator. Each run will record the actual Colab GPU, CUDA, Python, and PyTorch environment rather than assuming a particular accelerator tier.
