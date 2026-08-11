# Build Notes and Decision Log

## 11 August 2026 — research and planning pass

### Sources checked

- official [SEMICON India Hackathon 2026 page](https://i4c.in/hackathon-2026/);
- downloadable KLA detailed problem-statement deck;
- downloadable official idea-submission template;
- linked public Google Drive dataset folder and ZIP central-directory metadata;
- NAFNet, EDSR, SwinIR, Restormer, and LPIPS primary papers; and
- current PyTorch reproducibility, AMP, and ONNX documentation.

### Verified local repository state

- Repository was an empty Git repository with no commits.
- No prior implementation, README, data, hackathon state, or build documents existed.
- This planning pass intentionally created only documents, not an untested code scaffold.

### Dataset findings

- `train.zip`: 918,994,209 bytes.
- 3,200 matching real float32 pairs under `train/NoisyLR/` and `train/GT/`.
- All current training inputs are `128×128`; all current GT arrays are `256×256`.
- `Test_NoisyLR.zip`: 23,419,125 bytes with 400 real float32 `128×128` arrays.
- Test global observed range is approximately `[-0.22488, 2.15802]`.
- Both archives include `__MACOSX` and AppleDouble `._*` metadata entries that must be filtered.
- Six training pairs sampled across indices confirmed GT `[0,1]` and degraded values beyond that range.

### Decisions

- Use a compact NAFNet-style low-resolution backbone with a 2× pixel-shuffle residual head.
- Make bicubic, EDSR-lite, and unconditioned NAF-SR mandatory baselines.
- Treat statistics conditioning, synthetic degradation, LPIPS loss, ONNX, compilation, and TTA as removable ablations.
- Preserve raw degraded intensities and clamp only at the final metric/file boundary.
- Use a texture/source-aware pseudo-OOD holdout rather than trusting a random split.
- Prefer native PyTorch FP32 submission inference until BF16 parity is proven.
- Prioritize a top-level, no-edit, directory-to-directory `evaluation.py` above architectural novelty.
- Target a checkpoint small enough to live in the public repository without a fragile download step.

### Assumptions requiring confirmation

1. The judge input/output file format will remain float32 `.npy`, and output filenames should preserve input stems.
2. Metrics use output/GT in `[0,1]`, with no border crop, and LPIPS uses a specified backbone/preprocessing.
3. The currently public unlabeled 400-image test set may be used to produce required outputs but not for supervised tuning.
4. External weight hosting is allowed, but a local checkpoint remains the safer design.
5. Nine content slides and `TeamName_KLA_PS01.pdf` supersede stale text inside the template.
6. A demo video is safest to treat as required and capped at five minutes.
7. Test-time adaptation on organizer inputs is either prohibited or not worth the reproducibility risk unless explicitly allowed.

### Questions for the 12 August submission Q&A

- What exact positional/flag CLI signature will benchmarking use?
- Must outputs be `.npy`, and should their relative paths and stems exactly mirror inputs?
- Are predictions clipped to `[0,1]` before official PSNR/SSIM/LPIPS computation?
- Which SSIM window/options and LPIPS backbone are official, and is any border cropped?
- How are PSNR, SSIM, LPIPS, and H100 inference time weighted or combined?
- Is the video mandatory for KLA PS01, and what URL privacy setting is accepted?
- Is the limit nine content slides for KLA PS01 despite the template's older 6–7 slide sentence?
- May evaluation download weights, or must the judge machine operate offline?
- Is unsupervised test-time adaptation or self-ensemble permitted, and is its full time counted?

### Skill adaptation note

The available hackathon workflow skill is designed for Devpost and requires Devpost-specific state. This event is hosted by I4C and the repository had no Devpost state, so no Devpost consent/state flow was created. Its requirements-first and technical-spec structure was applied using I4C's official materials as the source of truth.

## 11 August 2026 — build start

- The participant explicitly authorized implementation with “let's start building then”; autonomous mode with verification checkpoints was selected.
- Local development environment: Python 3.13.2, PyTorch 2.6.0 CPU, NumPy 2.2.6, PyYAML 6.0.2, pytest 8.3.5.
- The participant selected Google Colab for every GPU-dependent section.
- Consequence: core logic remains in scripts/modules; a thin Colab notebook will orchestrate data access, training, resume, and artifact export without becoming part of judge inference.
- The first checklist item is the compliance-first bicubic vertical slice because it validates the exact directory-to-directory contract before learned-model work.

### Checklist item 1 verification

- Added the package scaffold, safe NumPy discovery/loading, bicubic 2× baseline, standalone `evaluation.py`, dependency files, data instructions, README, and tests.
- Automated verification: `python -m pytest -q` → 12 passed; `python -m compileall -q evaluation.py src tests` → passed.
- Public-data smoke verification: all 400 released test inputs restored from a foreign working directory on CPU in 0.698 seconds end-to-end (1.744 ms/image for this baseline run).
- Verified public outputs: 400/400 files, every output `256×256` float32, all finite, observed final range `[0,1]`.
- The smoke timing is a local bicubic CPU measurement, not a learned-model or H100 claim.

### Checklist item 2 verification

- Added deterministic paired-data indexing in `src/semirestore/manifest.py` and the standalone `scripts/build_manifest.py` entry point.
- Pairing is by exact stem and rejects duplicates, missing partners, invalid/non-finite arrays, wrong dimensionality, and targets that are not exactly 2× the input shape.
- The manifest records portable relative paths, shape/dtype, per-image range/mean/std, unassigned future split fields, and SHA-256 for both files.
- The JSON audit records count, expected-count gate, ignored metadata count, shape/dtype distributions, global ranges, bytes, manifest hash, and an explicit `input_clipped_or_normalized: false` statement.
- Evidence artifacts contain no machine-specific absolute paths and are deterministic when rerun against the same files.
- Automated verification: `python -m pytest -q` → 17 passed; compile checks passed; the CLI ran from a foreign working directory.
- Organizer ZIP central-directory verification: 3,200 real NoisyLR names, 3,200 real GT names, and 3,200 exact matching filenames.
- Real-array smoke audit: six pairs spread across indices 0–3199 passed; observed input range `[-0.0822575, 1.62851]`, target range `[0,1]`, inputs `128×128`, targets `256×256`.
- The complete 3,200-file checksum manifest will be generated by the same command after the 919 MB archive is extracted in the Colab/data-staging environment; no final artifact was fabricated from ZIP metadata alone.
