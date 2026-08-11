# Feasibility Study — KLA PS01 Semiconductor Image Restoration

**Event:** SEMICON India Hackathon 2026

**Problem statement:** AI-Based Restoration of Degraded Images for Semiconductor Inspection

**Assessment date:** 11 August 2026 (IST)

**Decision:** **Feasible in one focused GPU week, with a submission-ready baseline required by Day 2 and a hard model freeze before packaging.**

## 1. Executive assessment

The task is a supervised, single-image, 2× restoration problem rather than an open-ended generative problem. The released training data contains 3,200 aligned pairs, every currently released degraded input is `128×128` float32, and every ground truth is `256×256` float32. The released test set contains 400 degraded `128×128` arrays. This is a tractable scale for a compact convolutional restoration model and several controlled ablations on one modern CUDA GPU.

The main difficulty is not fitting the training set. It is generalizing to images from unseen sources while removing mixed degradations without erasing or inventing fine semiconductor structures. The project should therefore optimize for:

1. a reliable standalone evaluator before model novelty;
2. a source-aware or texture-clustered validation split;
3. conservative pixel/structure losses rather than adversarial hallucination;
4. a small, fast model whose full checkpoint can live in the public repository; and
5. reproducible evidence: PSNR, SSIM, LPIPS, runtime, ablations, and before/after/ground-truth panels.

## 2. Verified challenge facts

### 2.1 Task and data requirements

The [official I4C challenge page](https://i4c.in/hackathon-2026/) states that the solution must:

- process grayscale images;
- jointly handle speckle noise, Gaussian degradation, and 2× spatial-resolution loss;
- accept degraded inputs whose intensity may fall outside the ground-truth `[0, 1]` range;
- restore `128×128 → 256×256` and potentially `256×256 → 512×512` pairs;
- generalize to both in-distribution and out-of-distribution test sources; and
- remain fast enough for inference benchmarking on an NVIDIA H100.

The KLA problem deck adds two important points: degradation operations may appear in any order, and evaluation considers the quality of the data, model, loss design, and compute/training hygiene—not only a leaderboard number.

### 2.2 Audit of the currently linked public dataset

The linked Google Drive folder was inspected without committing the archives to this repository.

| Artifact | Observed contents | Practical consequence |
|---|---:|---|
| `train.zip` | 918,994,209 bytes; 3,200 `train/NoisyLR/*.npy` and 3,200 matching `train/GT/*.npy` files | Enough aligned data for supervised training and ablation work |
| Training input | 3,200 float32 arrays, all `128×128` | Current public task is consistently 2× SR |
| Training target | 3,200 float32 arrays, all `256×256` | Output head can be fixed at 2× while keeping spatial dimensions dynamic |
| `Test_NoisyLR.zip` | 23,419,125 bytes; 400 real `NoisyLR/*.npy` arrays | Final output generation is small and quick |
| Test input | 400 float32 arrays, all `128×128`; observed global range `[-0.22488, 2.15802]` | Never clip or cast the input before the model |
| Archive noise | Duplicate `__MACOSX/` and `._*` metadata entries | Dataset discovery must explicitly ignore these paths |

Six training pairs sampled across the index range confirmed ground truth in `[0,1]` and degraded maxima as high as `1.63`; this is a sample check, not a claim about the full training archive extrema.

### 2.3 Submission and schedule facts

The official page currently lists:

- registration and initial submission deadline: **16 August 2026**;
- Round 1 evaluation: **17–26 August 2026**;
- top-30 announcement: **27 August 2026**;
- Round 2 build period: **28 August–4 September 2026**;
- team eligibility: 2–4 undergraduate, graduate, postgraduate, PhD, or research-scholar members in any stream; and
- required portal artifacts: a PDF based on the idea template and a public GitHub repository. A demo video is described as optional in the track-specific section but required, with a five-minute maximum, in the general submission section. The safest course is to include one.

Because this assessment is dated 11 August, a seven-day sprint does not fit before the Round 1 deadline. The accompanying roadmap therefore includes a five-day submission-safe compression plan and a full seven-day engineering plan suitable for the semifinal window.

### 2.4 Requirement conflicts to resolve on 12 August

The live page and downloadable template are not fully consistent:

- the problem-specific page requires 8–9 slides after removing instructions, while text inside the downloaded template says 6–7 slides including title;
- the track section calls the video optional but recommended, while the general section lists a video as something participants need to submit; and
- the template uses `Team Name_PSNo`, while the track section gives `TeamName_KLA_PS01.pdf`.

Use the more specific track rules as the working interpretation: nine content slides, `TeamName_KLA_PS01.pdf`, and a video no longer than five minutes. Confirm these three points during the 12 August submission Q&A or in the portal before upload.

## 3. Technical feasibility

| Dimension | Rating | Evidence and condition |
|---|---|---|
| Data volume | Green | 3,200 aligned pairs are enough for a compact supervised model, patch sampling, and controlled validation. |
| Data quality | Amber | Alignment and range are favorable, but source labels and the exact forward-degradation parameters are absent. |
| Model complexity | Green | A 6–12M parameter CNN can jointly denoise and upscale at this resolution. |
| OOD generalization | Amber | Explicit pseudo-OOD validation and degradation augmentation are required; random splitting alone is inadequate. |
| Compute | Green with one CUDA GPU | Run a 200-step calibration before committing to epoch counts. CPU-only tuning is unlikely to support multiple ablations in a week. |
| Inference | Green | A compact low-resolution backbone plus one pixel-shuffle head should be far below the organizer's ten-second example on an H100; this remains an estimate until benchmarked. |
| Reproducibility | Green | `.npy` inputs, fixed shapes, deterministic manifests, and a small checkpoint make a standalone evaluator straightforward. |
| Immediate schedule | Amber/Red | Only five calendar days remain before Round 1; novelty must not delay a working evaluator, baseline, and PDF. |

## 4. Candidate approach comparison

| Approach | Quality potential | Speed | One-week risk | Decision |
|---|---:|---:|---:|---|
| Bicubic interpolation | Low | Excellent | Very low | Mandatory lower-bound baseline only |
| EDSR-lite residual CNN | Good for 2× SR; weaker for unknown mixed noise | Excellent | Low | Strong fallback model |
| SwinIR-lite | High | Good | Medium | Useful second model only if baseline work finishes early |
| Restormer variant | High for restoration and long-range context | Moderate | Medium/high | Too much tuning surface for the primary one-week path |
| **Conditioned NAFNet-SR-Lite** | **High expected quality/efficiency balance** | **Excellent** | **Low/medium** | **Recommended primary model** |
| GAN or diffusion SR | High perceptual sharpness, uncertain fidelity | Poor/moderate | High | Reject for the primary entry: hallucination and runtime are poor fits for inspection evidence |

NAFNet is a computationally efficient restoration baseline with simple blocks, while EDSR supplies a proven residual/pixel-shuffle SR pattern. SwinIR and Restormer remain useful references or fallback experiments, but their larger tuning surface is not justified until the compliance path works. See the research links in `spec.md`.

## 5. Recommended solution

Build **Conditioned NAFNet-SR-Lite**, a single grayscale network that operates mostly at low resolution and predicts a high-resolution residual:

- one-channel raw input; no input clipping;
- shallow convolutional stem;
- compact NAF encoder/bottleneck/decoder at low resolution;
- optional per-image degradation statistics encoded into light FiLM-style scale/shift parameters;
- four-channel reconstruction head followed by `PixelShuffle(2)`;
- a bicubic 2× skip connection so the network learns only the missing clean detail; and
- output clamping to `[0,1]` only after inference, when writing or scoring the restored result.

The conditioning branch is an ablation, not a dependency. The unconditioned model must train and run first. If conditioning does not improve the pseudo-OOD holdout, remove it.

## 6. Training and validation feasibility

### Split design

A random file split may place visually related structures in both train and validation sets and overstate generalization. Create a reproducible manifest and use:

- approximately 70% training;
- approximately 15% validation-ID sampled from the same texture clusters as training; and
- approximately 15% pseudo-OOD validation made from entire held-out texture clusters.

If KLA provides source labels, use them instead of clustering. Without labels, cluster inexpensive descriptors such as mean, standard deviation, gradient energy, entropy, and radial FFT energy. Store every stem and split in `data/splits/manifest.csv`.

### Augmentation policy

Use paired D4 geometry (horizontal/vertical flips and 90-degree rotations) from the first run. Add synthetic degradation only after estimating its plausible range from the training pairs:

- multiplicative speckle;
- additive Gaussian noise;
- mild Gaussian blur;
- 2× area/bicubic downsampling; and
- randomized degradation order, matching the problem deck's warning.

Generate synthetic LR from GT for only a configurable fraction of batches. Aggressive, unverified augmentation can create a domain farther from KLA's test data and should be rejected if pseudo-OOD performance falls.

### Loss and selection

Start with a stable fidelity loss and add terms one at a time:

`L = 1.0 × Charbonnier + 0.15 × (1 − SSIM) + 0.05 × Sobel-L1 + 0.01 × FFT-magnitude-L1`

Run the first baseline with Charbonnier alone. Add the other terms only through ablation. A small LPIPS loss may be tested in the final fine-tuning stage, but it must be retained only if LPIPS improves without unacceptable PSNR/SSIM loss or visible invented structures.

Select a checkpoint by a predeclared composite rank across PSNR (higher), SSIM (higher), LPIPS (lower), and pseudo-OOD performance. Do not select from the unlabeled public test inputs.

## 7. Success gates

These are internal gates, not organizer-published scoring thresholds.

### Minimum viable entry

- standalone `evaluation.py` processes a fresh directory without edits;
- all 400 public test inputs produce one matching `256×256` float32 output;
- output contains no NaN/Inf and lies in `[0,1]` after final clipping;
- model beats bicubic on all three official validation metrics;
- public repository clone works in a clean environment; and
- required checkpoint and restored outputs are downloadable.

### Competitive target

- at least +2 dB mean PSNR over bicubic on both ID and pseudo-OOD validation, or an evidence-backed explanation if the dataset's baseline is unusually strong;
- measurable SSIM improvement and at least 15% relative LPIPS reduction versus bicubic;
- no major tail collapse: report the worst decile, not only the mean;
- checkpoint below 60 MB and model below roughly 12M parameters; and
- default single-image inference comfortably below one second on the team's development GPU, with H100 time reported only if actually measured.

## 8. Principal risks and mitigations

| Risk | Probability / impact | Early signal | Mitigation / fallback |
|---|---|---|---|
| Random split leakage | High / high | Very high random-val score, weak texture-holdout score | Use cluster/source holdout and record manifest |
| Input clipping destroys signal | Medium / high | Saturated bright structures, biased intensity | Keep raw float input; clamp only final output |
| Perceptual loss hallucinates edges | Medium / high | Better LPIPS but worse PSNR or false line structures | Keep perceptual weight tiny or remove it; no GAN |
| Synthetic degradation is unrealistic | Medium / medium | Augmented model loses on real validation | Estimate ranges from real pairs; cap synthetic batch fraction |
| Novel branch delays completion | Medium / high | No stable run by end of Day 2 | Freeze EDSR-lite/NAFNet baseline and drop conditioning |
| Weight path or CLI fails on judge machine | Medium / critical | Works only from repository root or with manual edit | Resolve paths relative to script; clean-env subprocess test |
| Dependency mismatch | Medium / high | install failure or metric drift | Freeze the actual working environment; document Python/CUDA |
| Large files missing from clone | Medium / critical | pointer file instead of checkpoint | Keep model under GitHub's normal limit or verify Git LFS clone |
| Slide/video rule conflict | High / medium | Portal rejects format or judge misses evidence | Confirm in 12 August Q&A; prepare the conservative nine-slide + ≤5-minute package |

## 9. Go/no-go decision

Proceed if the team has access to one CUDA GPU for at least several multi-hour windows and can allocate one member to reproducibility/submission while another tunes the model. If GPU access is absent, submit a compact EDSR-lite or U-Net residual baseline with honest results rather than attempting a large transformer.

The project should be stopped or radically simplified if a clean-machine evaluator is not working by the end of Day 2. A benchmarkable baseline can win consideration; an ambitious but unrunnable model cannot be scored.
