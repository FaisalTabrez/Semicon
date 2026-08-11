# Deadline-Aware Development, Fine-Tuning, and Testing Roadmap

## Operating rules for the week

1. `evaluation.py` and the README command are the product; notebooks are supporting tools.
2. Every model change gets a run ID, resolved config, split hash, metrics row, runtime, and one-line conclusion.
3. Change one major variable per ablation.
4. Keep bicubic, EDSR-lite, and unconditioned NAF-SR results throughout the week.
5. Freeze the submission model before packaging; never retrain during final clean-machine verification.
6. If a gate fails, use the stated fallback immediately.

## Immediate Round 1 compression plan: 11–16 August 2026

There are only five days before the listed registration/submission deadline. Use this plan now even if the full seven-day plan below will be used during the 28 August–4 September semifinal window.

| Date | Primary outcome | Non-negotiable deliverable |
|---|---|---|
| 11 Aug | Requirements/data audit and repository skeleton | Verified dataset manifest design, bicubic baseline plan, draft slides 1–5 |
| 12 Aug | Runnable baseline and clarified submission rules | `evaluation.py` smoke path, EDSR-lite/NAF-SR starts, answers from submission Q&A |
| 13 Aug | First full metrics | One trained model, ID/OOD validation table, initial visuals |
| 14 Aug | One targeted improvement | Exactly one successful loss/augmentation/conditioning ablation |
| 15 Aug | Freeze and package | Frozen checkpoint, fresh-env pass, 400 test outputs, PDF/video draft |
| 16 Aug | Submit early and verify | Public repo and portal artifacts verified well before the unknown cutoff time |

If training slips, submit the simplest benchmarkable model with honest results. Do not trade away evaluator reliability for an unverified architecture.

## Full seven-day engineering sprint

### Day 1 — Compliance, data integrity, and zero-model baseline

**Goal:** prove the data and scoring path before training.

Tasks:

- create the planned repository structure and `.gitignore`;
- document the official download/extraction layout in `data/README.md`;
- implement discovery, pairing, `allow_pickle=False`, shape/range/finite checks, and macOS metadata filtering;
- generate `dataset_audit.json` and `manifest.csv`;
- implement bicubic inference and PSNR/SSIM/LPIPS preprocessing;
- create a texture/source-aware ID and pseudo-OOD split;
- implement fixtures and the first loader/metric tests;
- sketch slides 1–5 while decisions are fresh; and
- run the exact public test discovery and confirm 400 real inputs.

Exit gate:

- 3,200 unique paired stems and 400 unique real test stems are identified;
- no raw data is tracked by Git;
- bicubic scores are reproducible across two runs; and
- a report explicitly confirms input values were not clipped.

Fallback: if grouping is not ready, use a fixed stratified-by-statistics split for the first baseline and mark it temporary; do not use the public test set as validation.

### Day 2 — Standalone evaluator and learned baselines

**Goal:** have an end-to-end entry that could be submitted today.

Tasks:

- implement `edsr_lite.py` and unconditioned `naf_sr`;
- implement Charbonnier loss, AdamW, AMP, checkpointing, and CSV/JSON logging;
- overfit eight samples and verify loss decreases sharply;
- run a 200-step timing/memory calibration;
- train a short EDSR-lite baseline;
- implement `evaluation.py` with relative weight resolution and CPU/CUDA support;
- run the evaluator from outside the repository root; and
- ask the 12 August Q&A about slide count, video status, output format/naming, metric preprocessing, border cropping, external weight downloads, and test-time adaptation rules.

Exit gate:

- clean forward/backward pass with no NaN/Inf;
- `evaluation.py input_dir output_dir` writes correctly named `256×256` float32 outputs;
- the evaluator requires no source edit or prompt; and
- a learned baseline beats bicubic on mean validation PSNR.

Fallback: if NAF-SR is unstable, freeze EDSR-lite as the primary architecture and continue the same validation/submission work.

### Day 3 — Primary training and honest validation

**Goal:** obtain the first defensible primary-model result.

Tasks:

- train unconditioned NAF-SR with Charbonnier loss;
- validate on both ID and pseudo-OOD after fixed intervals;
- enable EMA as a separately scored checkpoint;
- measure per-image metrics and worst-decile performance;
- review at least 20 panels, biased toward failures and OOD clusters;
- record ringing, oversmoothing, checkerboards, intensity bias, or invented structures; and
- populate slides 6 and 7 with real, labeled preliminary values.

Exit gate:

- NAF-SR versus bicubic and EDSR-lite table exists on the identical split;
- best checkpoint can be loaded by a fresh process;
- no validation/test leakage; and
- one clear failure hypothesis is written for Day 4.

Fallback: if NAF-SR does not beat EDSR-lite, keep EDSR-lite and spend Day 4 on data/loss quality, not model size.

### Day 4 — Fine-tuning for structure and OOD robustness

**Goal:** improve the actual failure mode, not the training score.

Run short, controlled experiments in this order:

1. add SSIM and Sobel terms;
2. test fitted synthetic degradation at 15% of batches;
3. test statistics conditioning; and
4. only if needed, test a small FFT term or LPIPS fine-tune.

For each experiment:

- start from the same checkpoint or same seed/budget;
- change one item;
- score ID, OOD, worst decile, model size, and latency;
- inspect the same fixed image panel; and
- write `keep`, `reject`, or `inconclusive` in `reports/ablations.csv`.

Exit gate:

- at least one retained improvement has a measured pseudo-OOD benefit;
- ID regression stays within the predeclared tolerance; and
- there is no new visual evidence of false structures.

Fallback: retain the Day 3 checkpoint if no experiment clears the gate. A negative ablation is still credible evidence.

### Day 5 — Model selection, inference optimization, and stress testing

**Goal:** select the final quality/speed point and remove fragile features.

Tasks:

- rank all eligible checkpoints using the declared composite method;
- benchmark FP32 and BF16 parity; keep FP32 unless BF16 is materially faster and metric-safe;
- benchmark batch 1 and optional larger batches with synchronized GPU timing;
- test `128×128` and synthetic `256×256` inputs for dynamic 2× behavior;
- test dark, bright, low-variance, high-noise, negative-valued, and >1-valued fixtures;
- verify output range, dtype, shape, and count;
- decide explicitly whether conditioning, EMA, compilation, or TTA is in the final path; and
- freeze architecture and loss at the end of the day.

Exit gate:

- selected checkpoint, configuration, split hash, and reason are recorded;
- checkpoint meets size budget or has a verified download method;
- no default option depends on compilation cache, Internet, or a particular working directory; and
- final inference path has zero partial failures.

Fallback: remove every optional optimization and ship native FP32 PyTorch.

### Day 6 — Reproducibility, restored outputs, and repository hardening

**Goal:** make the judge's first run boring and successful.

Tasks:

- copy final checkpoint to `weights/model.pt` and generate SHA-256;
- create the exact frozen `requirements.txt` from the working environment;
- build a brand-new virtual environment from the public-style checkout;
- install only from documented instructions;
- run CPU fixture tests and CUDA inference;
- run the final evaluator on all 400 public test inputs;
- validate 400 outputs and place them in `restored_test_outputs/`;
- run secret and large-file checks before making the repository public;
- finish README setup, inference, training, data, hardware, metrics, and troubleshooting; and
- record final model size, training time, and measured local inference latency.

Exit gate:

- a clean clone can run the README inference command without contact or edits;
- all tests and the submission verifier pass;
- every required repository item is present; and
- the public repository does not contain raw training data, credentials, or dead absolute paths.

Fallback: repair documentation/evaluator defects before generating new metrics. Do not retrain.

### Day 7 — Evidence package, presentation, and final audit

**Goal:** submit a complete, internally consistent story.

Tasks:

- generate fixed before/bicubic/restored/GT panels at full view and crop scale;
- include one honest failure case and what it taught the team;
- finish nine content slides using the required template and remove its instruction slide;
- place PSNR, SSIM, LPIPS, model size, hardware, training time, and latency on slides;
- record a concise walkthrough using the exact evaluator command;
- verify the video is no longer than five minutes;
- export `TeamName_KLA_PS01.pdf` and inspect every link/page;
- verify GitHub is public and weights/output files download from a logged-out browser;
- run the checklist below; and
- upload early enough to recover from portal errors.

Exit gate:

- PDF, public repo, checkpoint, restored outputs, and video agree on model name and metrics;
- no slide contains a placeholder or unverified H100 claim;
- links work without team credentials; and
- portal confirmation is saved.

Fallback: remove unsupported claims and optional assets. Never replace real results with estimates.

## Experiment budget

Minimum set, in priority order:

1. bicubic;
2. EDSR-lite + Charbonnier;
3. NAF-SR + Charbonnier;
4. NAF-SR + structure loss;
5. best model + fitted synthetic degradation;
6. best model + statistics conditioning;
7. optional small LPIPS or FFT term; and
8. optional BF16/ONNX acceleration after quality freeze.

Stop after experiment 5 if less than two days remain. Experiments 6–8 are expendable.

## Submission checklist

### Portal and team

- [ ] Register every 2–4 member team before the 16 August deadline.
- [ ] Confirm slide count, video status, output format, and naming in the 12 August Q&A/portal.
- [ ] Save portal submission confirmation and the final uploaded PDF.

### PDF

- [ ] Use the official idea template and remove its instruction slide.
- [ ] Keep nine content slides unless the portal/Q&A explicitly overrides the track page.
- [ ] Include team, problem, idea, solution diagram, innovation, real results, feasibility, GitHub/video links, and references.
- [ ] Name the file `TeamName_KLA_PS01.pdf` under the working interpretation.
- [ ] Label preliminary versus final metrics and never claim unmeasured H100 timing.

### Public GitHub repository

- [ ] `README.md` lets a reviewer clone, install, and run inference without contacting the team.
- [ ] Top-level standalone `evaluation.py` accepts test-image and output-directory paths and needs no edits.
- [ ] `train.py` or a training notebook reproduces training from scratch.
- [ ] Final `.pt`, `.onnx`, or `.h5` weights are included or reliably downloadable and auto-located.
- [ ] `restored_test_outputs/` contains the actual final outputs.
- [ ] `requirements.txt` is the exact frozen working environment.
- [ ] All links work without credentials; Git LFS files are real content, not unresolved pointers.
- [ ] A fresh-machine run passes from the README exactly as written.
- [ ] No raw organizer training archive, API key, token, `.env`, private path, or unrelated large checkpoint is committed.

### Video

- [ ] Show the exact public evaluator command and a real directory run.
- [ ] Show quantitative results plus input/restored/GT visuals.
- [ ] State hardware, model size, and measured runtime.
- [ ] Include one limitation or failure mode.
- [ ] Keep duration at or below five minutes.
