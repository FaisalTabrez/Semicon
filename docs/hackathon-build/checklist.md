# Build Checklist

## Build Preferences

- **Build mode:** Autonomous, inferred from the explicit request to start building
- **Comprehension checks:** N/A
- **Git:** Do not commit or stage unless explicitly requested; keep each item independently testable
- **Verification:** Automated tests plus real-data smoke checks when public data is available
- **Check-in cadence:** Checkpoint after items 3, 6, and 9, or earlier if a risky assumption fails
- **Compute:** Local CPU for development/tests; Google Colab for training, AMP calibration, and GPU benchmarks
- **Wow moment:** Run the exact standalone evaluator, then show a faithful restored crop and measured quality/runtime evidence
- **Platform note:** This is an I4C project; no Devpost state or handoff actions will be created

## Checklist

- [x] **1. Build the compliance-first bicubic vertical slice**
  Spec ref: `spec.md > Standalone evaluation contract` and `spec.md > Data design`
  What to build: Scaffold the package, safe `.npy` discovery/validation, raw-range-preserving loading, bicubic 2× restoration, top-level `evaluation.py`, baseline README commands, and CPU tests.
  Acceptance: A foreign-working-directory subprocess converts valid grayscale `.npy` inputs to matching float32 `2H×2W` outputs without source edits, ignores macOS metadata, clips only the final output, and fails clearly on invalid input or unsafe overwrite.
  Verify: `python -m pytest -q` and an evaluator smoke run on a real public test sample.

- [x] **2. Build the paired-data manifest and audit**
  Spec ref: `spec.md > Data design > Discovery and validation rules`
  What to build: Pair `NoisyLR` and `GT` by exact stem, verify the 2× relation and checksums, create a deterministic manifest, and emit the dataset audit report.
  Acceptance: Current release resolves to 3,200 unique pairs; missing/duplicate/non-finite/wrong-shape samples fail with actionable paths; raw arrays remain outside Git.
  Verify: Run `scripts/build_manifest.py` against the extracted organizer training directories and inspect the generated counts/ranges.

- [x] **3. Add labeled metrics and evidence reports**
  Spec ref: `spec.md > Metrics and model selection`
  What to build: PSNR, SSIM, LPIPS preprocessing, diagnostic metrics, per-image CSV, aggregate JSON, and bicubic labeled evaluation.
  Acceptance: Metric preprocessing is fixed and tested; bicubic ID/OOD-ready evidence is reproducible; grayscale LPIPS conversion happens exactly once.
  Verify: Run metric unit tests and `evaluate_metrics.py` on fixtures plus the validation split.

- [x] **4. Implement and train EDSR-lite**
  Spec ref: `spec.md > Architecture > Mandatory baselines`
  What to build: EDSR-lite model, Charbonnier loss, checkpoint metadata, configuration, script-first training entry point, and a thin Colab launcher notebook.
  Acceptance: The model overfits eight samples, produces `2H×2W`, reloads in a fresh process, and beats bicubic on mean validation PSNR.
  Verify: Run the overfit test, short training configuration, and learned-checkpoint evaluator smoke test.

- [x] **5. Implement the NAF-SR primary model**
  Spec ref: `spec.md > Architecture > Primary model`
  What to build: NAF blocks, three-scale encoder/decoder, pixel-shuffle residual head, bicubic global skip, and self-describing checkpoint construction.
  Acceptance: Model stays within the parameter/checkpoint budget, trains without non-finite values, and is comparable to EDSR-lite on the identical split.
  Verify: Model unit tests, parameter report, short calibration run, and validation comparison.

- [x] **6. Complete the reproducible training engine**
  Spec ref: `spec.md > Training design`
  What to build: Seed control, DataLoader worker seeding, Colab-safe AMP, AdamW, scheduler, gradient clipping, EMA, resolved configuration, runtime reconnection/resume support, environment report, and best/last checkpoints.
  Acceptance: A repeated deterministic debug run matches within documented tolerance and every run records config, split hash, environment, and metrics.
  Verify: Run deterministic CPU debug training twice and compare artifacts; run a 200-step timing/memory calibration in the selected Colab GPU runtime.

- [ ] **7. Add OOD split and controlled robustness ablations**
  Spec ref: `spec.md > Data design > Split construction` and `spec.md > Phase C — OOD fine-tuning`
  What to build: Texture descriptors/clusters, grouped ID/OOD manifest assignment, D4 transforms, fitted synthetic degradation, and optional statistics conditioning.
  Acceptance: Splits preserve group integrity and every augmentation/conditioning change has an isolated keep/reject result without test-label leakage.
  Verify: Split invariants, paired-transform tests, and one-variable ablation table.

- [ ] **8. Select and benchmark the final model**
  Spec ref: `spec.md > Composite selection` and `spec.md > Performance tests`
  What to build: Composite ranking, fixed visual panel, synchronized Colab GPU latency benchmark, FP16/BF16 parity checks where supported, checkpoint hash, and model card.
  Acceptance: The final checkpoint has an auditable selection reason, measured local latency/VRAM/size, and no unsupported H100 claim.
  Verify: Run `scripts/benchmark.py`, validate the model card values, and inspect worst-decile panels.

- [ ] **9. Harden the public repository and generate test outputs**
  Spec ref: `spec.md > Repository structure` and `spec.md > Verification plan`
  What to build: Final frozen Colab training environment, CPU clean-clone verifier, CI smoke test, complete README, downloadable weights, and restored public test outputs.
  Acceptance: A clean environment runs the documented command without edits; 400 outputs have matching stems, float32 dtype, `256×256` shape, finite `[0,1]` values, and no missing files.
  Verify: Run `scripts/verify_submission.py` from a clean checkout and verify large files/links without credentials.

- [ ] **10. Prepare the I4C submission evidence handoff**
  Spec ref: `spec.md > Demo and submission flow` and `roadmap.md > Submission checklist`
  What to build: Nine-slide evidence map, figures, ablation/results tables, honest failure case, public repository link, and ≤5-minute demo outline.
  Acceptance: PDF, repository, checkpoint, outputs, metrics, and video use the same final model name/numbers and contain no placeholders or estimates presented as measurements.
  Verify: Complete the submission checklist and perform a logged-out link review before portal upload.
