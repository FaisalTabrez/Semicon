# Carinthia external-domain evidence

## Claim boundary

This is **external-domain controlled-degradation validation**, not native paired
restoration validation. Carinthia provides real semiconductor SEM reference
images but does not provide aligned degraded/clean LR/HR pairs. The model was
not trained, selected, or tuned on Carinthia before this cold evaluation.

## Provenance

- Dataset: Carinthia SEM Defect Dataset
- DOI: `10.5281/zenodo.10715190`
- Record: <https://zenodo.org/records/10715190>
- License: CC BY 4.0
- Images: 4,591 grayscale JPEGs across six defect classes
- `data.zip` MD5: `457011cf9063e5a49751f33ea468309d`
- Evaluation protocol commit: `6d53e00`
- Frozen model SHA-256: `273abd9d6dcfa9bdee71ac15016994962304b6c9d902898b4f4d503bed158c28`
- Seed: `2026`

Each decoded reference was reduced by 2x using antialiased bicubic sampling.
The two noisy conditions add deterministic per-image Gaussian and speckle noise
using the low and high endpoints fitted only from the organizer training split.
No degraded input was clipped before restoration. Metrics clamp prediction and
reference to `[0,1]` only at the scoring boundary.

## Full-dataset results

| Severity | Method | PSNR (dB) | SSIM | LPIPS-Alex |
|---|---:|---:|---:|---:|
| Downsample only | Bicubic | 40.97466 | 0.976963 | 0.124327 |
| Downsample only | NAF-SR | 38.79605 | 0.943228 | 0.404973 |
| Profile low | Bicubic | 30.16508 | 0.591593 | 0.440717 |
| Profile low | NAF-SR | 38.32106 | 0.939257 | 0.386178 |
| Profile high | Bicubic | 23.26672 | 0.236547 | 0.616874 |
| Profile high | NAF-SR | 27.36429 | 0.468892 | 0.497169 |

## Paired NAF-SR improvement over bicubic

| Severity | Delta PSNR (95% CI) | Delta SSIM (95% CI) | Delta LPIPS (95% CI) | PSNR / SSIM win rate |
|---|---:|---:|---:|---:|
| Downsample only | -2.17861 `[-2.19371, -2.16480]` | -0.033734 `[-0.033967, -0.033493]` | +0.280647 `[+0.278357, +0.283281]` | 0% / 0% |
| Profile low | +8.15598 `[+8.13635, +8.17655]` | +0.347664 `[+0.347049, +0.348237]` | -0.054539 `[-0.056523, -0.052587]` | 100% / 100% |
| Profile high | +4.09757 `[+4.08903, +4.10614]` | +0.232346 `[+0.231815, +0.232901]` | -0.119706 `[-0.120543, -0.118897]` | 100% / 100% |

LPIPS win rates were 0%, 82.03%, and 99.98% for downsample-only,
profile-low, and profile-high respectively. All intervals are 1,000-sample
bootstrap intervals over the 4,591 paired images.

## Interpretation

The frozen model transfers strongly when an external real SEM image is exposed
to noise within the organizer-derived degradation envelope. It is deliberately
conservative to describe it as a noise-aware restoration model: the
downsample-only control shows that it should not replace bicubic interpolation
for already-clean inputs. This limitation remains visible in the evidence and
Carinthia will not be reused to tune a clean/noisy router while retaining the
claim that this run was a cold external test.

## Best/worst visual audit

The pre-registered panel selected the minimum and maximum paired PSNR delta
within each severity. No obvious checkerboard pattern, invented periodic line,
or strong edge ringing was observed. Macro-scale defect locations, silhouettes,
and high-contrast boundaries remained recognizable in the noisy conditions.

The audit also exposes systematic over-smoothing. On downsample-only inputs the
model suppresses genuine background texture and broadens local contrast around
defects, consistent with the negative PSNR/SSIM/LPIPS result. Fine surface
striations are also attenuated in the worst profile-low and profile-high cases,
even though their macro morphology and all three aggregate metrics improve.
Accordingly:

- downsample-only visual gate: **fail**;
- profile-low/profile-high macro-morphology gate: **conditional pass**;
- universal fine-feature-preservation claim: **not supported**;
- recommended scope: noisy/degraded inspection restoration with a future
  independently calibrated clean-input bypass.

The full audit panel and its exact selected-image metadata are retained in the
persistent experiment artifacts. Carinthia is not reused to calibrate the
bypass or retrain the model, preserving the cold-test status of this evidence.
