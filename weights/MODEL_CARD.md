# Semiconductor Restoration Model Card

## Final model

- Model: conditioned NAF-SR, real paired training plus D4 geometry
- Synthetic degradation: disabled after the controlled 15% ablation was rejected
- Checkpoint SHA-256: `273abd9d6dcfa9bdee71ac15016994962304b6c9d902898b4f4d503bed158c28`
- Parameters: 9,111,684
- Checkpoint size: 34.87 MiB
- Training configuration: `configs/final_conditioned.yaml`
- Default inference precision: FP32

## Selection

The predeclared six-metric rank policy selected `conditioned_naf_sr`.
The policy ranks PSNR (higher), SSIM (higher), and LPIPS-Alex (lower) on both validation-ID and pseudo-OOD, averages the six ranks, and breaks ties in favour of pseudo-OOD performance.

- Validation-ID: PSNR 29.226240, SSIM 0.776572, LPIPS 0.271442
- Pseudo-OOD: PSNR 25.251257, SSIM 0.666073, LPIPS 0.353080

## Runtime benchmarks

Model-only, synchronized GPU timing; excludes checkpoint loading and file I/O.

- Tesla T4 eager FP32, batch 1 at 128×128: 14.956 ms median,
  17.684 ms p95, and 0.069 GiB peak allocated memory.
- NVIDIA A100-SXM4-40GB eager FP32, batch 1 at 128×128: 14.871 ms
  median and 67.25 images/s.
- NVIDIA A100-SXM4-40GB eager FP32, batch 16 at 128×128: 542.61
  images/s. This is the portable directory-inference default.
- A100 compiled FP32 channels-last, batch 1: 1.890 ms median after a
  20–23 second first-shape compilation.
- A100 compiled FP16 channels-last, batch 16: 1,958.49 images/s after
  compilation.

Compiled FP32 and FP16 both passed locked validation-ID and pseudo-OOD quality
parity. They are optional persistent-service modes, not the cold-start evaluator
default.

## External validation

The frozen checkpoint was evaluated without tuning on all 4,591 images in the
Carinthia SEM Defect Dataset under deterministic controlled degradations.
Relative to bicubic, NAF-SR gained 8.15598 dB under the low-noise organizer
profile and 4.09757 dB under the high-noise profile, but lost 2.17861 dB for
downsampling-only inputs. This is external-domain controlled-degradation
evidence, not native aligned LR/HR evidence.

## Limitations

- Pseudo-OOD texture clusters are a robustness proxy, not organizer test-label evidence.
- Carinthia supplies real SEM references but no native aligned degraded/clean pairs.
- The clean-input external gate failed: route already-clean inputs around the model.
- Restoration can smooth fine detail or sharpen ambiguous texture.
- Output is inspection assistance only, not a manufacturing disposition decision.


## Precision decision

- FP32 median latency: 14.956 ms
- BF16 median latency: 18.129 ms
- BF16 median speed change: -21.21%
- BF16 quality-safe under the declared parity limits: True
- Submission default: **FP32**
- Decision: FP32 retained: BF16 did not satisfy both quality and 10% speed requirements.

The last three numbers above are the original Tesla T4 precision decision.
On A100, FP16 is useful only for compiled, batched throughput; eager FP32 remains
the portable cold-start default.
