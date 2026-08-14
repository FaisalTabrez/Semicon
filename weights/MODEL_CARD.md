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

## T4 benchmark

Model-only, synchronized GPU timing; excludes checkpoint loading and file I/O.

- GPU: Tesla T4
- FP32 batch-1 median / p95: 14.956 / 17.684 ms
- FP32 batch-1 peak allocated CUDA memory: 0.069 GiB
- BF16 parity status: supported

## Limitations

- Pseudo-OOD texture clusters are a robustness proxy, not organizer test-label evidence.
- Restoration can smooth fine detail or sharpen ambiguous texture.
- Output is inspection assistance only, not a manufacturing disposition decision.


## Precision decision

- FP32 median latency: 14.956 ms
- BF16 median latency: 18.129 ms
- BF16 median speed change: -21.21%
- BF16 quality-safe under the declared parity limits: True
- Submission default: **FP32**
- Decision: FP32 retained: BF16 did not satisfy both quality and 10% speed requirements.
