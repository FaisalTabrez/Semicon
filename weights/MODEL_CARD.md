# Semiconductor Restoration Model Card

## Intended use

Restore one-channel degraded semiconductor-inspection `.npy` arrays by 2×. The evaluator writes float32 outputs clipped to `[0,1]`.

## Final checkpoint

- Model: statistics-conditioned NAF-SR; real paired data plus D4 geometry
- Checkpoint SHA-256: `273abd9d6dcfa9bdee71ac15016994962304b6c9d902898b4f4d503bed158c28`
- Parameters: 9,111,684
- Compact checkpoint size: 34.87 MiB
- Training configuration: `configs/final_conditioned.yaml`
- Inference default: native PyTorch FP32

## Evidence policy

The final checkpoint was selected from validation-ID and pseudo-OOD summaries using `scripts/select_final_model.py`. The six-metric rank averages tied with the real-pairs+D4 baseline, and the predeclared pseudo-OOD tie-break selected conditioning.

- Validation-ID: PSNR 29.226240, SSIM 0.776572, LPIPS-Alex 0.271442
- Pseudo-OOD: PSNR 25.251257, SSIM 0.666073, LPIPS-Alex 0.353080
- Tesla T4 FP32 model-only latency: 14.956 ms median, 17.684 ms p95, 0.0688 GiB peak allocated CUDA memory
- BF16 passed quality parity but was 21.21% slower, so it is not the default.

## Limitations

- Texture pseudo-OOD is a robustness proxy, not organizer test-label evidence.
- Restoration can oversmooth fine features or make uncertain structures appear sharper.
- Outputs support inspection assistance only; they are not a manufacturing disposition or defect decision.
