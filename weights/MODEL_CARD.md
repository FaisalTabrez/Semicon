# Semiconductor Restoration Model Card

This file is populated only after final checkpoint selection and benchmark evidence is generated.

## Intended use

Restore one-channel degraded semiconductor-inspection `.npy` arrays by 2×. The evaluator writes float32 outputs clipped to `[0,1]`.

## Final checkpoint

- Model: pending final artifact copy
- Checkpoint SHA-256: pending
- Training configuration: `configs/final_conditioned.yaml`
- Inference default: native PyTorch FP32

## Evidence policy

The final checkpoint must be selected from validation-ID and pseudo-OOD summaries using `scripts/select_final_model.py`. Performance, latency, memory, and file size values must be copied from generated JSON artifacts; no estimates belong in this card.

## Limitations

- Texture pseudo-OOD is a robustness proxy, not organizer test-label evidence.
- Restoration can oversmooth fine features or make uncertain structures appear sharper.
- Outputs support inspection assistance only; they are not a manufacturing disposition or defect decision.
