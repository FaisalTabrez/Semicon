# Width-32 student ablation

The frozen 9.11M-parameter conditioned NAF-SR teacher was compared with two
matched 4.11M-parameter students on an NVIDIA A100-SXM4-40GB. Both students use
the same depth as the teacher. The distilled run used 70% ground-truth
Charbonnier loss and 30% frozen-teacher output loss.

| Candidate | Batch-1 latency change | Batch-16 throughput change | ID PSNR delta | OOD PSNR delta | Decision |
|---|---:|---:|---:|---:|---|
| Supervised width-32 | +4.61% faster | +36.94% | -0.124526 dB | -0.131782 dB | Reject |
| Distilled width-32 | +3.99% faster | +36.85% | -0.090543 dB | -0.097410 dB | Reject |

Distillation improved every locked quality metric relative to the supervised
student, but the distilled model still changed ID/OOD SSIM by
`-0.002645/-0.004487` and LPIPS-Alex by `+0.007476/+0.013452` versus the
teacher. It also missed the required 25% batch-1 latency reduction because
width compression retains the same sequence of kernel launches. The teacher
therefore remains the packaged model; this ablation is not used to claim a
student deployment improvement.
