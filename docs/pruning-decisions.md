# Pruning Ratio — Decision Log

## Initial default: 0.30

The pipeline launched with `DEFAULT_RATIO = 0.30`, removing 30% of tracked output channels per layer. Evaluation on coco8 after pruning (no fine-tuning) produced mAP50 = 0.0 across all tested configurations. The model emitted bounding boxes but all confidence scores were in the range 0.001–0.004, below NMS threshold, functionally equivalent to training from scratch with random weights.

Investigation identified two compounding causes:

-The primary cause was a structural bug: SPPF (layer 9) had a residual connection (`add=True`) that performs element-wise addition of its output `y` and its input `x`.
The pipeline was generating output masks for cv2 independently of the input mask, producing masks with equal channel counts but different indices. PyTorch accepted the addition without error because shapes matched, but the channels being summed were
semantically misaligned, corrupting all features from layer 9 onward — the entire neck and detection head.

-The secondary cause was that 30% structured pruning of a nano model with minimal redundancy destroys representational capacity even after the bug fix; the ratio sweep confirmed mAP50 remains near zero at ratio=0.10 without fine-tuning.

## A note on coco8

All evaluation figures in this document use coco8, the 8-image Ultralytics sanity dataset (`coco8.yaml`). It was chosen purely for speed — a full coco8 val run takes seconds on CPU, making it practical for ratio sweeps and quick checks during development. The numbers are not representative of real-world performance; they are
only used to confirm whether a pruned model is functional (mAP > 0) or collapsed (mAP ≈ 0).

Fine-tuning uses full COCO. coco8 appears in the fine-tuning
notebook only as a pre-train sanity check: if the loaded checkpoint scores below mAP50 = 0.001 on coco8, the run aborts immediately rather than wasting GPU hours on a broken model. The threshold is intentionally low — a 10% pruned model scores ~0.004 untuned, but still rejects a fully collapsed checkpoint (mAP50 = 0.0), such as one
produced by the SPPF bug.

## Ratio sweep after bug fix (coco8, untuned)

| Ratio | mAP50 before fix | mAP50 after fix |
|-------|-----------------|-----------------|
| 0.01  | 0.58            | 0.86            |
| 0.03  | —               | 0.81            |
| 0.05  | 0.037           | 0.28            |
| 0.10  | 0.0004          | 0.0039          |
| 0.30  | 0.0             | 0.0             |

## Revised default: 0.10, fine-tuning assumed

With fine-tuning as a mandatory post-step, the relevant criterion is not untuned mAP, but whether the pruned model retains enough structure to recover during training. At ratio=0.10 the model is damaged but structurally intact — loss decreases and mAP recovers during fine-tuning. At ratio=0.30 recovery requires significantly more epochs and may not reach original accuracy on a nano model.

`DEFAULT_RATIO` was set to 0.10. A warning is emitted for any ratio above 0.05 indicating that fine-tuning is required. The pipeline does not perform fine-tuning itself; that is handled by the separate Kaggle notebook.