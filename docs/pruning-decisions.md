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
notebook only as a pre-train sanity check: if the loaded checkpoint scores below mAP50 = 0.001 on coco8, the run aborts immediately rather than wasting GPU hours on a broken model. The threshold is intentionally low — a 10% uniformly pruned model scores ~0.004 untuned, while the default depth allocation scores ~0.45, but both still reject a fully collapsed checkpoint (mAP50 = 0.0), such as one
produced by the SPPF bug.

## Ratio sweep after bug fix (coco8, untuned, uniform layer ratios)

| Ratio | mAP50 before fix | mAP50 after fix |
|-------|-----------------|-----------------|
| 0.01  | 0.58            | 0.86            |
| 0.03  | —               | 0.81            |
| 0.05  | 0.037           | 0.28            |
| 0.10  | 0.0004          | 0.0039          |
| 0.30  | 0.0             | 0.0             |

## Revised default: 0.10, fine-tuning assumed

With fine-tuning as a mandatory post-step, the relevant criterion is not untuned mAP, but whether the pruned model retains enough structure to recover during training. At ratio=0.10 the model is damaged but structurally intact — loss decreases and mAP recovers during fine-tuning. At ratio=0.30 recovery requires significantly more epochs and may not reach original accuracy on a nano model.

`DEFAULT_RATIO` was set to 0.10. `DEFAULT_LAYER_RATIOS` was later set to `depth` after the layer-ratio A/B below. A warning is emitted for any ratio above 0.05 indicating that fine-tuning is required. The pipeline does not perform fine-tuning itself; that is handled by the separate Kaggle notebook.

## Scoring methods

The mask generator keeps the top `(1 - ratio)` channels per output conv. Three
scoring methods are available via `scoring=` or `--scoring`:

- `l1` (default): `|W|.sum(dim=(1, 2, 3))` per filter. No calibration data.
- `bn`: `|gamma|` of the BatchNorm that follows the conv.
- `bn_l1`: `|gamma| * |W|.sum(dim=(1, 2, 3))` (Network Slimming).

`bn` and `bn_l1` fall back to L1 when the conv has no BatchNorm (for example the
bare convolutions in `Detect`).

A/B on coco8 with uniform layer ratios (4 train and 4 val images, 1 epoch
fine-tune, CPU):

| ratio | method | params    | untuned mAP50 | 1-epoch mAP50 |
|-------|--------|-----------|---------------|---------------|
| 0.05  | l1     | 2,444,808 | 0.2845        | 0.2954        |
| 0.05  | bn     | 2,444,808 | 0.0184        | 0.0332        |
| 0.05  | bn_l1  | 2,444,808 | 0.1658        | 0.1658        |
| 0.10  | l1     | 2,332,312 | 0.0039        | 0.0045        |
| 0.10  | bn     | 2,332,312 | 0.0079        | 0.0069        |
| 0.10  | bn_l1  | 2,332,312 | 0.0000        | 0.0000        |

L1 beats both alternatives at ratio 0.05, where the model is still functional.
At ratio 0.10 all three collapse under uniform allocation, so the difference
there is not meaningful. Parameter counts are identical per ratio because all
methods keep the same number of channels; only the selection differs.

Decision: keep `l1` as the default. `bn` and `bn_l1` stay available behind the
flag for further experiments, but neither is a demonstrated improvement on this
test.

## Layer ratio allocation

`layer_ratios` controls how the ratio is distributed across the network:

- `uniform`: apply the same ratio to every tracked output convolution.
- `depth`: exempt the first tracked convolution and the Detect head, then ramp
  the ratio linearly from 0 at the first tracked convolution to `ratio` at the
  deepest.

A/B on coco8 with L1 scoring (4 train and 4 val images, 1 epoch fine-tune, CPU):

| ratio | layer ratios | params    | untuned mAP50 | 1-epoch mAP50 |
|-------|--------------|-----------|---------------|---------------|
| 0.05  | uniform      | 2,444,808 | 0.2845        | 0.2954        |
| 0.05  | depth        | 2,506,030 | 0.8419        | 0.8471        |
| 0.10  | uniform      | 2,332,312 | 0.0039        | 0.0045        |
| 0.10  | depth        | 2,447,937 | 0.4529        | 0.5087        |

`depth` is a large win at both ratios. At 0.10 it lifts untuned mAP50 from
0.0039 to 0.4529 and the 1-epoch result from 0.0045 to 0.5087. `bn_l1` with
`depth` also lost to L1 (untuned 0.4800 at 0.05, 0.1826 at 0.10), so the scorer
decision is independent of the layer-ratio mode.

Tradeoff: `depth` prunes fewer parameters because it protects early layers. At
0.10 it removes about 124k parameters (4.8%, 2,572,280 to 2,447,937) versus
about 240k (9.3%) for `uniform` (to 2,332,312). The accuracy gain is much
larger than the extra parameters kept.

Decision: set `DEFAULT_LAYER_RATIOS = "depth"`. `uniform` stays available via
`--layer-ratios uniform`. coco8 is tiny and one epoch is a weak fine-tune, so
this is a directional result, not a final benchmark.
