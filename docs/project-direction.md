# Project Direction — Pruning Use Cases

## Original goal: Ångström-class model

The project goal is to produce an Ångström-class model: one size smaller than yolo26n nano, with accuracy recovered close to baseline via fine-tuning on full COCO.
The name references the unit of length one order below nanometre. The model earns the designation through the full pipeline:
prune at ratio=0.10, fine-tune on full COCO, recover accuracy.

## Discovery: pruning as a standalone feature (ratio=0.03)

During development it became clear that at ratio=0.03 the pruned model is functional without fine-tuning — mAP50 0.81 vs baseline 0.94, a 14% degradation, deployable immediately.
This makes the pruning pipeline itself a useful standalone feature: a
user passes a model in, gets a lighter model out, no training required.

This is separate from the Ångström goal. The 0.03 output is not an Ångström model, it has not been fine-tuned and does not achieve the accuracy target. It is the pruning pipeline used as a compression utility.

## Two use cases

**Pruning as a feature (ratio=0.03, no fine-tuning)**
- Pipeline exposed as a tool: prune any compatible model on demand
- mAP50 ~0.81 untuned (14% degradation from 0.94 baseline)
- ~1.5% actual parameter reduction (outer-conv pruning only)
- No GPU required after pruning, deployable immediately
- Useful when a small accuracy trade-off is acceptable

**Ångström model (ratio=0.10, fine-tuning required)**
- Primary project goal
- Prune at 0.10, fine-tune on full COCO
- Untuned mAP50 ~0.004 — not deployable until fine-tuned
- Expected to recover close to baseline after fine-tuning
- ~4.5% actual parameter reduction
- Result is a named model one size class below nano

## Order of work

1. Run the Kaggle fine-tuning notebook on the ratio=0.10 artifact.
2. Confirm accuracy recovery on full COCO val.
3. If recovery holds, publish as the Ångström model.
4. Document the 0.03 pruning feature separately as a utility, not a model release.

## Open questions

- After fine-tuning at 0.10, how close to baseline does it actually recover?
  That result determines whether the Ångström designation is justified.
- Does the 0.03 feature hold on full COCO val, or is coco8 too small to trust 0.81?