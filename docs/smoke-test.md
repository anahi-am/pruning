# Fine-Tuning Smoke Test — Report and Plan

## Summary

The smoke test ran end-to-end (5 epochs, val2017 used as both train and val) and the training loop is healthy: losses decreased and mAP50 rose monotonically every epoch.

However, the run did **not** train the pruned model. `YOLO.train()` rebuilds the network from `model.yaml`, which still describes the original architecture, so it reconstructed the unpruned 2,572,280-parameter network and transferred only 572/708 tensors. The fine-tuned artifact was therefore the original architecture with 136 randomly initialized layers, which is why mAP stayed near zero.

The HuggingFace artifact itself is correct: it is byte-identical to the local file (sha256 `70dc0ac4…`), loads at 2,180,062 fused parameters, and scores mAP50 0.0039 on coco8. The upload is not the problem.

More epochs will not help until the rebuild is bypassed.

## Smoke test results

Config: 5 epochs, 5,000 val2017 images as train and val, batch 16, imgsz 640, Tesla T4, ~10.5 minutes. Ultralytics selected `AdamW(lr=0.000119)` via `optimizer=auto` (the configured `lr0=0.001` was ignored).

| Epoch | box_loss | cls_loss | l1_loss | mAP50 |
|-------|----------|----------|---------|-------|
| 1     | 3.469    | 6.726    | 0.0461  | 0.00012 |
| 2     | 3.063    | 5.620    | 0.0358  | 0.00036 |
| 3     | 2.882    | 5.431    | 0.0311  | 0.00073 |
| 4     | 2.757    | 5.330    | 0.0289  | 0.00142 |
| 5     | 2.692    | 5.277    | 0.0274  | 0.00192 |

Final `best.pt`: mAP50 0.00191, mAP50-95 0.00063.

The trend is monotonically upward, but the gain decelerates (3.0x, 2.0x, 1.9x, 1.35x per epoch) and the absolute level is near zero even though train and val are the same images. The loop learns; the model is wrong.

## Root cause

`model.yaml` stores the original channel widths (64/128/256/512/1024), while the pruned convs are 14/28/57/115/230. The checkpoint carries the pruned module, so `.val()` works, but training does not use it.

`YOLO.train()` at `ultralytics/engine/model.py:895` does:

```python
self.trainer.model = self.trainer.get_model(weights=self.model, cfg=self.model.yaml)
```

`DetectionTrainer.get_model` builds `DetectionModel(cfg)` from the original yaml and then `load()`s the pruned weights, intersecting only matching shapes (`ultralytics/nn/tasks.py:379`).

Evidence:

- pruned module: 2,332,312 parameters
- rebuilt from `model.yaml`: 2,572,280 parameters
- `Transferred 572/708 items from pretrained weights`

## Verified bypass

Assigning the loaded module to the trainer before training keeps the pruned architecture. `DetectionTrainer.setup_model` returns early when `self.model` is an `nn.Module` (`ultralytics/engine/trainer.py:897`):

```python
trainer = DetectionTrainer(overrides={...})
trainer.model = model.model
trainer.train()
```

Verified at setup: the trainer keeps a `DetectionModel` with 2,332,312 parameters.

## Progress

- Step 1 (trainer bypass) implemented in `src/fine_tuning.py` and `src/fine_tuning_smoke.py`.
- Step 2 (parameter-count guard) implemented: both scripts assert the trainer kept the loaded pruned model.
- Step 3 (1-epoch coco8 check) passed on CPU: the trainer kept 2,332,312 parameters, training completed, and `best.pt` validated as 2,180,062 fused parameters.
- Steps 4-7 still require the Colab/Kaggle GPU runs.
- Note: pretrained Ultralytics checkpoints load with `requires_grad=False` (`yolo26n.pt` is 0/366); the scripts now call `requires_grad_(True)` before training to avoid the trainer re-enabling every layer with a warning.

## Plan

1. Replace `model.train(...)` with the `DetectionTrainer` bypass in `src/fine_tuning.py` and `src/fine_tuning_smoke.py`.
2. Add an assertion that the trainer model has the expected parameter count (2,332,312 unfused) so a future yaml rebuild fails fast.
3. Verify end-to-end with a 1-epoch coco8 run: confirm the parameter count stays 2,332,312 and the `Transferred 572/708` message no longer appears.
4. Re-run the smoke test (5 epochs, val2017) and confirm the pruned model trains, then record the new curve.
5. Set `optimizer` explicitly (e.g. `AdamW`) so `lr0`/`lrf` are honored instead of auto-selection.
6. Once the smoke test validates the pruned model, run the full Kaggle fine-tune. Expect more than 10 epochs given the 10% pruning starting point.
7. Record final numbers in `docs/pruning-decisions.md`.

## Open questions

- Keep the trainer bypass as the supported fine-tuning path, or also make the pruning pipeline emit a pruned `model.yaml`? The latter is unreliable because C3k2 internal channels are pruned arbitrarily and cannot be reconstructed from the yaml alone.
- The pre-training sanity check (coco8, threshold 0.001) does not catch architecture/weight mismatch; a parameter-count assertion would.
