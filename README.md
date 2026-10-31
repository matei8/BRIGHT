# BRIGHT Baseline — U-Net (Variant A)

Clean, from-our-own-code implementation of the early-fusion U-Net baseline for
multimodal building-damage segmentation on the **BRIGHT** dataset (pre-event optical
RGB + post-event SAR → per-pixel `background / intact / damaged / destroyed`). This is
the reference every later experiment is measured against:

| ID | Change | Why |
|----|--------|-----|
| **A (current stage)** | U-Net, early 4-channel fusion | reference point |
| B | two-stream encoders (optical / SAR) + concat fusion | the two sensors look very different |
| C | cross-attention fusion | SAR doesn't align pixel-perfectly with optical |
| D | decoupled localization + damage heads | best benchmark models do this |
| E4 | class-weighted loss for "Damaged" | minority class, baseline's weakest (~39% IoU) |
| E5 | encoder initialization (scratch vs pretrained) | ImageNet should help optical, little for SAR |
| E6 | cross-event transfer | zero-shot mIoU is <40% for every benchmark model |

## Decision: build it here, mirroring the official repo

> **Build our own clean U-Net in this `codebase/`, implementing the exact
> architecture and training recipe used in the official BRIGHT repo (which *is* the
> paper's U-Net) — rather than running the repo's research scripts in place, or
> inventing a different U-Net.** Train on the **local GPU** with a
> **reduced schedule**, using a **4-channel early-fusion input** (optical RGB + a single
> SAR band).

**Why not just run the repo.** `ProjectBaseline/BRIGHT/bda_benchmark/script/standard_ML/train_UNet.py`
is research-grade: hardcoded paths, hardcoded `.cuda()`, an 800k-iteration schedule,
val/test interleaved in the train loop, no seeding or clean checkpointing. It's a poor
base to grow variants B–E from, and it ships **no pretrained weights**
(`saved_weights/`, `model_checkpoints/` are empty), so the baseline must be trained
regardless.

**Why not reinvent from scratch.** The repo's model, loss, and metrics are exactly what
the paper's published numbers are measured against. We mirror the architecture faithfully
and **port the generic components verbatim** (Lovász loss, `Evaluator` metrics) rather
than risk subtle numeric drift that would contaminate the later "one change at a time"
comparisons.

**Honesty note on the published number.** The published U-Net row (64.94 set-level mIoU)
used a **6-channel** input (RGB + SAR replicated to 3 channels) and the full schedule. We
build the **4-channel** design the paper *describes* (RGB + 1 SAR), on a reduced schedule,
so our number will not match 64.94. Treat **64.94 as the external anchor we cite**, and
**our 4-channel U-Net as the internal control** for B–E under one identical harness.

## Project layout

Classic ML-project structure: configuration, library code, model architecture, runnable
scripts, and per-run outputs each get their own place.

```
codebase/
  config/
    baseline.yaml            # paths, hyperparams, reduced schedule, seed, norm stats, flags
  src/                       # importable library code
    dataset.py               # BrightDataset: 4-ch (optical 3 + SAR 1), official split
    transforms.py            # channel-aware normalize + crop/flip/rot (from imutils)
    losses.py                # CE + Lovász-softmax (ported verbatim)
    metrics.py               # Evaluator: loc(2) / clf(4) / total(4) (ported verbatim)
    engine.py                # shared train-epoch + evaluate() loop (reused by B–E)
    utils.py                 # seed, device, logging, run-dir + checkpoint helpers
  architecture/
    unet.py                  # UNet(in_channels=4, num_classes=4) — variant A
  scripts/
    eda.py                   # EDA phase  → results/eda/
    preprocess.py            # compute optical+SAR norm stats [+ optional --precrop cache]
    train.py                 # training + in-loop validation, save best checkpoint
    evaluate.py              # load checkpoint → full metric suite on val/test + error analysis
  splits/standard_ML/        # official train/val/test name lists (copied from the repo)
  results/
    <run>/                   # config snapshot, metrics.csv, training curves, predictions
    eda/                     # EDA deliverable: plots + stats.json + summary.md
  checkpoints/
    <run>/                   # best.pth, last.pth
  requirements.txt
  README.md                  # this file
```

**Design notes.** `src/` is the reusable library (data **and** the small shared
loss/metrics/engine/utils pieces that B–E will inherit). `scripts/` are thin entry
points. `architecture/` holds only model definitions — the one file that changes between
variants. Each run writes its weights to `checkpoints/<run>/` and everything else
(config snapshot, metric CSV, curves, predicted maps) to `results/<run>/`, so runs are
self-documenting and reproducible.

## Current state (verified)

- **Data is ready.** `../dataset/BRIGHT/{pre-event,post-event,target}` each hold 4,246
  `.tif`s. Pre-event optical tiles are `(1024,1024,3)` uint8; post-event SAR tiles are
  `(1024,1024)` single-band uint8; target tiles hold labels `{0,1,2,3}` (255 = ignore).
- **Official split** (use as-is, do **not** re-split): `splits/standard_ML/{train,val,test}_set.txt`
  = **2918 / 450 / 878** (sums to 4246; 7:1:2 per event, then merged).
- **Reference code ported from** `../../ProjectBaseline/BRIGHT/bda_benchmark/`:
  `model/UNet.py`, `util_func/lovasz_loss.py`, `util_func/metrics.py`,
  `dataset/make_data_loader.py`, `dataset/imutils.py`.
- **Recipe to match:** AdamW, lr `1e-4`, weight decay `5e-3`; loss
  `CE + 0.75·Lovász-softmax`; augmentations random crop / fliplr / flipud / rot90;
  optical normalized with ImageNet mean/std, SAR with its own stats.
- **Environment caveat (Windows).** `import torch` aborts with
  `OMP: Error #15 … libiomp5md.dll already initialized` unless
  `KMP_DUPLICATE_LIB_OK=TRUE` is set first. Every entry script sets
  `os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")` before importing torch.

## The 4-channel changes (vs. the 6-channel repo)

- **`architecture/unet.py`** — verbatim architecture (ConvBlock = ×2 conv-BN-ReLU;
  5-level encoder 64→1024; transpose-conv decoder with skip concats; 1×1 head). **Only
  change: `in_channels=4`.** Input H,W must be divisible by 16 (512 and 1024 both fine).
- **`src/dataset.py`** — mirror `MultimodalDamageAssessmentDatset`, but keep SAR as a
  single channel: **drop `np.stack((post_img,)*3)`**, expand to `(H,W,1)`. Return
  `optical (3,H,W)`, `sar (1,H,W)`, `loc_label`, `clf_label`, `name`; the model input is
  `torch.cat([optical, sar], dim=1)` → 4 channels. `loc_label` maps classes 2,3 → 1.
- **`src/transforms.py`** — `normalize_img` keeps ImageNet mean/std for the 3 optical
  channels and adds a **separate SAR normalization** (SAR mean/std from `preprocess.py`);
  `random_crop` padding is made channel-aware (per-image channel count instead of a
  hardcoded 3); flips/rotations are channel-agnostic, ported as-is.
- **`src/losses.py`** — `loss = CE(ignore_index=255) + 0.75·lovasz_softmax(softmax(logits), label, ignore=255)`.
  (We add `ignore_index=255` to CE, which the repo's U-Net path omitted — safe and
  correct — and fix the `classes is 'present'` identity check that warns on Python 3.13.)
- **`src/metrics.py`** — `Evaluator` ported verbatim; three-evaluator pattern
  (`loc` 2-class, `clf` 4-class damage-region, `total` 4-class).

## Cropping

Training crops 1024 tiles to **512** patches **on the fly** (random crop per sample each
epoch), matching the repo recipe — this keeps the informative-region oversampling
(`cat_max_ratio`) and full augmentation variety. `scripts/preprocess.py --precrop` can
instead cache fixed 512 patches to disk as an opt-in speed lever (config flag selects the
source); default is on-the-fly.

## EDA phase (`scripts/eda.py` → `results/eda/`)

A written deliverable (dataset statistics and characteristics analysis). Outputs PNG
plots + `stats.json` (machine-readable, including the normalization constants) +
`summary.md` (tables for the paper). Four analysis bundles:

1. **Integrity + class balance** — split counts (2918/450/878); every name has
   pre+post+target; splits disjoint; labels ⊆ `{0,1,2,3,255}`; all tiles 1024². Pixel
   class distribution reproducing the reported **82.8% intact / 10.7% damaged / 6.5%
   destroyed** building split and **~7:1** background:building. Tile-level composition
   (% of tiles containing each class, % with any damage, damage-ratio histogram).
2. **Event / disaster breakdown** — parse event prefixes → tiles per event × split;
   confirm 14 events / 23 regions; highlight the size imbalance (e.g. Turkey earthquake
   ~1114 tiles vs Hawaii wildfire ~65) that motivates the **event-level mIoU** metric;
   per-event class breakdown (which events lack damaged/destroyed).
3. **Image + normalization stats** — optical per-channel mean/std (vs ImageNet); **SAR
   mean/std** (feeds `config/baseline.yaml`); intensity histograms; nodata / all-black
   tile check; inverse- and median-frequency class weights as candidates for the E4
   targeted-loss experiment.
4. **Qualitative overlays + SAR signal** — `pre-optical | post-SAR | target` panels for a
   few tiles per disaster type (co-registration & modality sanity); SAR-intensity
   distribution by damage class (motivates multimodal fusion B/C).

## Evaluation

`scripts/evaluate.py --split {val,test} --run <name>` loads a checkpoint and reports the
full metric suite: **set-level mIoU** (main metric), **event-level mIoU** (averaged over
the 14 events), **F1_loc** (building vs background), **F1_clf** (damage classification),
**OA** (sanity only), and **per-class IoU**. Training (`train.py`) validates in-loop to
select `best.pth`; `evaluate.py` produces the final numbers and error analysis
(per-class, per-disaster-type, confusion matrices, predicted vs reference maps).

## Build order

1. Scaffold dirs + `requirements.txt` (incl. matplotlib); KMP env guard in entry scripts.
2. Port `architecture/unet.py`, `src/{losses,metrics,transforms,dataset,utils,engine}.py`.
3. `scripts/preprocess.py` → compute and save optical/SAR normalization stats into config.
4. `scripts/eda.py` → the four bundles → `results/eda/`.
5. `config/baseline.yaml` → reduced schedule, batch/crop, seed, flags.
6. `scripts/train.py`, `scripts/evaluate.py`.

## Verification

1. **EDA** runs end-to-end, reproduces the published class percentages, writes stats+plots.
2. **Smoke.** One train batch → `optical (B,3,512,512)`, `sar (B,1,512,512)`, concat
   `(B,4,512,512)`; one forward+backward on the GPU fits memory at the chosen batch (drop
   to 4 if OOM).
3. **Short run** (~a few hundred iters): loss decreases, val loop runs, `best.pth` is
   written, metrics print.
4. **Full reduced-schedule run → baseline numbers.** Report set-level mIoU, per-class IoU,
   F1_loc, F1_clf, OA on test. Expect **below 64.94** (reduced schedule + 4-ch) with
   **"Damaged" the lowest-IoU class**, and event-level mIoU well below set-level —
   replicating the documented failures the later experiments target.
5. **Reproducibility.** Fixed seed → stable numbers across reruns; config + seed recorded
   with each run so B–E run under an identical harness.

## Carried forward

The same harness (`src/` + `config/` + `scripts/`) is the foundation for variants **B–E**
— only `architecture/` and the fusion/input wiring change later.
