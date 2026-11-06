# EVA-CLIP / CLIPSelf / DeCLIP on PACO — protocol (fixed 2026-11-06 ~02:40 UTC, before any PACO probe was trained)

## Models (all EVA02-CLIP-B/16, frozen, official repo code)
| key | checkpoint (SHA256) | source | dense feature (official path) |
|---|---|---|---|
| eva | EVA02_CLIP_B_psz16_s8B.pt (4aab21dd…ddf0) | HF QuanSun/EVA-CLIP @11afd20 | CLIPSelf `encode_dense` v2 (last block without attention mixing) → LN → head 512 → L2 |
| clipself | eva_vitb16_coco_clipself_patches.pt (9a8e9109…32a1) | author OneDrive (README Google-Drive links are 404; issue #32/#38 point to OneDrive) | same path |
| declip | evab_dinov2B_csa_560_0.25_seg.pt (2196da65…97c4), internal name `evab_dinov2B_qq_560_0.25`, epoch 6 | HF xiaomoguhzz/DeCLIP_evab_dinov2B_csa_560_0.25_seg @3984d86 (linked from the repo ZSSS branch) | DeCLIP `encode_dense` mode `qq` (inference mode of a qq_vfm_distill model, training/zero_shot.py) = CONTENT feature → L2 |
| eva_c0 (oracle) | = eva | | eva path on the GT crop (224 letterbox) |

Not used for primary: DeCLIP `epoch_6.pt` of `DeCLIP_EVA-B_DINOv2-B_csa_0.05_2.0` (CAT-Seg branch, retrained, csa mode, non-default loss weights); DeCLIP2 (TPAMI); CLIPSelf CC3M / proposals / LVIS.
Precision: fp16 autocast = the repos' default `--precision amp` (bf16 corrupts EVA dense tokens; verify/precision_check.txt).

## Resolution (decided after a zero-shot sanity check, before PACO)
Zero-shot box classification (COCO val2017 instance boxes, 133-class repo text embeddings, RoIAlign 1x1, Top-1 mAcc):
1024: EVA 17.5, CLIPSelf 71.3, DeCLIP 68.0 | 560: EVA 42.3, CLIPSelf 69.4, DeCLIP 73.0 (paper, CLIPSelf Table 2 @1024: EVA 18.2, CLIPSelf 72.1).
- **Primary = 1024 letterbox (64×64 tokens)** for all three: CLIPSelf training/eval resolution and the resolution of the
  published CLIPSelf-vs-DeCLIP comparison. It favours CLIPSelf and is far from EVA's native 224.
- **Secondary (pre-registered) = 560 (35×35 tokens)**: DeCLIP training resolution; reported separately, not mixed with the primary.
- Oracle = EVA-CLIP GT crop re-encoded at 224 (EVA native, = CLIPSelf teacher input), 14×14 tokens. Separate control.

## PACO protocol (unchanged from the DINOv2 IVC / MSD test-primary runs; src.densecmp.train = src.ivc.train restricted to R2)
Data/splits/labels: RC.load_data(EXCL) (train minus 6 near-duplicates), PACO dev = 1,000 held-out train images (rng 0), val 2,051 imgs,
test-primary = test ROIs not in the decoder COCO pool (26,442 ROIs / 4,848 imgs). Reader R2 (2 cross-attention blocks, mask cov>0 on
the full-image grid, geo = [rel, cov, log size / patch]) + input adapter LN(512)→Linear(512,768) (only change; same for every model).
Same optimiser/schedule/batching/seed (reader init seed 7, data seed 0), 10 epochs, epoch = max PACO-dev M0 Overall (IVC/MSD rule).
Test scored once with the selected checkpoint (val re-score must reproduce saved val scores). Metric M2 primary, M0 reference,
size bins = manifest n_in (DINOv2 24×24 grid; NOT redefined on the EVA grid), paired image bootstrap B=1000 seed 0.
Exposure sensitivity: test-primary images from COCO val2017 only (CLIPSelf/DeCLIP self-distilled on COCO train2017 images).

**Change 2026-11-06 03:3x UTC (user request; at that time only EVA@1024 was finished/test-scored, CLIPSelf@1024 was mid-training (its epoch-4 dev M0 had been seen in a progress check), no CLIPSelf/DeCLIP val/test table had been made):** the 560 secondary resolution is dropped. Runs = EVA / CLIPSelf / DeCLIP @1024 + EVA GT-crop oracle @224.
