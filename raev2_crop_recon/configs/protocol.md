# RAEv2 ROI crop-latent reconstruction pilot — protocol (fixed 2026-11-04 ~16:20 KST, before calibration / training / PACO)

Question: does pixel supervision through the pretrained, frozen RAEv2 pixel decoder help a CoordMLP turn small full-image ROI
features into crop-like latents (vs the existing feature matching)? Primary comparison **C − A**.

## Official model (verified, results/verify_pair.json)
- Repo nanovisionx/RAEv2 @ 8a0d238f8dc3 (2026-05-21); HF nyu-visionx/RAEv2-models @ 9770b7b980fa.
- `stage1/imagenet/dinov2b-k1/decoder.pt` sha256 aabfee19…5a76 (= HF linked etag), 415.35M params, ViT-XL decoder config
  (`configs/decoder/ViTXL`: 1152 wide, 28 layers, 16 heads), pos-embed (1, 257, 1152), decoder_pred 768 = 16·16·3 → 256×256 RGB.
  Missing / unexpected keys: none (asserted).
- `stats.pt`: mean, var (768, 16, 16) per position — used in full shape, never reduced to per-channel constants.
- Encoder `dinov2-vit-b` (RAEv2 DINOv2Encoder default) = torch.hub facebookresearch/dinov2 `dinov2_vitb14_reg`
  (weights sha256 73182a08…, 4 registers, final LN replaced by LN **without affine**, pos-embed resampled to 16×16 by timm).
  Not the timm lvd142m model of the previous experiments.
- RAE resolution = 256 (decoder output frame), encoder input 224 (official `F.interpolate(x, 224, bicubic)` after ImageNet norm),
  latent 16×16×768. resolution is NOT changed to 224.
- Pair check (ImageNet val 256, official sample preprocessing = centre square crop → 256 BICUBIC): default PSNR 18.72 / LPIPS 0.256;
  [woreg] 8.37 / 0.878; [norm] 16.21 / 0.334 → only the default encoder matches. Empirical per-position stats vs stats.pt:
  mean corr 0.993, var ratio 0.995.
- Env: research/envs/dinov2_ord_py310 (torch 2.7.1, transformers 4.52.1; official pins torch 2.10 / transformers 4.57.1 — only
  generic transformers utils are used by the decoder; numerical agreement with the official RAE class checked in sanity S1).

## Preprocessing / geometry (differs from the official centre-crop sampler by design)
- No full-image centre crop (it can cut ROIs). Full image and GT crop: the same external 256 letterbox
  (s = 256 / max(w, h), round, centred, PIL BICUBIC, padding = ImageNet mean colour (124,116,104)), uint8 canvas →
  official encoder preprocess (/255, ImageNet norm, bicubic 256→224) → raw tokens.
- One encoder token (14 px at 224) = one 16×16 cell of the 256 canvas; all geometry is computed in the 256 frame with the existing
  `roi_geometry` code (S=256, PATCH=16, G=16): ROI(+1-cell ring) tokens, crop-canvas coords, bil_grid, coverage.
- Pixel target = the 256 letterbox canvas of the ORIGINAL GT crop (same u8 canvas the teacher encoder sees); loss on content
  pixels (content mask from the crop letterbox rect); padding excluded (L1 masked, LPIPS on composite with GT outside).
- Content rects, n_in on the RAE grid and cache offsets are stored in perception_cache/feat/*_rois.parquet.

## Latent spaces
- raw = encoder output tokens (final LN w/o affine). normalised = (raw − mean[c,i,j]) / sqrt(var[c,i,j] + 1e-5) (official).
- ROI sampling (gather / bilinear) on raw. CoordMLP output = raw (its own existing per-channel mean_T/std_T output buffers,
  = teacher-crop channel stats; this is the unchanged CoordMLP structure, not the RAE stats). Feature loss on raw.
- Decoder input = raw directly; decoder(raw) ≡ official RAE.decode(normalise(raw)) (no double de-normalisation; S1 checks it).
- Latent noise off (no noising anywhere; official RAE class instantiated with noise_tau 0 in eval for S1).

## Model
- CoordMLP = R00 structure: MLRDecoder("final") = CoordMLP body (k=4 nearest ROI tokens + Fourier offsets, ROI summary + box vector,
  crop-canvas query position P + full-grid query position) + zero-init adapter; models_mlr.py with only G/PATCH/S changed
  (output grid 16×16 = RAE latent). 10.16M params. Fresh init (torch seed 0) saved once
  (perception_cache/init/coordmlp_init_s0.pt) and loaded by A, B and C. No R00 / mlp_s0 weights reused.

## Data
- CoordMLP training: COCO decoder pool (coco_rois.parquet split 'train': 60k COCO train2017 images × 4 ROIs = 3 GT + 1 random) MINUS
  all 4,139 pool images that appear in PACO val or test → 55,861 images. Stream = existing sampler relation.build_stream
  (4 categories × 4 images, sqrt-balanced), seed 2027, 12,000 steps, saved as image ids (perception_cache/stream/).
  50,491 unique images / 201,964 ROIs; overlap with PACO val/test images 0, with COCO dev / rae head_val 0 (results/data_overlap.json).
  PACO attribute labels are never used for CoordMLP training (PACO train images may appear as plain COCO images).
- Cache (fp16): full 16×16 raw latents of every stream image; teacher latents of every stream ROI on content cells only
  (padding cells have weight 0 and are never targets); dev (coco_rois 'dev', 2,000 ROIs) and hval (rae_crop_pilot head_val,
  3,194 ROIs) likewise. Encoder fp32, TF32 off.

## Training (A / B / C identical except the loss)
- A: L_feat = MSE_w + 1.0 (1 − cos)_w (existing loss, w = teacher content coverage).
- B: λ · L_pix,  L_pix = L1 + β LPIPS-VGG (lpips 0.1.4).  C: L_feat + λ · L_pix.
- β, λ from `code/calib.py` at the shared init on the first 8 stream rows (fixed train batches), Z-level gradient norms:
  β = 1 if r = ||∇LPIPS||/||∇L1|| ≤ 1 else largest of {0.1, 0.2, 0.5, 1} with βr ≤ 1;
  λ = largest of {0.01 … 100} with λ q ≤ 1, q = ||∇L_pix||/||∇L_feat|| (pixel term on equal footing). Never changed afterwards.
- AdamW lr 1e-4 (0.9, 0.95) wd 0.05, warmup 600 + cosine to 0 at 12k, clip 1.0, BF16 autocast (CoordMLP, RAE decoder, LPIPS),
  64 ROIs / step, micro-batch size from the pixel-condition memory test (sanity S7), same partition and accumulation in A/B/C,
  torch seed 0. Frozen: encoder (cache), RAE decoder (eval, requires_grad False, fingerprint checked), LPIPS.
  Student pixel path keeps gradients (no no_grad / detach / clamp). Activation checkpointing only if memory requires (recorded).
- GPUs: A → 0, B → 1, C → 7. Checkpoints 4k / 8k / 12k. COCO dev every 1k (feature + RGB metrics), grad norms every 250 steps.

## Sanity gates (results/sanity.json; must pass before the main runs)
S1 wrapper = official RAE class; S2 teacher crop recon looks right; S3 bilinear recon; S4 shapes / spaces; S5 coordinates
(synthetic coordinate image); S6 pixel gradient reaches CoordMLP, decoder unchanged; S7 memory; S8 256-ROI overfit
(feature-only and pixel-only, fixed 256 dev ROIs; loss must clearly decrease — not required to reach 0; weights discarded).

## Evaluation
- Fixed step: **12k** is the main A/B/C comparison; 4k is diagnostic only (dev/val). No test-based change of settings.
- Reconstruction (hval 3,194 ROIs, + PACO test-primary ROIs by size bin): RGB L1, PSNR, LPIPS, mean colour error dE_mean, regional
  dE_8×8, GT-aligned detail hp_corr, hf_ratio; latent MSE / cos vs teacher, spatial-variance ratio; also teacher recon and bilinear.
- PACO: existing M2 (official-style category-wise AP), test-primary (ROIs whose images are outside the old decoder pool; the same
  ROI list as all previous experiments), R2 AttnReader + R0, reader seeds 0,1,2, best epoch by val pooled mAP.
  Conditions, all on the RAE latent: Full (direct GT-ROI read-out of full 16×16 tokens), Bilinear, GT-crop oracle, A, B, C (12k).
  Size bins: primary = the existing F336 n_in bins (D1-4 part / D5-16 object / D17+); SigLIP2-bin 5-16 and RAE-grid n_in secondary.
  Paired image bootstrap (1,000, seed 0) on test-primary. Primary: C − A (overall, D5-16 object); also B − A, C − B, each vs Full /
  Bilinear / Crop. Existing DINOv2 F336 / C0 numbers are not comparable (different backbone, resolution) and are not used as baselines.
- Stop rule: if pixel supervision improves reconstruction but not PACO AP, stop without further loss / architecture search.
