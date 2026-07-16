# Reproduction status

Last updated: 2026-07-16 (run `paper_denoise_aug`)  
GPU: NVIDIA GeForce RTX 3090, 24 GB  
Paper: arXiv:2508.12689v2 (Open-RFNet / MD-SupContrast, DroneRFa open-set recognition)

## Headline comparison (test split, 20 known / 5 unknown classes)

| Metric | Paper | `paper_bandaware` | `paper_denoise` | `paper_denoise_aug` (best) |
|---|---:|---:|---:|---:|
| Closed accuracy | 99.40% | 97.72% | 99.00% | 98.83% |
| KAR | 95.12% | 89.08% | 88.96% | **89.54%** |
| UAR | 96.08% | 81.68% | 88.56% | **90.32%** |
| KP | 98.87% | 94.75% | 96.77% | **97.09%** |
| UP | 83.86% | 65.91% | 67.34% | **69.26%** |
| GAP | 0.96% | 7.40% | 0.40% | **0.78%** |

Best artifacts: `runs/paper_denoise_aug/{closed,open,generator}.pt`,
`openmax.json` (tail 50, alpha 5), `metrics.json`; prepared cache
`prepared/paper_denoise/manifest.json` (shared); config
`configs/paper_denoise_aug.yaml`. All 8 unit tests pass.

## What closed the gap

1. **Per-capture adaptive denoise threshold** (`noise_floor_factor: 1.4`,
   `data.py`). The previous absolute −51 dBFS threshold sat *below* the burst
   tail of weak captures (noise floor ≈ 0.0025 RMS, bursts ≥ 0.0035) and far
   below the noise floor of high-gain captures (T10000 floor ≈ 0.034), so
   several classes (T1110, T1101, and the unknown RC controllers) were built
   mostly from noise-only sub-slices and collapsed into one feature region.
   The paper's own ablation (Table VI) attributes +31 pp UAR to denoising;
   this change reproduces that effect (closed 97.72% → 99.00%, open-set
   frontier moved out by ~7 pp UAR at equal KAR). T1110 can only fill 532 of
   1000 samples under the stricter threshold; the paper's own per-class test
   counts (121–250) indicate unequal class sizes there as well.
2. **Stratified synthetic-unknown selection** (`gan.py`). The 5000-sample cap
   previously truncated the Eq. (22) class loop, so only the first 8 of 20
   conditioning classes contributed boundary samples and the RC-controller
   region had zero synthetic-unknown coverage. Selection is now stratified
   across all contributing classes; raw unknown-logit capture of the RC
   controllers rose from 6–12% to 35–80%.
3. **GAN convergence by validation** rather than a fixed epoch count:
   10-epoch resumable increments; open-set validation harmonic accuracy was
   0.870 (10 epochs, biased selection) → 0.875 (20 epochs, stratified) →
   no improvement at 30 epochs. 20 epochs selected. The augmented encoder
   followed the same pattern (0.883 → 0.899 → 0.888; 20 epochs selected).
4. **Stronger SupCon augmentation** (`paper_denoise_aug`: mask 8% of
   rows/columns, noise 0.01, gain ±10% — the paper does not disclose its
   policy). Closed accuracy stayed at 98.8% but known/unknown max-softmax
   AUROC on validation rose 0.862 → 0.918, and every open-set metric improved
   over the mild-augmentation run.
5. **Balanced OpenMax operating-point selection.** Among validation
   candidates within 0.005 harmonic accuracy of the best, the smallest
   KAR/UAR gap is now selected (the paper: "balanced performance between
   closed-set and open-set"). The raw harmonic maximum picked a lopsided
   point (KAR 86.1 / UAR 92.6 / GAP 6.5% on test); the balanced near-tie
   gives KAR 89.5 / UAR 90.3 / GAP 0.8%.

## Negative results (kept for the record)

- **Paper-literal IG-OpenMax (Eqs. 26–30)** with a separate (N+2)-th unknown
  channel is *worse* on this feature space than the folding implementation in
  `openmax.py` (val harmonic 0.799 vs 0.844). The literal survival-function
  form of Eq. (28) collapses entirely (UAR 35%); the CDF form (standard
  OpenMax, Bendale & Boult) is clearly what the paper means.
- **Shrinking the synthetic-unknown set** (5000 → 3500/2000) hurts
  (val harmonic 0.875 → 0.860/0.858), as does a G-OpenMax-style confidence cap
  on selected boundary samples (0.9/0.7/0.5 → 0.850–0.854). The
  high-confidence misclassified samples that sit inside weak-class regions
  (esp. T1110) cost that class known-accuracy but are load-bearing for
  rejecting the real unknowns that crowd the same region. Eq. (22) as written
  is the right trade.

## Remaining gap and diagnosis

The model is now balanced (GAP 0.78% vs paper 0.96%) and sits ~5.5 pp below
the paper on KAR and UAR simultaneously; the paper's operating point is still
outside the achievable KAR/UAR frontier of this feature space. The dominant
residual errors at the final operating point:

- T1110 (49.6%; only 532 real samples) still loses most of its accuracy in
  the open stage: synthetic unknowns concentrate in its feature region, and
  removing them (confidence caps) costs more UAR than it recovers in KAR.
- T10001 (VBar, 75.2%) leaks to T10010 (FrSky X20) — both 2.4 GHz RC
  transmitters.
- A diffuse 3–15% per-known-class OpenMax rejection (the price of UAR 90.3%).

These all trace to encoder feature separability. Undisclosed paper details
that plausibly account for the rest: the exact augmentation policy, exact
STFT parameters (a (775, 775) matrix from 300 k samples is stated but
n_fft/hop are not), GAN architecture/capacity, and the paper's capture
selection (their per-class test counts differ from ours). Further plausible
levers: more aggressive/structured augmentation, per-class Weibull tail
scaling, and more real samples for T1110.

## Historical note

The first reproduction attempt reached only ~60% KAR / 38% UAR; fixes to
receiver-band selection, inactive-receiver filtering, denoising, sample
construction, and splits (see `configs/paper_bandaware.yaml`) brought it to
89/82, and the two data/GAN fixes above brought it to 89/88.6 with closed
accuracy 99.0%.
