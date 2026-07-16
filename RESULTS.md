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

- The literal survival-function form of Eq. (28) collapses entirely (UAR 35%
  on the earlier model); the CDF form (standard OpenMax, Bendale & Boult) is
  clearly what the paper means — the survival form penalizes inliers instead
  of outliers.
- **Paper's separate (N+2)-th unknown channel (Eqs. 26–30)**: an early test
  suggested it was much worse (val harmonic 0.799 vs 0.844), but that test
  multiplied *negative* logits by c<1 (inflating them — Bendale's formulation
  assumes positive activations) and used the weaker pre-denoise-fix model.
  Re-tested on the final model with a relu-guarded shrink, the faithful N+2
  form is statistically equivalent to the folding implementation (test
  harmonic 0.895 vs 0.902) and better balanced (GAP 0.61% vs 2.34% at each
  variant's own validation-selected operating point). Folding remains the
  default; the two differ only in whether the redistributed probability mass
  competes with or adds to the trained unknown logit.
- **Shrinking the synthetic-unknown set** (5000 → 3500/2000) hurts
  (val harmonic 0.875 → 0.860/0.858), as does a G-OpenMax-style confidence cap
  on selected boundary samples (0.9/0.7/0.5 → 0.850–0.854). The
  high-confidence misclassified samples that sit inside weak-class regions
  (esp. T1110) cost that class known-accuracy but are load-bearing for
  rejecting the real unknowns that crowd the same region. Eq. (22) as written
  is the right trade.

## Equation-level alignment audit (2026-07-16)

A line-by-line pass over Eqs. (1)–(36) and Algorithm 1 against the code
confirmed exact matches for the metrics (Eq. 36), preprocessing (Eqs. 2–4),
both feature branches and fusion (Eqs. 5–12), the generative stage
(Eqs. 20–23), Weibull fitting (Eqs. 24–25), and WGAN-GP (Eqs. 31–32).
Parameter-count constraints pin the undisclosed architecture dimensions:
ResNet 700,528 vs the paper's 700,816 (−0.04%), full model 206.08 M vs
205.81 M (+0.13%); one additional TransformerEncoder layer would add
~1.05 M, so L=1 is the uniquely consistent depth. Remaining known
deviations, all minor and documented:

- **Eq. (28) rank weights**: the paper's literal (α−k)/α makes α=1 a no-op
  (likely another off-by-one typo); the library uses Bendale's (α−k+1)/α.
  Both were evaluated — test results are equivalent within selection noise.
- **Top-α ranking scope**: the library ranks known logits only; Algorithm 1
  sorts all N+1 channels (part of the fold-vs-N+2 difference shown
  equivalent above).
- **Weibull location**: fixed at 0 on raw distances vs libMR's translated
  two-parameter fit; absorbed by tail-size selection.
- **Ltotal ambiguity (untested alternative)**: Table I defines a combined
  Lsup+Lce loss that the text never uses; Fig. 4's accuracy curve over the
  30 pre-training epochs hints at joint training. This reproduction uses the
  sequential reading (30 SupCon epochs, then 10 classifier epochs with the
  encoder fine-tuned at 1e-4; full freezing was much worse).
- **Eq. (13) A(i)**: described as "all negative samples", but Khosla et al.
  (cited) define A(i) as all samples except the anchor; the implementation
  follows Khosla.

## Split protocol and leakage audit

Splits are assigned per 3 ms sample (slice level), matching the paper's
sample-level Dtrain/Dtest description (Eqs. 16–17; the paper never mentions
capture-level splitting, and DroneRFa has only 8–16 captures per class).
An audit of the prepared manifest confirms:

- **No literal data reuse**: all 73,596 one-millisecond sub-slice windows are
  unique, aligned, and non-overlapping — no I/Q segment appears in more than
  one sample, so train and test never share raw data.
- **Capture-level mixing exists by construction**: 246 of 313 capture-channels
  contribute slices to both train and test. Temporally adjacent slices from
  the same capture are correlated (same distance, channel state, hardware
  session), so absolute accuracies are optimistic relative to a
  held-out-capture deployment scenario. This affects the paper's numbers
  identically, so the paper-vs-reproduction comparison remains
  apples-to-apples; a capture-level split would be a stricter, different
  benchmark than the one the paper reports.

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
