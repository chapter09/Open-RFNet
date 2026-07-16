# Open-RFNet reproduction

This is a clean-room PyTorch reproduction of **Open-RFNet** from Gao et al.,
“Multi-Domain Supervised Contrastive Learning for UAV Radio-Frequency Open-Set
Recognition” ([arXiv:2508.12689v2](https://arxiv.org/abs/2508.12689)). It uses the
local DroneRFa dataset at `/home/coder/DroneRFa`.

The implementation follows the paper's four-stage pipeline:

1. Slice I/Q recordings into 3 ms (300,000-point) samples, discard weak 1 ms
   sub-slices, compute normalized STFT spectrograms, and cache them as float16.
2. Train the ResNet-18 texture branch and dual time/frequency TransformerEncoder
   branches with supervised contrastive learning, then train the closed-set head.
3. Train a conditional WGAN-GP, retain generated examples misclassified by the
   closed model, freeze the feature extractor, and retrain a `K+1` classifier.
4. Fit per-class Weibull tails to mean activation vector distances and apply
   IG-OpenMax calibration at inference.

## Important reproduction limits

The authors did not release source code in the paper or in a public GitHub
repository found on 2026-07-14. Several values required for exact numerical
reproduction are also omitted. They are never hidden in this project: every one
is exposed in YAML.

| Item | Paper | This reproduction |
|---|---|---|
| Dataset subset | Near outdoor + indoor | Exact label/file rule implemented |
| Open split | 20 known / 5 unknown | Exact Table IV split |
| Slice | 3 ms / 300,000 points | Exact |
| STFT output | 775 x 775 | Exact in `paper.yaml` |
| STFT window/hop | Not reported | 1024 / 384, then bilinear resize |
| Denoising threshold | Not reported | -56 dBFS complex RMS; background bypasses it |
| Examples per class | Not reported | 1,000, inferred from 0.4% test-accuracy steps |
| Train/test split | Not reported | 75/25, with 10% of train used for validation |
| Model widths/depths | Not reported | Chosen to approximate the reported 205.8M parameters |
| Optimizer | Adam + cosine schedule | Exact family; unreported LR is configurable |
| SupCon / classifier epochs | 30 / 10 | Exact |
| Augmentation policy | Not reported | Gain, noise, shifts, time/frequency masks |
| GAN schedule / OpenMax tail | Not reported | Configurable defaults |

The paper reports training on a 32 GB Tesla V100 with batch size 128. The host
used here has a 24 GB RTX 3090. `configs/rtx3090.yaml` preserves the method while
reducing the spectrogram/model size. `configs/paper.yaml` preserves the reported
775 x 775 input and batch size.

## Setup

```bash
cd /home/coder/Open-RFNet
python -m venv --system-site-packages .venv
.venv/bin/python -m pip install -e '.[test]'
```

Validate the source dataset. The command detects and skips unreadable HDF5
files; on this host, `T0010_D01_S1111.mat` is truncated but is not part of the
paper's near-distance subset.

```bash
.venv/bin/open-rfnet inspect --dataset /home/coder/DroneRFa
```

Inspect model size:

```bash
.venv/bin/open-rfnet model-info --config configs/paper.yaml
.venv/bin/open-rfnet model-info --config configs/rtx3090.yaml
```

Run a quick end-to-end validation:

```bash
.venv/bin/open-rfnet reproduce --config configs/smoke.yaml
```

Run the short all-class sanity experiment recorded in `RESULTS.md`:

```bash
.venv/bin/open-rfnet prepare --config configs/quick.yaml
.venv/bin/open-rfnet reproduce --config configs/quick.yaml
```

Run the practical RTX 3090 experiment:

```bash
.venv/bin/open-rfnet prepare --config configs/rtx3090.yaml
.venv/bin/open-rfnet reproduce --config configs/rtx3090.yaml
```

To attempt the literal paper-scale configuration:

```bash
.venv/bin/open-rfnet prepare --config configs/paper.yaml
.venv/bin/open-rfnet reproduce --config configs/paper.yaml
```

Each stage is separately resumable at the command level:

```bash
.venv/bin/open-rfnet train-closed --config configs/rtx3090.yaml
.venv/bin/open-rfnet train-gan --config configs/rtx3090.yaml
.venv/bin/open-rfnet train-open --config configs/rtx3090.yaml
.venv/bin/open-rfnet evaluate --config configs/rtx3090.yaml
```

Outputs are written under the configured `run_dir`:

- `closed.pt`: MD-SupContrast encoder and closed-set classifier
- `generator.pt`: conditional WGAN-GP generator
- `synthetic_unknown.pt`: Eq. (22) boundary examples
- `open.pt`: frozen encoder plus retrained `K+1` head
- `openmax.json`: mean activation vectors and Weibull parameters
- `metrics.json`: KAR, UAR, KP, UP, GAP, and per-class accuracy

## Dataset details encoded from the paper

- Known: `T0000 T0010 T0011 T0100 T0101 T0110 T0111 T1000 T1001 T1010
  T1011 T1100 T1101 T1110 T1111 T10000 T10010 T10100 T10101 T10111`
- Unknown: `T0001 T10001 T10011 T10110 T11000`
- Outdoor models use only `D00` (20-40 m); indoor models have no distance tag.
- RF0/RF1 band selection follows Table III and the DroneRFa acquisition notes.

The paper's Table V prints the `Y` unknown class as `T10000`; Table III and
Table IV establish that `Y` is `T11000`, which is what this code uses.

