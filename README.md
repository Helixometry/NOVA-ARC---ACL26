# NOVA-ARC: Prosody as Supervision

### Bridging the Non-Verbal–Verbal for Multilingual Speech Emotion Recognition

<p align="center">
  <img src="image/novaarc.png" alt="NOVA-ARC" width="120"/>
</p>

<p align="center">
  <a href="https://2026.aclweb.org/"><img src="https://img.shields.io/badge/ACL-2026-red?style=flat-square" alt="ACL 2026"/></a>
  <a href="#citation"><img src="https://img.shields.io/badge/cite-BibTeX-blue?style=flat-square" alt="cite"/></a>
  <img src="https://img.shields.io/badge/python-3.8%2B-green?style=flat-square" alt="python"/>
  <img src="https://img.shields.io/badge/pytorch-2.0%2B-orange?style=flat-square" alt="pytorch"/>
  <img src="https://img.shields.io/badge/license-MIT-lightgrey?style=flat-square" alt="license"/>
</p>

---

> **Accepted at ACL 2026 (Main)**
>
> *Girish\*, Mohd Mujtaba Akhtar\*, Muskaan Singh*
> (\* Equal Contribution)

---

## Overview

**NOVA-ARC** (**N**On-**V**erbal to **V**erbal **A**daptation via hyperbolic **A**lignment, **R**adial calibration, and **C**odebook tokens) is a geometry-aware framework for low-resource multilingual Speech Emotion Recognition (SER).

The key insight: **non-verbal vocalizations** (laughter, sighs, cries) express emotion through pure paralinguistic acoustics — independent of language. NOVA-ARC treats multilingual SER as an *unsupervised non-verbal → verbal transfer* problem:

- **Source**: labeled non-verbal vocalizations (NVV)
- **Target**: unlabeled verbal speech across multiple languages

No target-language emotion labels are needed at training time.

<p align="center">
  <img src="image/SAttri_eusipco.drawio.png" alt="NOVA-ARC Framework" width="750"/>
</p>

---

## Key Contributions

- **New formulation** for multilingual SER: use labeled non-verbal expressions as supervision and adapt to unlabeled verbal speech without any target emotion labels.
- **NOVA-ARC framework** combining:
  - Hyperbolic Poincaré ball geometry for hierarchical affective structure
  - HVQ prosody codebook for discrete paralinguistic tokens
  - Hyperbolic Emotion Lens (HEL) for intensity calibration across domains
  - Optimal transport prototype alignment for unsupervised adaptation
- **First end-to-end** framework to formulate low-resource multilingual SER as unsupervised transfer from labeled non-verbal to unlabeled verbal speech.
- Consistently outperforms Euclidean counterparts and strong SSL baselines across 6 datasets and 4 encoders.

---

## Architecture

```
Raw Waveform
     │
     ▼
SSL Encoder  (voc2vec / WavLM / wav2vec 2.0 / MMS)
     │   frame-level features  (B, T, D)
     ▼
Linear Projection  →  expmap₀  →  Poincaré Ball
     │   hyperbolic frame embeddings  {xₜ}
     ▼
HVQ Prosody Codebook          ←  Poincaré distance assignment
     │   discrete tokens  {qₜ}        +  straight-through gradient
     ▼
Möbius Addition  (xₜ ⊕ qₜ)   ←  continuous + discrete fusion
     │   bottleneck embeddings  {bₜ}
     ▼
Hyperbolic Emotion Lens (HEL)  ←  r  →  r^α  radial calibration
     │   calibrated embeddings  {b̃ₜ}
     ▼
Hyperbolic Attention Pooling   ←  logmap₀ → weighted sum → expmap₀
     │   utterance embedding  u♭  (B, d)
     ▼
Linear Classifier  →  softmax  →  emotion prediction

━━━━━━━━━━━━━━━  Domain Adaptation (target domain)  ━━━━━━━━━━━━━━━

Source prototypes μ^(c)  (Fréchet mean per class)
     +  target utterances  b̃ᵀ
     ▼
Sinkhorn Optimal Transport  →  soft pseudo-labels  q̂ⱼ
     ▼
L_total = L_S  +  λ_OPT · L_OPT  +  λ_OT-CE · L_OT-CE
```

---

## Results

### Cross-corpus Adaptation — NOVA-ARC (Table 3)

Trained on **ASVP-ESD (non-verbal)** as labelled source, evaluated on verbal targets.

| Target Dataset | Language | voc2vec (EUC) | **voc2vec (HYP)** | WavLM (HYP) | MMS (HYP) |
|:---|:---:|:---:|:---:|:---:|:---:|
| ASVP-ESD (V) | English | 87.31 | **92.40** | 91.03 | 89.43 |
| MESD | Spanish | 84.58 | **90.67** | 81.09 | 86.79 |
| AESDD | Greek | 79.63 | **84.39** | 82.98 | 82.03 |
| RAVDESS | English | 87.04 | **93.79** | 92.47 | 89.51 |
| Emo-DB | German | 86.71 | **92.46** | 91.26 | 88.11 |
| CREMA-D | English | 85.26 | **91.32** | 90.76 | 87.94 |

*Accuracy (%) reported. EUC = Euclidean baseline, HYP = NOVA-ARC hyperbolic.*

### Ablation Study (Table 4)

Source: ASVP-ESD (NVV) → Target: ASVP-ESD (Verbal), voc2vec encoder.

| Configuration | Accuracy | Macro-F1 |
|:---|:---:|:---:|
| Euclidean geometry | 87.31 | 85.06 |
| No VQ codebook (continuous only) | 74.22 | 70.43 |
| Token only (discrete only) | 76.90 | 73.18 |
| Concat/MLP instead of Möbius fusion | 65.36 | 62.24 |
| No HEL | 72.75 | 51.44 |
| Euclidean OT | 80.24 | 75.64 |
| Adversarial DA | 53.49 | 43.76 |
| OT-UDA baseline | 50.78 | 41.33 |
| **NOVA-ARC (full)** | **92.40** | **89.79** |

---

## Installation

```bash
git clone https://github.com/girish281003/NOVA-ARC.git
cd NOVA-ARC
pip install -r requirements.txt
```

Or install as a pip package:

```bash
pip install -e .
```

**Dependencies:** Python 3.8+, PyTorch 2.0+, transformers, scikit-learn

---

## Quick Start

### Minimal Example

```python
from nova_arc import NOVAARC, NOVAARCConfig
from nova_arc.encoders import WavLMEncoder

# 1. Pick an encoder
encoder = WavLMEncoder(freeze=True)          # output_dim = 768

# 2. Configure  (num_classes is the only required field)
config = NOVAARCConfig(
    num_classes = 5,        # number of emotion classes in your dataset
    epochs      = 30,
    device      = "cuda",
)

# 3. Build
model = NOVAARC(encoder=encoder, config=config)

# 4. Train  —  source = labelled NVV,  target = unlabelled verbal speech
model.fit(source_loader, target_loader)

# 5. Evaluate
metrics = model.evaluate(test_loader)
print(metrics)   # {"accuracy": 0.924, "weighted_f1": 0.897}

# 6. Predict
preds = model.predict(test_loader)   # (N,) int64 class indices

# 7. Save / load
model.save("nova_arc.pt")
model = NOVAARC.load("nova_arc.pt", encoder=WavLMEncoder(), map_location="cuda")
```

### DataLoader Format

```python
# Each loader yields: (waveforms, labels)
#   waveforms : (B, T_samples)  float32, 16 kHz, normalised to [-1, 1]
#   labels    : (B,)            int64,  0 … num_classes-1
#
# For the unlabelled target loader, labels can be -1 (ignored during training)

from torch.utils.data import DataLoader

source_loader = DataLoader(source_dataset, batch_size=16, shuffle=True)
target_loader = DataLoader(target_dataset, batch_size=16, shuffle=True)
test_loader   = DataLoader(test_dataset,   batch_size=16, shuffle=False)
```

### Built-in Encoders

| Class | HuggingFace / Path | `output_dim` | Best for |
|:---|:---|:---:|:---|
| `Voc2VecEncoder` | local checkpoint path | 768 | Non-verbal vocalizations |
| `WavLMEncoder` | `microsoft/wavlm-base-plus` | 768 | Verbal speech |
| `Wav2Vec2Encoder` | `facebook/wav2vec2-base` | 768 | Verbal speech |
| `MMSEncoder` | `facebook/mms-1b` | 1024 | Multilingual speech |

```python
from nova_arc.encoders import get_encoder

encoder = get_encoder("wavlm")
encoder = get_encoder("voc2vec", model_path="path/to/voc2vec")
encoder = get_encoder("mms")
```

### Custom Encoder

Plug in any feature extractor by subclassing `BaseEncoder`:

```python
from nova_arc.encoders import BaseEncoder
import torch.nn as nn

class MyEncoder(BaseEncoder):
    def __init__(self):
        super().__init__(output_dim=512)
        self.backbone = ...             # your model here

    def forward(self, waveforms):       # (B, T_samples)  →  (B, T_frames, 512)
        return self.backbone(waveforms)

model = NOVAARC(encoder=MyEncoder(), config=NOVAARCConfig(num_classes=5))
```

---

## Configuration Reference

`NOVAARCConfig` controls every aspect of the architecture. All fields have defaults except `num_classes`.

```python
from nova_arc import NOVAARCConfig

config = NOVAARCConfig(
    # Required
    num_classes      = 5,

    # Architecture toggles
    use_hyperbolic   = True,       # False → pure Euclidean (also disables codebook + HEL)
    use_codebook     = True,       # False → skip HVQ prosody codebook
    use_hel          = True,       # False → skip Hyperbolic Emotion Lens
    use_ot           = True,       # False → source-only training (no target_loader needed)
    pooling          = "attention",# "attention" | "mean" | "max"

    # Geometry
    hyperbolic_dim   = 256,        # embedding dimension d
    curvature        = 1.0,        # Poincaré ball curvature c  (κ = -c)

    # HVQ Codebook
    codebook_size    = 256,        # K
    commitment_weight= 0.25,       # β
    vq_loss_weight   = 1.0,        # λ_VQ

    # Optimal Transport
    sinkhorn_eps     = 0.05,       # ε_OT
    sinkhorn_iters   = 50,
    lambda_opt       = 1.0,        # λ_OPT
    lambda_ce        = 1.0,        # λ_OT-CE

    # Training
    lr               = 1e-4,
    epochs           = 30,
    device           = "cuda",
)
```

**Common ablation configs:**

```python
# Euclidean baseline
NOVAARCConfig(num_classes=5, use_hyperbolic=False, use_ot=False)

# Hyperbolic, source-only (no adaptation)
NOVAARCConfig(num_classes=5, use_ot=False)

# No HEL
NOVAARCConfig(num_classes=5, use_hel=False)

# No codebook
NOVAARCConfig(num_classes=5, use_codebook=False)
```

---

## Hyperparameters

| Hyperparameter | Value |
|:---|:---|
| Hyperbolic curvature | κ = −1.0 |
| Hyperbolic latent dim | d = 256 |
| Bottleneck dim | d_b = 128 |
| VQ codebook size | K = 256 |
| VQ commitment weight | β = 0.25 |
| HEL exponent (init) | α = 1.0 (learned) |
| OT regularisation | ε_OT = 0.05 |
| Sinkhorn iterations | 50 |
| Optimizer | AdamW (β=0.9, 0.98) |
| LR (encoder) | 3 × 10⁻⁵ |
| LR (new layers) | 1 × 10⁻⁴ |
| Weight decay | 0.01 |
| Gradient clipping | 1.0 |
| Epochs | 30 |
| Schedule | 10% warmup + cosine decay |
| Batch size | 16 |
| Prototype refresh | once per epoch |

---

## Repository Structure

```
NOVA-ARC/
│
├── nova_arc/
│   ├── __init__.py          # Public API  (NOVAARC, NOVAARCConfig, encoders)
│   ├── hyperbolic.py        # Poincaré ball ops: expmap, logmap, Möbius, Fréchet mean
│   ├── encoders.py          # BaseEncoder + built-in SSL wrappers
│   ├── codebook.py          # Hyperbolic VQ codebook
│   ├── emotion_lens.py      # Hyperbolic Emotion Lens (HEL)
│   ├── pooling.py           # Hyperbolic attention pooling
│   ├── prototypes.py        # PrototypeBank (Fréchet mean, epoch refresh)
│   ├── transport.py         # Sinkhorn OT + soft pseudo-labels
│   ├── losses.py            # L_S, L_OPT, L_OT-CE combined loss
│   └── model.py             # NOVAARC nn.Module + fit / evaluate / predict / save / load
│
├── configs/
│   └── default.yaml         # All hyperparameters
│
├── examples/
│   └── quick_start.py       # Full usage walkthrough
│
├── test.py                  # End-to-end test with synthetic data (no downloads)
├── index.html               # Paper website
├── setup.py
└── requirements.txt
```

---

## Datasets

| Dataset | Language | Utterances | Type | Role |
|:---|:---:|:---:|:---|:---|
| [ASVP-ESD](https://www.kaggle.com/datasets/dejolilandry/asvpesdall) | English | ~4000 | NVV + Verbal | Source (NVV) / Target (Verbal) |
| [MESD](https://data.mendeley.com/datasets/cy34mh68j9) | Spanish | ~864 | Verbal | Target |
| [AESDD](http://m3c.web.auth.gr/research/aesdd-speech-emotion-recognition/) | Greek | ~500 | Verbal | Target |
| [RAVDESS](https://zenodo.org/record/1188976) | English | 7356 | Verbal | Target |
| [Emo-DB](http://emodb.bilderbar.info/docu/) | German | 800 | Verbal | Target |
| [CREMA-D](https://github.com/CheyneyComputerScience/CREMA-D) | English | 7442 | Verbal | Target |

All datasets are standardised to the shared label space: **happy · anger · disgust · sadness · fear**

---

## Run the Test

A fully self-contained test using synthetic data — no dataset downloads required:

```bash
python test.py
```

Output includes:
- Per-epoch training loss breakdown (`L_S`, `L_OPT`, `L_OT-CE`)
- Labelled test-set accuracy, weighted F1, macro F1
- Full per-class classification report (precision / recall / F1)
- Unlabelled target-domain analysis: predicted class distribution, confidence, OT weights, prediction entropy

---

## Citation

If you use NOVA-ARC in your research, please cite:

```bibtex
@inproceedings{girish2026novarc,
  title     = {Prosody as Supervision: Bridging the Non-Verbal--Verbal
               for Multilingual Speech Emotion Recognition},
  author    = {Girish and Akhtar, Mohd Mujtaba and Singh, Muskaan},
  booktitle = {Proceedings of the 64th Annual Meeting of the Association
               for Computational Linguistics (ACL 2026)},
  year      = {2026},
}
```

---

## License

This project is licensed under the MIT License.

---

<p align="center">
  <b>ACL 2026</b> &nbsp;·&nbsp;
  Girish* &nbsp;·&nbsp;
  Mohd Mujtaba Akhtar* &nbsp;·&nbsp;
  Muskaan Singh
  <br/>
  <i>* Equal Contribution</i>
</p>
