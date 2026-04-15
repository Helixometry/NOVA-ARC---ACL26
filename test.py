"""
test.py
-------
Step 1 : Create dummy labelled source data   (5 emotion classes)
Step 2 : Create dummy unlabelled target data
Step 3 : Import NOVA-ARC library and build the model
Step 4 : Train
Step 5 : Evaluate and print full report
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from nova_arc import NOVAARC, NOVAARCConfig
from nova_arc.encoders import BaseEncoder
from sklearn.metrics import classification_report, accuracy_score, f1_score

# ─────────────────────────────────────────────────────────────────────────────
# Settings
# ─────────────────────────────────────────────────────────────────────────────

EMOTIONS    = ["neutral", "happy", "sad", "angry", "fear"]
NUM_CLASSES = len(EMOTIONS)          # 5
SR          = 16_000                 # 16 kHz
DURATION    = 2                      # seconds
FRAME_LEN   = SR * DURATION          # 32 000 samples per clip


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — Labelled source dataset  (simulates your real labelled split)
# ─────────────────────────────────────────────────────────────────────────────

class LabelledDataset(Dataset):
    """
    Dummy labelled dataset.
    Each sample: (waveform [T], label [int 0..4])

    In real use, replace this with your own Dataset that loads actual audio.
    """
    def __init__(self, n_samples=300, seed=0):
        rng = torch.Generator().manual_seed(seed)
        # Random waveforms
        self.waves  = torch.randn(n_samples, FRAME_LEN, generator=rng)
        # Balanced class labels  0,1,2,3,4 ...
        self.labels = torch.arange(n_samples) % NUM_CLASSES
        # Add a tiny per-class signal so the model can learn something
        for c in range(NUM_CLASSES):
            self.waves[self.labels == c] += 0.04 * c

    def __len__(self):
        return len(self.waves)

    def __getitem__(self, idx):
        return self.waves[idx], int(self.labels[idx])


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — Unlabelled target dataset  (simulates your real unlabelled split)
# ─────────────────────────────────────────────────────────────────────────────

class UnlabelledDataset(Dataset):
    """
    Dummy unlabelled dataset.
    Each sample: (waveform [T],)   — no labels needed for OT adaptation.
    """
    def __init__(self, n_samples=150, seed=99):
        rng = torch.Generator().manual_seed(seed)
        self.waves = torch.randn(n_samples, FRAME_LEN, generator=rng)

    def __len__(self):
        return len(self.waves)

    def __getitem__(self, idx):
        # Return a dummy label of -1 so DataLoader can batch it normally
        return self.waves[idx], -1


# Build datasets and loaders
source_ds  = LabelledDataset(n_samples=400, seed=0)
target_ds  = UnlabelledDataset(n_samples=200, seed=1)
test_ds    = LabelledDataset(n_samples=100, seed=2)

source_loader = DataLoader(source_ds, batch_size=32, shuffle=True)
target_loader = DataLoader(target_ds, batch_size=32, shuffle=True)
test_loader   = DataLoader(test_ds,   batch_size=32, shuffle=False)

print("=" * 60)
print("  Data ready")
print(f"  Source (labelled)   : {len(source_ds)} samples, {NUM_CLASSES} classes")
print(f"  Target (unlabelled) : {len(target_ds)} samples")
print(f"  Test   (labelled)   : {len(test_ds)} samples")
print(f"  Classes             : {EMOTIONS}")
print("=" * 60)


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 — Define encoder and build the model
# ─────────────────────────────────────────────────────────────────────────────
# The encoder is the ONLY part you bring to NOVA-ARC.
# In a real experiment use:
#     from nova_arc.encoders import WavLMEncoder
#     encoder = WavLMEncoder(freeze=True)   # downloads checkpoint once
#
# Here we use a minimal DummyEncoder so this test runs on CPU
# without downloading anything.

ENC_DIM  = 128
N_FRAMES = 50   # simulated number of time frames


class DummyEncoder(BaseEncoder):
    """
    Placeholder for any real SSL encoder (WavLM, voc2vec, wav2vec2 …).

    A real SSL encoder takes raw waveforms and returns frame-level features:
        (B, T_samples) → (B, T_frames, feature_dim)

    This dummy version does one linear projection + repeats it N_FRAMES times
    to produce the same expected shape — just enough to test the library.
    """
    def __init__(self):
        super().__init__(output_dim=ENC_DIM, freeze=False)
        self.proj = nn.Linear(FRAME_LEN, ENC_DIM)

    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
        frame = self.proj(waveforms)                          # (B, ENC_DIM)
        return frame.unsqueeze(1).expand(-1, N_FRAMES, -1).contiguous()
        # returns (B, N_FRAMES, ENC_DIM)  — same contract as any real encoder


# Build NOVA-ARC
config = NOVAARCConfig(
    num_classes    = NUM_CLASSES,   # must match your dataset
    hyperbolic_dim = ENC_DIM,       # same as encoder output dim
    use_hyperbolic = True,
    use_codebook   = True,
    use_hel        = True,
    use_ot         = True,
    pooling        = "attention",
    epochs         = 10,
    lr             = 1e-3,
    device         = "cpu",
)

model = NOVAARC(encoder=DummyEncoder(), config=config)

total     = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"\n  Model built")
print(f"  Total parameters     : {total:,}")
print(f"  Trainable parameters : {trainable:,}")
print()


# ─────────────────────────────────────────────────────────────────────────────
# Step 4 — Train
# ─────────────────────────────────────────────────────────────────────────────

print("  Training ...")
print("-" * 60)
model.fit(source_loader, target_loader, verbose=True)


# ─────────────────────────────────────────────────────────────────────────────
# Step 5a — Labelled test-set report  (ground truth available)
# ─────────────────────────────────────────────────────────────────────────────

import torch.nn.functional as F

device = torch.device(config.device)
model.eval()

all_preds, all_labels = [], []
with torch.no_grad():
    for wav, lbl in test_loader:
        logits, _, _ = model(wav.to(device))
        all_preds.extend(logits.argmax(-1).cpu().tolist())
        all_labels.extend(lbl.tolist())

print("\n" + "=" * 60)
print("  [TEST SET]  Labelled evaluation report")
print("=" * 60)
print(f"  Samples     : {len(all_labels)}")
print(f"  Accuracy    : {accuracy_score(all_labels, all_preds) * 100:.2f}%")
print(f"  Weighted F1 : {f1_score(all_labels, all_preds, average='weighted'):.4f}")
print(f"  Macro F1    : {f1_score(all_labels, all_preds, average='macro'):.4f}")
print()
print(classification_report(
    all_labels,
    all_preds,
    target_names = EMOTIONS,
    digits       = 4,
    zero_division= 0,
))


# ─────────────────────────────────────────────────────────────────────────────
# Step 5b — Unlabelled target-domain report  (no ground truth)
#
# Since there are no labels we report:
#   • Predicted class distribution  — is the model spreading across classes?
#   • Per-class average confidence  — how sure is the model per class?
#   • Per-sample prediction entropy — overall model uncertainty on this domain
#   • OT soft-label distribution    — what the transport plan actually assigned
# ─────────────────────────────────────────────────────────────────────────────

tgt_preds      = []   # argmax predictions
tgt_probs      = []   # full softmax probability vectors
tgt_soft_labels= []   # OT soft labels from the final prototype bank

prototypes = model.prototype_bank.prototypes.to(device)

# Rebuild class prior from source
src_labels_all = torch.tensor([lbl for _, lbl in source_ds])
class_prior    = torch.bincount(src_labels_all, minlength=NUM_CLASSES).float()
class_prior    = class_prior / class_prior.sum()

from nova_arc.transport import compute_ot

model.eval()
with torch.no_grad():
    for wav, _ in target_loader:
        wav    = wav.to(device)
        logits, u_hyp, _ = model(wav)

        probs  = F.softmax(logits, dim=-1)          # (B, C)
        tgt_preds.extend(probs.argmax(-1).cpu().tolist())
        tgt_probs.append(probs.cpu())

        # OT soft labels for this batch
        ot_out = compute_ot(
            prototypes  = prototypes,
            target_hyp  = u_hyp.detach(),
            class_prior = class_prior.to(device),
            curvature   = config.curvature,
            eps         = config.sinkhorn_eps,
            n_iter      = config.sinkhorn_iters,
        )
        tgt_soft_labels.append(ot_out["soft_labels"].cpu())   # (B, C)

tgt_probs       = torch.cat(tgt_probs,       dim=0)   # (N, C)
tgt_soft_labels = torch.cat(tgt_soft_labels, dim=0)   # (N, C)

# Predicted class counts
pred_counts = torch.bincount(
    torch.tensor(tgt_preds), minlength=NUM_CLASSES
)

# Per-class average confidence  (mean softmax probability of predicted class)
avg_conf = torch.zeros(NUM_CLASSES)
for c in range(NUM_CLASSES):
    mask = torch.tensor(tgt_preds) == c
    if mask.any():
        avg_conf[c] = tgt_probs[mask, c].mean()

# Per-sample entropy  H = -Σ p log p  (higher → more uncertain)
entropy = -(tgt_probs * (tgt_probs + 1e-8).log()).sum(dim=-1)   # (N,)

# OT soft-label distribution  (mean soft label per class across all target samples)
ot_dist = tgt_soft_labels.mean(dim=0)   # (C,)

print("\n" + "=" * 60)
print("  [TARGET SET]  Unlabelled domain report")
print("=" * 60)
print(f"  Samples : {len(tgt_preds)}")
print()

# Prediction distribution
print(f"  {'Class':<10} {'Predicted':>10} {'Avg Conf':>10} {'OT Weight':>10}")
print(f"  {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
for c, name in enumerate(EMOTIONS):
    bar = "#" * int(pred_counts[c].item() / len(tgt_preds) * 20)
    print(
        f"  {name:<10} {pred_counts[c].item():>10}  "
        f"{avg_conf[c].item():>9.4f}  "
        f"{ot_dist[c].item():>9.4f}  {bar}"
    )

print()
print(f"  Mean prediction entropy : {entropy.mean().item():.4f}  "
      f"(max possible = {torch.log(torch.tensor(float(NUM_CLASSES))):.4f})")
print(f"  Min entropy sample      : {entropy.min().item():.4f}  (most confident)")
print(f"  Max entropy sample      : {entropy.max().item():.4f}  (most uncertain)")
print()
print("  Note: 'Predicted' = model argmax  |  'OT Weight' = transport-plan assignment")
print("        High entropy = model is uncertain on this domain (expected with DummyEncoder)")
