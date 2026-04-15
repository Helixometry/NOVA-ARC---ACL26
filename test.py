"""
test.py
-------
End-to-end test of the NOVA-ARC library.

This file shows exactly how a user works with the library:

  Step 1  —  Extract features using ANY tool you like.
             (Here we simulate WavLM-style 768-dim frame features.)

  Step 2  —  Build labelled and unlabelled DataLoaders.

  Step 3  —  Import NOVA-ARC, set config, build model.

  Step 4  —  Train.

  Step 5  —  Evaluate: labelled test report + unlabelled target report.

  Step 6  —  Save checkpoint, reload, verify.

Run:
    python test.py
"""

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, accuracy_score, f1_score

from nova_arc import NOVAARC, NOVAARCConfig


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — Feature extraction  (done OUTSIDE the library)
#
# In a real experiment you would do something like:
#
#     from transformers import WavLMModel
#     model = WavLMModel.from_pretrained("microsoft/wavlm-base-plus")
#     with torch.no_grad():
#         features = model(waveforms).last_hidden_state  # (B, T, 768)
#
# Here we simulate that output with random tensors of the same shape.
# ─────────────────────────────────────────────────────────────────────────────

EMOTIONS    = ["neutral", "happy", "sad", "angry", "fear"]
NUM_CLASSES = len(EMOTIONS)     # 5  ← your number of classes
FEAT_DIM    = 768               # ← dimension of your feature extractor output
N_FRAMES    = 50                # ← number of time frames per utterance


class FeatureDataset(Dataset):
    """
    Simulates pre-extracted frame-level features.

    In real use, replace this with a Dataset that loads your
    already-extracted features from disk  (e.g. .npy files, .pt files).

    Each item: (features, label)
        features : (N_FRAMES, FEAT_DIM)   float32
        label    : int  0 … NUM_CLASSES-1   (or -1 for unlabelled)
    """
    def __init__(self, n_samples=300, labelled=True, seed=0):
        rng = torch.Generator().manual_seed(seed)
        self.features = torch.randn(n_samples, N_FRAMES, FEAT_DIM, generator=rng)
        self.labels   = torch.arange(n_samples) % NUM_CLASSES
        self.labelled = labelled

        # Add a tiny per-class signal so the model can learn something real
        for c in range(NUM_CLASSES):
            self.features[self.labels == c] += 0.03 * c

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feat  = self.features[idx]              # (N_FRAMES, FEAT_DIM)
        label = int(self.labels[idx]) if self.labelled else -1
        return feat, label


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — Build DataLoaders
# ─────────────────────────────────────────────────────────────────────────────

source_loader = DataLoader(
    FeatureDataset(n_samples=400, labelled=True,  seed=0),
    batch_size=32, shuffle=True,
)
target_loader = DataLoader(
    FeatureDataset(n_samples=200, labelled=False, seed=1),   # unlabelled
    batch_size=32, shuffle=True,
)
test_loader = DataLoader(
    FeatureDataset(n_samples=100, labelled=True,  seed=2),
    batch_size=32, shuffle=False,
)

print("=" * 62)
print("  NOVA-ARC  —  Library Test")
print("=" * 62)
print(f"  Feature dim   : {FEAT_DIM}   (e.g. WavLM output)")
print(f"  Classes       : {NUM_CLASSES}  -> {EMOTIONS}")
print(f"  Source        : {len(source_loader.dataset)} labelled samples")
print(f"  Target        : {len(target_loader.dataset)} unlabelled samples")
print(f"  Test          : {len(test_loader.dataset)} labelled samples")


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 — Configure and build NOVA-ARC
#
# Change any parameter freely.  Only num_classes and input_dim are required.
# ─────────────────────────────────────────────────────────────────────────────

config = NOVAARCConfig(
    # ── Required ───────────────────────────────────────────────
    num_classes    = NUM_CLASSES,   # must match your dataset
    input_dim      = FEAT_DIM,      # must match your feature extractor

    # ── Architecture ───────────────────────────────────────────
    use_hyperbolic = True,          # Poincaré ball geometry
    use_codebook   = True,          # HVQ prosody codebook
    use_hel        = True,          # Hyperbolic Emotion Lens
    use_ot         = True,          # Sinkhorn OT domain adaptation
    pooling        = "attention",   # "attention" | "mean" | "max"

    # ── Dimensions ─────────────────────────────────────────────
    hidden_dim     = 256,           # internal Poincaré ball dim
    curvature      = 1.0,

    # ── HVQ Codebook ───────────────────────────────────────────
    codebook_size     = 256,
    commitment_weight = 0.25,

    # ── Optimal Transport ──────────────────────────────────────
    sinkhorn_eps   = 0.05,
    sinkhorn_iters = 50,
    lambda_opt     = 1.0,
    lambda_ce      = 1.0,

    # ── Training ───────────────────────────────────────────────
    lr     = 1e-3,
    epochs = 10,
    device = "cpu",
)

model = NOVAARC(config)

total_p     = sum(p.numel() for p in model.parameters())
trainable_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\n  Parameters    : {total_p:,} total  |  {trainable_p:,} trainable")


# ─────────────────────────────────────────────────────────────────────────────
# Step 4 — Train
# ─────────────────────────────────────────────────────────────────────────────

print("\n  Training ...")
print("-" * 62)
model.fit(source_loader, target_loader, verbose=True)


# ─────────────────────────────────────────────────────────────────────────────
# Step 5a — Labelled test-set report
# ─────────────────────────────────────────────────────────────────────────────

device = torch.device(config.device)
model.eval()

all_preds, all_labels = [], []
with torch.no_grad():
    for feat, lbl in test_loader:
        logits, _, _ = model(feat.to(device))
        all_preds.extend(logits.argmax(-1).cpu().tolist())
        all_labels.extend(lbl.tolist())

print("\n" + "=" * 62)
print("  [TEST SET]  Labelled evaluation")
print("=" * 62)
print(f"  Accuracy    : {accuracy_score(all_labels, all_preds) * 100:.2f}%")
print(f"  Weighted F1 : {f1_score(all_labels, all_preds, average='weighted', zero_division=0):.4f}")
print(f"  Macro F1    : {f1_score(all_labels, all_preds, average='macro',    zero_division=0):.4f}")
print()
print(classification_report(all_labels, all_preds, target_names=EMOTIONS,
                             digits=4, zero_division=0))


# ─────────────────────────────────────────────────────────────────────────────
# Step 5b — Unlabelled target-domain report
# ─────────────────────────────────────────────────────────────────────────────

from nova_arc.transport import compute_ot

prototypes  = model.prototype_bank.prototypes.to(device)
src_labels  = torch.tensor([int(lbl) for _, lbl in source_loader.dataset])
class_prior = torch.bincount(src_labels, minlength=NUM_CLASSES).float()
class_prior = class_prior / class_prior.sum()

tgt_preds, tgt_probs, tgt_soft = [], [], []

model.eval()
with torch.no_grad():
    for feat, _ in target_loader:
        feat   = feat.to(device)
        logits, u_hyp, _ = model(feat)
        probs  = F.softmax(logits, dim=-1)

        tgt_preds.extend(probs.argmax(-1).cpu().tolist())
        tgt_probs.append(probs.cpu())

        ot_out = compute_ot(
            prototypes  = prototypes,
            target_hyp  = u_hyp.detach(),
            class_prior = class_prior.to(device),
            curvature   = config.curvature,
            eps         = config.sinkhorn_eps,
            n_iter      = config.sinkhorn_iters,
        )
        tgt_soft.append(ot_out["soft_labels"].cpu())

tgt_probs = torch.cat(tgt_probs)   # (N, C)
tgt_soft  = torch.cat(tgt_soft)    # (N, C)

pred_counts = torch.bincount(torch.tensor(tgt_preds), minlength=NUM_CLASSES)
avg_conf    = torch.zeros(NUM_CLASSES)
for c in range(NUM_CLASSES):
    mask = torch.tensor(tgt_preds) == c
    if mask.any():
        avg_conf[c] = tgt_probs[mask, c].mean()

entropy  = -(tgt_probs * (tgt_probs + 1e-8).log()).sum(dim=-1)
ot_dist  = tgt_soft.mean(dim=0)

print("=" * 62)
print("  [TARGET SET]  Unlabelled domain report")
print("=" * 62)
print(f"  Samples : {len(tgt_preds)}")
print()
print(f"  {'Class':<10} {'Predicted':>10}  {'Avg Conf':>9}  {'OT Weight':>9}  Distribution")
print(f"  {'-'*10} {'-'*10}  {'-'*9}  {'-'*9}  {'-'*20}")
for c, name in enumerate(EMOTIONS):
    bar = "#" * int(pred_counts[c].item() / len(tgt_preds) * 20)
    print(
        f"  {name:<10} {pred_counts[c].item():>10}  "
        f"{avg_conf[c].item():>9.4f}  "
        f"{ot_dist[c].item():>9.4f}  {bar}"
    )

print()
print(f"  Mean entropy : {entropy.mean():.4f}  (max = {torch.log(torch.tensor(float(NUM_CLASSES))):.4f})")
print(f"  Min entropy  : {entropy.min():.4f}  (most confident sample)")
print(f"  Max entropy  : {entropy.max():.4f}  (most uncertain sample)")


# ─────────────────────────────────────────────────────────────────────────────
# Step 6 — Save checkpoint and reload
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 62)
print("  Checkpoint")
print("=" * 62)

model.save("checkpoints/nova_arc.pt")

reloaded = NOVAARC.load("checkpoints/nova_arc.pt", map_location="cpu")

orig_acc     = model.evaluate(test_loader)["accuracy"]
reloaded_acc = reloaded.evaluate(test_loader)["accuracy"]
match        = "OK" if abs(orig_acc - reloaded_acc) < 1e-5 else "MISMATCH"
print(f"  Original  accuracy : {orig_acc * 100:.2f}%")
print(f"  Reloaded  accuracy : {reloaded_acc * 100:.2f}%  [{match}]")
