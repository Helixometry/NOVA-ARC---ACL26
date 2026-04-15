"""
examples/quick_start.py
-----------------------
Shows all the ways to use NOVA-ARC.

Run::
    python examples/quick_start.py
"""

import torch
from torch.utils.data import Dataset, DataLoader
from nova_arc import NOVAARC, NOVAARCConfig
from nova_arc.encoders import BaseEncoder
import torch.nn as nn


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Prepare your dataset
# ─────────────────────────────────────────────────────────────────────────────
# Your Dataset must return:
#   waveform : (T_samples,)  float32, 16 kHz, normalised to [-1, 1]
#   label    : int           class index  0 … num_classes-1

class MyEmotionDataset(Dataset):
    """Replace with your real dataset."""
    def __init__(self, n=200, sr=16_000, duration=2, num_classes=5):
        self.n   = n
        self.len = sr * duration
        self.C   = num_classes

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        wav = torch.randn(self.len)
        lbl = torch.randint(0, self.C, ()).item()
        return wav, lbl


NUM_CLASSES = 5
source_loader = DataLoader(MyEmotionDataset(400, num_classes=NUM_CLASSES), batch_size=16, shuffle=True)
target_loader = DataLoader(MyEmotionDataset(200, num_classes=NUM_CLASSES), batch_size=16, shuffle=True)
test_loader   = DataLoader(MyEmotionDataset(100, num_classes=NUM_CLASSES), batch_size=16, shuffle=False)


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Pick / build your encoder
# ─────────────────────────────────────────────────────────────────────────────

# ── Option A: built-in encoders (download from HuggingFace on first use) ──
#   from nova_arc.encoders import WavLMEncoder, Wav2Vec2Encoder, Voc2VecEncoder
#   encoder = WavLMEncoder(freeze=True)              # output_dim=768
#   encoder = Wav2Vec2Encoder(freeze=True)           # output_dim=768
#   encoder = Voc2VecEncoder("path/to/checkpoint")  # output_dim=768

# ── Option B: write your own encoder ──────────────────────────────────────

class MyEncoder(BaseEncoder):
    """
    Minimal custom encoder.
    Inherit BaseEncoder, set output_dim, implement forward().
    forward() must return (B, T_frames, output_dim).
    """
    def __init__(self):
        super().__init__(output_dim=128)
        self.net = nn.Sequential(
            nn.Conv1d(1, 64,  kernel_size=400, stride=160),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )

    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
        # waveforms: (B, T_samples)
        x = waveforms.unsqueeze(1)     # (B, 1, T)
        x = self.net(x)                # (B, 128, T')
        return x.permute(0, 2, 1)      # (B, T', 128)

encoder = MyEncoder()


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Configure the model
# ─────────────────────────────────────────────────────────────────────────────
# num_classes is the ONLY required field — everything else has sensible defaults.
# Toggle components on/off freely.

# ── Full NOVA-ARC (paper setting) ──────────────────────────────────────────
config_full = NOVAARCConfig(
    num_classes    = NUM_CLASSES,   # ← must match your dataset
    hyperbolic_dim = 128,           # must match encoder.output_dim here
    use_hyperbolic = True,          # Poincaré ball geometry
    use_codebook   = True,          # HVQ prosody codebook
    use_hel        = True,          # Hyperbolic Emotion Lens
    use_ot         = True,          # Sinkhorn OT domain adaptation
    pooling        = "attention",   # learnable attention pooling
    epochs         = 3,             # increase for real training
    device         = "cpu",
)

# ── Pure Euclidean (no hyperbolic, no OT) ─────────────────────────────────
config_euclidean = NOVAARCConfig(
    num_classes    = NUM_CLASSES,
    hyperbolic_dim = 128,
    use_hyperbolic = False,   # ← disables codebook + HEL automatically
    use_ot         = False,
    pooling        = "mean",
    epochs         = 3,
    device         = "cpu",
)

# ── Hyperbolic, source-only (no OT) ────────────────────────────────────────
config_no_ot = NOVAARCConfig(
    num_classes    = NUM_CLASSES,
    hyperbolic_dim = 128,
    use_hyperbolic = True,
    use_codebook   = True,
    use_hel        = True,
    use_ot         = False,   # ← no target domain needed
    epochs         = 3,
    device         = "cpu",
)

# ── Ablate HEL only ────────────────────────────────────────────────────────
config_no_hel = NOVAARCConfig(
    num_classes    = NUM_CLASSES,
    hyperbolic_dim = 128,
    use_hel        = False,
    epochs         = 3,
    device         = "cpu",
)

# ── Ablate codebook only ───────────────────────────────────────────────────
config_no_cb = NOVAARCConfig(
    num_classes    = NUM_CLASSES,
    hyperbolic_dim = 128,
    use_codebook   = False,
    epochs         = 3,
    device         = "cpu",
)

# ── Max pooling instead of attention ──────────────────────────────────────
config_maxpool = NOVAARCConfig(
    num_classes    = NUM_CLASSES,
    hyperbolic_dim = 128,
    pooling        = "max",
    epochs         = 3,
    device         = "cpu",
)


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Build + train
# ─────────────────────────────────────────────────────────────────────────────

print("=" * 60)
print("Full NOVA-ARC (hyperbolic + OT)")
print("=" * 60)
model = NOVAARC(encoder=MyEncoder(), config=config_full)
model.fit(source_loader, target_loader, verbose=True)
print(model.evaluate(test_loader))

print("\n" + "=" * 60)
print("Euclidean baseline (no hyperbolic, no OT)")
print("=" * 60)
model_euc = NOVAARC(encoder=MyEncoder(), config=config_euclidean)
model_euc.fit(source_loader, verbose=True)   # no target_loader needed
print(model_euc.evaluate(test_loader))

print("\n" + "=" * 60)
print("Source-only hyperbolic (no OT)")
print("=" * 60)
model_src = NOVAARC(encoder=MyEncoder(), config=config_no_ot)
model_src.fit(source_loader, verbose=True)
print(model_src.evaluate(test_loader))


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Predict
# ─────────────────────────────────────────────────────────────────────────────
preds = model.predict(test_loader)
print(f"\nPredictions (first 10): {preds[:10].tolist()}")


# ─────────────────────────────────────────────────────────────────────────────
# 6.  Save & reload
# ─────────────────────────────────────────────────────────────────────────────
model.save("checkpoints/nova_arc.pt")

reloaded = NOVAARC.load(
    "checkpoints/nova_arc.pt",
    encoder      = MyEncoder(),
    map_location = "cpu",
)
print(f"\nReloaded accuracy: {reloaded.evaluate(test_loader)['accuracy']:.4f}")
