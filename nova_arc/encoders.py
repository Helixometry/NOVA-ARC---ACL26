"""
nova_arc/encoders.py
--------------------
Encoder interface + built-in wrappers for common SSL speech models.

Users can either:
  1. Use a built-in encoder  (Wav2Vec2Encoder, WavLMEncoder, etc.)
  2. Subclass BaseEncoder with their own model

The only contract:
  - Input  : raw waveform tensor  (batch, samples)  at 16 kHz
  - Output : frame-level features (batch, T, dim)

Example — custom encoder:
    class MyEncoder(BaseEncoder):
        def __init__(self):
            super().__init__(output_dim=512)
            self.backbone = ...

        def forward(self, waveforms):      # (B, T_samples)
            return self.backbone(waveforms) # (B, T_frames, 512)
"""

from abc import ABC, abstractmethod
import torch
import torch.nn as nn


# ─── Base class ──────────────────────────────────────────────────────────────

class BaseEncoder(nn.Module, ABC):
    """
    Abstract base encoder.  All encoders must inherit from this class.

    Args:
        output_dim (int): dimension of frame-level output features.
        freeze (bool):    if True the encoder weights are frozen (default True).
    """

    def __init__(self, output_dim: int, freeze: bool = True):
        super().__init__()
        self.output_dim = output_dim
        self._freeze    = freeze

    def _apply_freeze(self):
        """Call after loading backbone weights."""
        if self._freeze:
            for p in self.parameters():
                p.requires_grad = False

    @abstractmethod
    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
        """
        Args:
            waveforms: (B, T_samples) float tensor, 16 kHz, normalised to [-1, 1]

        Returns:
            (B, T_frames, output_dim) frame-level feature tensor
        """
        raise NotImplementedError


# ─── Built-in encoders ───────────────────────────────────────────────────────

class Wav2Vec2Encoder(BaseEncoder):
    """
    facebook/wav2vec2-base  (or any wav2vec2 checkpoint on HuggingFace).
    Output dim: 768
    """

    def __init__(self, model_name: str = "facebook/wav2vec2-base", freeze: bool = True):
        super().__init__(output_dim=768, freeze=freeze)
        try:
            from transformers import Wav2Vec2Model
        except ImportError:
            raise ImportError("Install transformers: pip install transformers")

        self.backbone = Wav2Vec2Model.from_pretrained(model_name)
        self._apply_freeze()

    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
        out = self.backbone(waveforms, output_hidden_states=False)
        return out.last_hidden_state   # (B, T, 768)


class WavLMEncoder(BaseEncoder):
    """
    microsoft/wavlm-base-plus  (or any WavLM checkpoint).
    Output dim: 768
    """

    def __init__(self, model_name: str = "microsoft/wavlm-base-plus", freeze: bool = True):
        super().__init__(output_dim=768, freeze=freeze)
        try:
            from transformers import WavLMModel
        except ImportError:
            raise ImportError("Install transformers: pip install transformers")

        self.backbone = WavLMModel.from_pretrained(model_name)
        self._apply_freeze()

    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
        out = self.backbone(waveforms, output_hidden_states=False)
        return out.last_hidden_state   # (B, T, 768)


class Voc2VecEncoder(BaseEncoder):
    """
    voc2vec — foundation model for non-verbal vocalizations.
    Built on the wav2vec 2.0 base architecture.
    Output dim: 768

    Checkpoint: https://github.com/koudounasalkis/voc2vec
    Pass the local path to the downloaded checkpoint as `model_path`.
    """

    def __init__(self, model_path: str, freeze: bool = True):
        super().__init__(output_dim=768, freeze=freeze)
        try:
            from transformers import Wav2Vec2Model, Wav2Vec2Config
        except ImportError:
            raise ImportError("Install transformers: pip install transformers")

        self.backbone = Wav2Vec2Model.from_pretrained(model_path)
        self._apply_freeze()

    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
        out = self.backbone(waveforms, output_hidden_states=False)
        return out.last_hidden_state   # (B, T, 768)


class MMSEncoder(BaseEncoder):
    """
    facebook/mms-1b — massively multilingual wav2vec 2.0.
    Output dim: 1024
    """

    def __init__(self, model_name: str = "facebook/mms-1b", freeze: bool = True):
        super().__init__(output_dim=1024, freeze=freeze)
        try:
            from transformers import Wav2Vec2Model
        except ImportError:
            raise ImportError("Install transformers: pip install transformers")

        self.backbone = Wav2Vec2Model.from_pretrained(model_name)
        self._apply_freeze()

    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
        out = self.backbone(waveforms, output_hidden_states=False)
        return out.last_hidden_state   # (B, T, 1024)


# ─── Registry ────────────────────────────────────────────────────────────────

ENCODER_REGISTRY = {
    "wav2vec2": Wav2Vec2Encoder,
    "wavlm":    WavLMEncoder,
    "voc2vec":  Voc2VecEncoder,
    "mms":      MMSEncoder,
}


def get_encoder(name: str, **kwargs) -> BaseEncoder:
    """
    Convenience factory.

    Example:
        enc = get_encoder("wavlm")
        enc = get_encoder("wav2vec2", model_name="facebook/wav2vec2-large")
    """
    name = name.lower()
    if name not in ENCODER_REGISTRY:
        raise ValueError(
            f"Unknown encoder '{name}'. "
            f"Available: {list(ENCODER_REGISTRY.keys())}. "
            f"Or subclass BaseEncoder for a custom encoder."
        )
    return ENCODER_REGISTRY[name](**kwargs)
