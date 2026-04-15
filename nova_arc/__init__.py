"""
nova_arc
--------
NOVA-ARC: Non-Verbal→Verbal Cross-Domain Speech Emotion Recognition
via Hyperbolic Optimal Transport.

Public API::

    from nova_arc import NOVAARC, NOVAARCConfig
    from nova_arc.encoders import WavLMEncoder, Wav2Vec2Encoder, Voc2VecEncoder, get_encoder

Minimal usage::

    encoder = WavLMEncoder(freeze=True)
    config  = NOVAARCConfig(num_classes=5, epochs=20, device="cuda")
    model   = NOVAARC(encoder=encoder, config=config)
    model.fit(source_loader, target_loader)
    metrics = model.evaluate(test_loader)
"""

from nova_arc.model    import NOVAARC, NOVAARCConfig
from nova_arc.encoders import (
    BaseEncoder,
    Wav2Vec2Encoder,
    WavLMEncoder,
    Voc2VecEncoder,
    MMSEncoder,
    get_encoder,
)

__version__ = "0.1.0"
__author__  = "Girish, Mohd Mujtaba Akhtar, Muskaan Singh"

__all__ = [
    "NOVAARC",
    "NOVAARCConfig",
    "BaseEncoder",
    "Wav2Vec2Encoder",
    "WavLMEncoder",
    "Voc2VecEncoder",
    "MMSEncoder",
    "get_encoder",
]
