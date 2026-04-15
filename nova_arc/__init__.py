"""
nova_arc
--------
NOVA-ARC: Non-Verbal to Verbal Cross-Domain Speech Emotion Recognition
via Hyperbolic Optimal Transport  —  ACL 2026

Usage
-----
    from nova_arc import NOVAARC, NOVAARCConfig

    config = NOVAARCConfig(num_classes=5, input_dim=768)
    model  = NOVAARC(config)
    model.fit(source_loader, target_loader)
    model.evaluate(test_loader)
    model.save("checkpoint.pt")

    model = NOVAARC.load("checkpoint.pt")
"""

from nova_arc.model import NOVAARC, NOVAARCConfig

__version__ = "0.1.0"
__author__  = "Girish, Mohd Mujtaba Akhtar, Muskaan Singh"

__all__ = ["NOVAARC", "NOVAARCConfig"]
