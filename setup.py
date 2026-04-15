from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name             = "nova-arc",
    version          = "0.1.0",
    author           = "Girish, Mohd Mujtaba Akhtar, Muskaan Singh",
    description      = (
        "NOVA-ARC: Non-Verbal→Verbal Cross-Domain Speech Emotion Recognition "
        "via Hyperbolic Optimal Transport (ACL 2026)"
    ),
    long_description = long_description,
    long_description_content_type = "text/markdown",
    url              = "https://github.com/girish281003/NOVA-ARC",
    packages         = find_packages(),
    python_requires  = ">=3.8",
    install_requires = [
        "torch>=2.0",
        "transformers>=4.35",
        "scikit-learn>=1.3",
        "pyyaml>=6.0",
        "tqdm>=4.65",
    ],
    extras_require   = {
        "dev": ["pytest", "black", "isort"],
    },
    classifiers      = [
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
