from setuptools import setup, find_packages

setup(
    name="senri-llm",
    version="0.1.0",
    description="Senri: Orthogonal Basis Routed Infinite Attention for Ultra-Long Context",
    author="",
    python_requires=">=3.9",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.36.0",
        "accelerate>=0.25.0",
        "datasets>=2.14.0",
        "sentencepiece>=0.1.99",
        "einops>=0.7.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=23.0.0",
            "isort>=5.12.0",
        ],
    },
)
