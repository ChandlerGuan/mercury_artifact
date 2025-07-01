"""
Setup configuration for Mercury package.
"""
from setuptools import setup, find_packages

setup(
    name="mercury",
    version="0.1.0",
    description="A compiler for auto-distributing tensor computations",
    packages=find_packages(),
    python_requires=">=3.8",
    author="mercury developers",
    install_requires=[
        "torch>=2.6.0",
        "numpy>=1.21.0",
        "flash-attn>=2.7.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pylint>=2.0.0",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)