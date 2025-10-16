from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="quantum-medical-enhancement",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Quantum computing for medical image enhancement",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/quantum-medical-image-enhancement",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Topic :: Scientific/Engineering :: Physics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "matplotlib>=3.4.0",
        "pillow>=8.3.0",
        "opencv-python>=4.5.0",
        "scikit-image>=0.18.0",
        "scikit-learn>=0.24.0",
        "scipy>=1.7.0",
        "qiskit>=0.45.0",
        "qiskit-aer>=0.13.0",
    ],
)
