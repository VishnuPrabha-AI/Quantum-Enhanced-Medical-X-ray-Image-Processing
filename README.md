# Quantum-Enhanced-Medical-X-ray-Image-Processing

A hybrid classical-quantum framework for enhancing medical X-ray images using Qiskit and advanced image processing techniques.

## Overview

This project implements a complete quantum computing pipeline for medical image enhancement, combining:
- Classical preprocessing methods (Gaussian filtering, unsharp masking)
- Quantum-inspired algorithms (Fourier domain processing)
- Real quantum circuits using Qiskit (QFT, variational quantum circuits)
- Hybrid classical-quantum enhancement
- Comprehensive performance analysis and visualization

## Features

- **Classical Baseline**: Traditional image enhancement methods
- **Quantum Circuits**: Real quantum implementations with Qiskit
- **Hybrid Approach**: Best-of-both-worlds enhancement
- **Batch Processing**: Statistical analysis across multiple images
- **Comprehensive Metrics**: PSNR, SSIM, and custom quality scores
- **Visualization Dashboard**: Complete analysis and comparison tools

## Installation

```bash
# Clone the repository
git clone https://github.com/VishnuPrabha-AI/Quantum-Enhanced-Medical-X-ray-Image-Processing.git
cd quantum-medical-image-enhancement

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```

## Dataset

Download the Chest X-Ray dataset:
- [Kaggle: Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
- Extract to `data/chest_xray/`

## Quick Start

```python
from examples.run_complete_project import run_complete_quantum_project

# Run the complete pipeline
project_data = run_complete_quantum_project()

# Save results
from examples.save_results import save_project_results
save_project_results(project_data)
```

## Usage

### Basic Enhancement

```python
from src.classical.processor import XRayProcessor
from src.quantum.enhancer import QuantumMedicalImageEnhancer

# Load and process images
processor = XRayProcessor('data/chest_xray/')
images, labels = processor.load_images(max_images=20)

# Apply quantum enhancement
enhancer = QuantumMedicalImageEnhancer(patch_size=4)
enhanced = enhancer.quantum_enhance_image(images[0], method='qft_filter')
```

### Hybrid Enhancement

```python
from src.hybrid.enhancer import HybridQuantumEnhancer

hybrid = HybridQuantumEnhancer(processor)
result = hybrid.hybrid_enhance(noisy_image)
```

### Batch Analysis

```python
from src.analysis.batch_processor import BatchQuantumProcessor

batch = BatchQuantumProcessor(processor)
results, metrics = batch.process_image_batch(images, labels)
```

## Performance

| Method | Avg PSNR Improvement | Avg SSIM Improvement |
|--------|---------------------|---------------------|
| Classical | ~2.5 dB | ~0.045 |
| Quantum-Inspired | ~3.1 dB | ~0.052 |
| Hybrid | ~3.8 dB | ~0.067 |

## Project Structure

```
quantum-medical-image-enhancement/

├── classical/       # Classical image processing
├── quantum/         # Quantum circuits and enhancement
├── hybrid/          # Hybrid classical-quantum methods
├── analysis/        # Batch processing and metrics
└── visualization/   # Dashboard and plots

```

## License

MIT License - see LICENSE file for details

## Contact

For questions or collaboration: vishnuprabhakrishnakumar@gmail.com

## Acknowledgments

- Qiskit for quantum computing framework
- Chest X-Ray dataset contributors
- Quantum computing research community
"""
