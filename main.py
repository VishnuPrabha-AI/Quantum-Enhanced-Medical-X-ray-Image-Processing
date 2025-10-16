import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.classical.processor import XRayProcessor
from src.quantum.enhancer import QuantumMedicalImageEnhancer
from src.hybrid.enhancer import HybridQuantumEnhancer
from src.analysis.batch_processor import BatchQuantumProcessor
from src.visualization.dashboard import ComprehensiveVisualization


def run_complete_quantum_project():
    """Run the complete quantum medical image enhancement project."""
    print("="*60)
    print("COMPLETE QUANTUM MEDICAL IMAGE ENHANCEMENT PROJECT")
    print("="*60)
    
    # CHANGE THIS PATH TO YOUR DATASET LOCATION
    data_path = r"data/chest_xray"
    
    # Step 1: Initialize and load data
    print("\n1. FOUNDATION: Loading medical images...")
    processor = XRayProcessor(data_path)
    images, labels = processor.load_images(max_images=20)
    
    if len(images) == 0:
        print("ERROR: No images loaded! Please check your data path.")
        print(f"Expected path: {data_path}")
        return None
    
    # Use first image for detailed analysis
    original_img = images[0]
    noisy_img = processor.add_noise([original_img], noise_level=0.15)[0]
    
    # Step 2: Classical methods
    print("\n2. CLASSICAL METHODS: Baseline enhancement...")
    classical_enhanced = processor.classical_enhancement([noisy_img])[0]
    quantum_inspired = processor.quantum_fourier_enhancement(noisy_img)
    
    # Step 3: Real quantum circuits
    print("\n3. QUANTUM CIRCUITS: Real Qiskit implementation...")
    quantum_enhancer = QuantumMedicalImageEnhancer(patch_size=4)
    
    quantum_qft = quantum_enhancer.quantum_enhance_image(noisy_img, method='qft_filter')
    quantum_var = quantum_enhancer.quantum_enhance_image(noisy_img, method='variational')
    
    # Step 4: Hybrid approach
    print("\n4. HYBRID METHOD: Classical-quantum combination...")
    hybrid_enhancer = HybridQuantumEnhancer(processor)
    hybrid_result = hybrid_enhancer.hybrid_enhance(noisy_img)
    
    # Step 5: Batch processing
    print("\n5. BATCH ANALYSIS: Multi-image statistical evaluation...")
    batch_processor = BatchQuantumProcessor(processor)
    
    batch_results, batch_metrics = batch_processor.process_image_batch(
        images, labels, methods=['classical', 'quantum-inspired', 'hybrid']
    )
    
    # Compile all results
    single_image_results = {
        'classical': classical_enhanced,
        'quantum-inspired': quantum_inspired,
        'quantum-qft': quantum_qft,
        'quantum-variational': quantum_var,
        'hybrid': hybrid_result
    }
    
    # Calculate single image metrics
    single_image_metrics = {}
    for method, enhanced in single_image_results.items():
        metrics = processor.evaluate_enhancement(original_img, enhanced, noisy_img)
        single_image_metrics[method] = {
            'psnr': [metrics['psnr_improvement']], 
            'ssim': [metrics['ssim_improvement']]
        }
    
    # Step 6: Performance analysis
    print("\n6. PERFORMANCE ANALYSIS:")
    analysis = batch_processor.analyze_performance_patterns(batch_metrics)
    
    print("\nBatch Processing Results:")
    for method, stats in analysis.items():
        print(f"{method.upper()}:")
        print(f"  Average PSNR improvement: {stats['avg_psnr_improvement']:.2f} ± {stats['std_psnr_improvement']:.2f} dB")
        print(f"  Average SSIM improvement: {stats['avg_ssim_improvement']:.3f} ± {stats['std_ssim_improvement']:.3f}")
        print(f"  Consistency score: {stats['consistency_score']:.3f}")
    
    print("\nSingle Image Results:")
    for method, metrics in single_image_metrics.items():
        print(f"{method.upper()}:")
        print(f"  PSNR improvement: {metrics['psnr'][0]:.2f} dB")
        print(f"  SSIM improvement: {metrics['ssim'][0]:.3f}")
    
    # Step 7: Comprehensive visualization
    print("\n7. VISUALIZATION: Creating comprehensive dashboard...")
    visualizer = ComprehensiveVisualization()
    
    # Combine metrics for visualization
    combined_metrics = {}
    for method in single_image_metrics.keys():
        if method in batch_metrics:
            combined_metrics[method] = batch_metrics[method]
        else:
            combined_metrics[method] = single_image_metrics[method]
    
    # Create main dashboard
    visualizer.create_complete_dashboard(original_img, noisy_img, single_image_results, combined_metrics)
    
    # Create quantum analysis
    visualizer.plot_quantum_analysis()
    
    
    print("\nBEST PERFORMING METHODS:")
    best_psnr = max(analysis.items(), key=lambda x: x[1]['avg_psnr_improvement'])
    best_ssim = max(analysis.items(), key=lambda x: x[1]['avg_ssim_improvement'])
    most_consistent = max(analysis.items(), key=lambda x: x[1]['consistency_score'])
    
    print(f"  Best PSNR improvement: {best_psnr[0]} ({best_psnr[1]['avg_psnr_improvement']:.2f} dB)")
    print(f"  Best SSIM improvement: {best_ssim[0]} ({best_ssim[1]['avg_ssim_improvement']:.3f})")
    print(f"  Most consistent: {most_consistent[0]} (score: {most_consistent[1]['consistency_score']:.3f})")
    
    return {
        'processor': processor,
        'images': images,
        'labels': labels,
        'single_results': single_image_results,
        'batch_results': batch_results,
        'metrics': combined_metrics,
        'analysis': analysis,
        'visualizer': visualizer
    }


if __name__ == "__main__":
    run_complete_quantum_project()

