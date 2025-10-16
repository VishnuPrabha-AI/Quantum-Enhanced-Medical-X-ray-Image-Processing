import numpy as np
from src.hybrid.enhancer import HybridQuantumEnhancer


class BatchQuantumProcessor:
    """Process multiple images and analyze patterns."""
    
    def __init__(self, processor):
        """
        Initialize batch processor.
        
        Args:
            processor: Instance of XRayProcessor
        """
        self.processor = processor
        
    def process_image_batch(self, images, labels, methods=['classical', 'quantum-inspired', 'hybrid']):
        """
        Process a batch of images with different methods.
        
        Args:
            images: Array of images
            labels: List of image labels
            methods: List of enhancement methods to apply
            
        Returns:
            Tuple of (results dict, metrics dict)
        """
        results = {method: [] for method in methods}
        metrics = {method: {'psnr': [], 'ssim': []} for method in methods}
        
        # Initialize hybrid enhancer
        hybrid_enhancer = HybridQuantumEnhancer(self.processor)
        
        for i, original in enumerate(images[:5]):
            print(f"Processing image {i+1}/5...")
            
            # Add noise
            noisy = self.processor.add_noise([original])[0]
            
            # Apply each method
            for method in methods:
                try:
                    if method == 'classical':
                        enhanced = self.processor.classical_enhancement([noisy])[0]
                    elif method == 'quantum-inspired':
                        enhanced = self.processor.quantum_fourier_enhancement(noisy)
                    elif method == 'hybrid':
                        enhanced = hybrid_enhancer.hybrid_enhance(noisy)
                    else:
                        enhanced = noisy
                        
                    results[method].append(enhanced)
                    
                    # Calculate metrics
                    eval_metrics = self.processor.evaluate_enhancement(original, enhanced, noisy)
                    metrics[method]['psnr'].append(eval_metrics['psnr_improvement'])
                    metrics[method]['ssim'].append(eval_metrics['ssim_improvement'])
                    
                except Exception as e:
                    print(f"Error processing image {i} with {method}: {e}")
                    results[method].append(noisy)
                    metrics[method]['psnr'].append(0)
                    metrics[method]['ssim'].append(0)
        
        return results, metrics
    
    def analyze_performance_patterns(self, metrics):
        """
        Analyze which methods work best for different image types.
        
        Args:
            metrics: Dictionary of metrics from batch processing
            
        Returns:
            Analysis dictionary with statistics
        """
        analysis = {}
        
        for method, data in metrics.items():
            psnr_values = data['psnr']
            ssim_values = data['ssim']
            
            analysis[method] = {
                'avg_psnr_improvement': np.mean(psnr_values),
                'std_psnr_improvement': np.std(psnr_values),
                'avg_ssim_improvement': np.mean(ssim_values),
                'std_ssim_improvement': np.std(ssim_values),
                'consistency_score': 1 / (1 + np.std(psnr_values))
            }
            
        return analysis
