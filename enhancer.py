import numpy as np
from .circuits import QuantumImageProcessor


class QuantumMedicalImageEnhancer:
    """Complete quantum medical image enhancement system."""
    
    def __init__(self, patch_size=4):
        """
        Initialize quantum medical image enhancer.
        
        Args:
            patch_size: Size of image patches for processing
        """
        self.patch_size = patch_size
        self.quantum_processor = QuantumImageProcessor(n_qubits=int(np.log2(patch_size**2)))
        
    def quantum_enhance_image(self, image, method='qft_filter'):
        """
        Enhance full image using quantum patch processing.
        
        Args:
            image: Input image array
            method: Enhancement method ('qft_filter' or 'variational')
            
        Returns:
            Enhanced image
        """
        rows, cols = image.shape
        enhanced = np.zeros_like(image)
        weight_map = np.zeros_like(image)
        
        # Process non-overlapping patches
        for i in range(0, rows - self.patch_size + 1, self.patch_size):
            for j in range(0, cols - self.patch_size + 1, self.patch_size):
                patch = image[i:i+self.patch_size, j:j+self.patch_size]
                
                try:
                    enhanced_patch = self.quantum_processor.process_image_patch(patch, method)
                    enhanced[i:i+self.patch_size, j:j+self.patch_size] += enhanced_patch
                    weight_map[i:i+self.patch_size, j:j+self.patch_size] += 1
                except Exception as e:
                    enhanced[i:i+self.patch_size, j:j+self.patch_size] += patch
                    weight_map[i:i+self.patch_size, j:j+self.patch_size] += 1
        
        mask = weight_map > 0
        enhanced[mask] = enhanced[mask] / weight_map[mask]
        
        return enhanced
