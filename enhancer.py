import numpy as np
from .circuits import QuantumImageProcessor
from scipy import ndimage
from qiskit import QuantumCircuit, transpile
from qiskit_aer import Aer
import warnings
warnings.filterwarnings('ignore')

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
class HybridQuantumEnhancer:
    """Combines classical preprocessing with quantum enhancement."""
    
    def __init__(self, classical_processor, n_qubits=4):
        """
        Initialize hybrid quantum enhancer.
        
        Args:
            classical_processor: Instance of XRayProcessor
            n_qubits: Number of qubits for quantum processing
        """
        self.classical_processor = classical_processor
        self.n_qubits = n_qubits
        self.backend = Aer.get_backend('statevector_simulator')
        
    def hybrid_enhance(self, noisy_image, classical_method='gaussian'):
        """
        Apply classical preprocessing followed by quantum enhancement.
        
        Args:
            noisy_image: Noisy input image
            classical_method: Classical preprocessing method
            
        Returns:
            Enhanced image
        """
        # Step 1: Classical preprocessing
        if classical_method == 'gaussian':
            preprocessed = self.classical_processor.classical_enhancement([noisy_image])[0]
        else:
            preprocessed = noisy_image
            
        # Step 2: Quantum post-processing on selected regions
        enhanced = self._quantum_region_enhancement(preprocessed)
        
        return enhanced
    
    def _quantum_region_enhancement(self, image):
        """
        Apply quantum enhancement to high-contrast regions only.
        
        Args:
            image: Preprocessed image
            
        Returns:
            Enhanced image with quantum processing
        """
        # Detect edges/important regions
        edges = ndimage.sobel(image)
        high_contrast_mask = edges > np.percentile(edges, 75)
        
        # Apply quantum enhancement only to important regions
        enhanced = image.copy()
        rows, cols = image.shape
        
        for i in range(0, rows-4, 8):
            for j in range(0, cols-4, 8):
                if high_contrast_mask[i:i+4, j:j+4].sum() > 8:
                    patch = image[i:i+4, j:j+4]
                    try:
                        quantum_patch = self._apply_quantum_processing(patch)
                        enhanced[i:i+4, j:j+4] = quantum_patch
                    except:
                        pass
                        
        return enhanced
    
    def _apply_quantum_processing(self, patch):
        """
        Apply quantum processing to a single patch.
        
        Args:
            patch: Image patch
            
        Returns:
            Quantum-processed patch
        """
        flat_patch = patch.flatten()
        target_size = 2**self.n_qubits
        
        if len(flat_patch) != target_size:
            indices = np.linspace(0, len(flat_patch)-1, target_size)
            flat_patch = np.interp(indices, np.arange(len(flat_patch)), flat_patch)
        
        # Normalize
        norm = np.sqrt(np.sum(flat_patch**2))
        if norm > 0:
            amplitudes = flat_patch / norm
        else:
            return patch
            
        # Apply quantum circuit
        qc = QuantumCircuit(self.n_qubits)
        qc.initialize(amplitudes, range(self.n_qubits))
        
        # Simple enhancement circuit
        for i in range(self.n_qubits):
            qc.ry(np.pi/6, i)
            
        for i in range(self.n_qubits-1):
            qc.cx(i, i+1)
            
        # Execute
        job = self.backend.run(transpile(qc, self.backend))
        result = job.result()
        statevector = result.get_statevector()
        processed = np.abs(statevector.data)
        
        # Reshape and normalize back
        processed_patch = processed[:16].reshape(4, 4)
        processed_patch = (processed_patch - processed_patch.min()) / (processed_patch.max() - processed_patch.min())
        
        return processed_patch
