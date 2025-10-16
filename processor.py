import os
import numpy as np
from PIL import Image
from skimage import filters
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import warnings
warnings.filterwarnings('ignore')


class XRayProcessor:
    """Classical X-ray image processor with enhancement and evaluation methods."""
    
    def __init__(self, data_path):
        """
        Initialize the X-ray image processor.
        
        Args:
            data_path: Path to the chest X-ray dataset
        """
        self.data_path = data_path
        self.images = []
        self.labels = []
        
    def load_images(self, max_images=50):
        """
        Load a subset of images for processing.
        
        Args:
            max_images: Maximum number of images to load
            
        Returns:
            Tuple of (images array, labels list)
        """
        categories = ['NORMAL', 'PNEUMONIA']
        
        for category in categories:
            folder_path = os.path.join(self.data_path, 'train', category)
            if not os.path.exists(folder_path):
                print(f"Warning: {folder_path} not found!")
                continue
                
            count = 0
            for filename in os.listdir(folder_path):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')) and count < max_images//2:
                    img_path = os.path.join(folder_path, filename)
                    
                    # Load and preprocess image
                    img = Image.open(img_path).convert('L')
                    img = img.resize((64, 64))
                    img_array = np.array(img) / 255.0
                    
                    self.images.append(img_array)
                    self.labels.append(category)
                    count += 1
                    
        print(f"Loaded {len(self.images)} images")
        return np.array(self.images), self.labels
    
    def add_noise(self, images, noise_type='gaussian', noise_level=0.15):
        """
        Add noise to images to test enhancement algorithms.
        
        Args:
            images: Array of images
            noise_type: Type of noise ('gaussian' or 'salt_pepper')
            noise_level: Noise intensity level
            
        Returns:
            Array of noisy images
        """
        noisy_images = []
        
        for img in images:
            if noise_type == 'gaussian':
                noise = np.random.normal(0, noise_level, img.shape)
                noisy_img = img + noise
            elif noise_type == 'salt_pepper':
                noisy_img = img.copy()
                salt = np.random.random(img.shape) < noise_level/2
                noisy_img[salt] = 1
                pepper = np.random.random(img.shape) < noise_level/2
                noisy_img[pepper] = 0
            else:
                noisy_img = img
                
            noisy_img = np.clip(noisy_img, 0, 1)
            noisy_images.append(noisy_img)
            
        return np.array(noisy_images)
    
    def classical_enhancement(self, noisy_images):
        """
        Classical image enhancement baseline using Gaussian and unsharp masking.
        
        Args:
            noisy_images: Array of noisy images
            
        Returns:
            Array of enhanced images
        """
        enhanced_images = []
        
        for img in noisy_images:
            enhanced = filters.gaussian(img, sigma=0.8)
            enhanced = filters.unsharp_mask(enhanced, radius=1, amount=1)
            enhanced_images.append(enhanced)
            
        return np.array(enhanced_images)
    
    def quantum_fourier_enhancement(self, img):
        """
        Quantum-inspired Fourier Transform for image enhancement.
        
        Args:
            img: Single image array
            
        Returns:
            Enhanced image
        """
        f_transform = np.fft.fft2(img)
        f_shift = np.fft.fftshift(f_transform)
        
        rows, cols = img.shape
        crow, ccol = rows//2, cols//2
        
        mask = np.zeros((rows, cols))
        
        for i in range(rows):
            for j in range(cols):
                d = np.sqrt((i - crow)**2 + (j - ccol)**2)
                quantum_factor = np.cos(d * np.pi / (rows/4)) * np.exp(-d/(rows/8))
                mask[i, j] = max(0, quantum_factor)
        
        f_shift_filtered = f_shift * (0.3 + 0.7 * mask)
        f_ishift = np.fft.ifftshift(f_shift_filtered)
        enhanced = np.fft.ifft2(f_ishift)
        enhanced = np.abs(enhanced)
        
        return enhanced
    
    def evaluate_enhancement(self, original, enhanced, noisy):
        """
        Evaluate enhancement quality using PSNR and SSIM metrics.
        
        Args:
            original: Original clean image
            enhanced: Enhanced image
            noisy: Noisy image
            
        Returns:
            Dictionary of evaluation metrics
        """
        original = np.clip(original, 0, 1)
        enhanced = np.clip(enhanced, 0, 1)
        noisy = np.clip(noisy, 0, 1)
        
        psnr_noisy = peak_signal_noise_ratio(original, noisy, data_range=1.0)
        psnr_enhanced = peak_signal_noise_ratio(original, enhanced, data_range=1.0)
        
        ssim_noisy = structural_similarity(original, noisy, data_range=1.0)
        ssim_enhanced = structural_similarity(original, enhanced, data_range=1.0)
        
        return {
            'psnr_improvement': psnr_enhanced - psnr_noisy,
            'ssim_improvement': ssim_enhanced - ssim_noisy,
            'psnr_enhanced': psnr_enhanced,
            'ssim_enhanced': ssim_enhanced
        }
