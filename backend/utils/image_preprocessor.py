"""
Image Preprocessing Pipeline for Eye Disease Detection
Uses OpenCV and NumPy for image preprocessing and augmentation
"""

import cv2
import numpy as np
from PIL import Image
import logging
from typing import Tuple, Union, Optional
from tensorflow.keras.preprocessing.image import ImageDataGenerator

logger = logging.getLogger(__name__)

class ImagePreprocessor:
    """
    Image preprocessing class for retinal fundus images
    Handles resizing, normalization, augmentation, and quality checks
    """
    
    def __init__(self, target_size: Tuple[int, int] = (224, 224)):
        """
        Initialize the image preprocessor
        
        Args:
            target_size: Target size for processed images (height, width)
        """
        self.target_size = target_size
        self.data_generator = None
        self._setup_augmentation()
    
    def _setup_augmentation(self):
        """Setup data augmentation generator for training"""
        self.data_generator = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            zoom_range=0.2,
            brightness_range=[0.8, 1.2],
            fill_mode='nearest',
            rescale=1.0/255.0  # Normalize to [0, 1]
        )
    
    def preprocess_for_prediction(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess a single image for prediction
        
        Args:
            image: Input image as numpy array (RGB format)
            
        Returns:
            Preprocessed image ready for model input
        """
        try:
            # Ensure image is in RGB format
            if len(image.shape) == 3 and image.shape[2] == 3:
                processed_image = image.copy()
            else:
                raise ValueError("Image must be in RGB format")
            
            # Resize image
            processed_image = cv2.resize(processed_image, self.target_size)
            
            # Enhance image quality
            processed_image = self._enhance_image_quality(processed_image)
            
            # Normalize pixel values to [0, 1]
            processed_image = processed_image.astype(np.float32) / 255.0
            
            # Add batch dimension
            processed_image = np.expand_dims(processed_image, axis=0)
            
            return processed_image
            
        except Exception as e:
            logger.error(f"Error in image preprocessing: {str(e)}")
            raise
    
    def preprocess_for_training(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for training (includes quality enhancement)
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Preprocessed image for training
        """
        try:
            # Resize image
            processed_image = cv2.resize(image, self.target_size)
            
            # Enhance image quality
            processed_image = self._enhance_image_quality(processed_image)
            
            # Convert to RGB if needed (OpenCV loads as BGR)
            if len(processed_image.shape) == 3:
                processed_image = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)
            
            return processed_image
            
        except Exception as e:
            logger.error(f"Error in training preprocessing: {str(e)}")
            raise
    
    def _enhance_image_quality(self, image: np.ndarray) -> np.ndarray:
        """
        Enhance image quality using various OpenCV techniques
        
        Args:
            image: Input image
            
        Returns:
            Enhanced image
        """
        try:
            # Convert to LAB color space for better enhancement
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            
            # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to L channel
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l_enhanced = clahe.apply(l)
            
            # Merge channels back
            enhanced_lab = cv2.merge([l_enhanced, a, b])
            enhanced_image = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)
            
            # Apply slight Gaussian blur to reduce noise
            enhanced_image = cv2.GaussianBlur(enhanced_image, (3, 3), 0)
            
            # Apply unsharp masking for better edge definition
            enhanced_image = self._unsharp_mask(enhanced_image, strength=0.5)
            
            return enhanced_image
            
        except Exception as e:
            logger.warning(f"Image enhancement failed, using original: {str(e)}")
            return image
    
    def _unsharp_mask(self, image: np.ndarray, strength: float = 0.5) -> np.ndarray:
        """
        Apply unsharp masking to enhance edges
        
        Args:
            image: Input image
            strength: Strength of sharpening (0.0 to 1.0)
            
        Returns:
            Sharpened image
        """
        try:
            # Create Gaussian blur
            blurred = cv2.GaussianBlur(image, (5, 5), 0)
            
            # Create unsharp mask
            unsharp_mask = cv2.addWeighted(image, 1 + strength, blurred, -strength, 0)
            
            return unsharp_mask
            
        except Exception as e:
            logger.warning(f"Unsharp masking failed: {str(e)}")
            return image
    
    def detect_circular_region(self, image: np.ndarray) -> Optional[Tuple[int, int, int]]:
        """
        Detect circular retinal region using Hough Circle Transform
        
        Args:
            image: Input retinal image
            
        Returns:
            Tuple of (center_x, center_y, radius) if circle is detected, None otherwise
        """
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
            # Apply Gaussian blur
            blurred = cv2.GaussianBlur(gray, (9, 9), 2)
            
            # Detect circles using HoughCircles
            circles = cv2.HoughCircles(
                blurred,
                cv2.HOUGH_GRADIENT,
                dp=1,
                minDist=int(gray.shape[0] / 2),
                param1=50,
                param2=30,
                minRadius=int(min(gray.shape) * 0.3),
                maxRadius=int(min(gray.shape) * 0.6)
            )
            
            if circles is not None:
                circles = np.round(circles[0, :]).astype("int")
                # Return the first (best) circle
                return (circles[0][0], circles[0][1], circles[0][2])
            
            return None
            
        except Exception as e:
            logger.warning(f"Circle detection failed: {str(e)}")
            return None
    
    def crop_to_circular_region(self, image: np.ndarray) -> np.ndarray:
        """
        Crop image to focus on the circular retinal region
        
        Args:
            image: Input retinal image
            
        Returns:
            Cropped image focused on retinal region
        """
        try:
            circle = self.detect_circular_region(image)
            
            if circle is not None:
                center_x, center_y, radius = circle
                
                # Create crop boundaries with some padding
                padding = int(radius * 0.1)
                x1 = max(0, center_x - radius - padding)
                y1 = max(0, center_y - radius - padding)
                x2 = min(image.shape[1], center_x + radius + padding)
                y2 = min(image.shape[0], center_y + radius + padding)
                
                # Crop the image
                cropped = image[y1:y2, x1:x2]
                
                return cropped
            
            # If no circle detected, return original image
            return image
            
        except Exception as e:
            logger.warning(f"Circular cropping failed: {str(e)}")
            return image
    
    def augment_images(self, images: np.ndarray, labels: np.ndarray, 
                      augmentation_factor: int = 2) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply data augmentation to training images
        
        Args:
            images: Array of training images
            labels: Corresponding labels
            augmentation_factor: How many augmented versions to create per image
            
        Returns:
            Tuple of (augmented_images, augmented_labels)
        """
        try:
            augmented_images = []
            augmented_labels = []
            
            for i, (image, label) in enumerate(zip(images, labels)):
                # Add original image
                augmented_images.append(image)
                augmented_labels.append(label)
                
                # Generate augmented versions
                image_batch = np.expand_dims(image, 0)
                
                count = 0
                for batch in self.data_generator.flow(image_batch, batch_size=1):
                    if count >= augmentation_factor:
                        break
                    
                    aug_image = batch[0]
                    augmented_images.append(aug_image)
                    augmented_labels.append(label)
                    count += 1
            
            return np.array(augmented_images), np.array(augmented_labels)
            
        except Exception as e:
            logger.error(f"Image augmentation failed: {str(e)}")
            return images, labels
    
    def check_image_quality(self, image: np.ndarray) -> dict:
        """
        Assess various image quality metrics
        
        Args:
            image: Input image
            
        Returns:
            Dictionary containing quality metrics
        """
        try:
            quality_metrics = {}
            
            # Convert to grayscale for analysis
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image
            
            # Check image dimensions
            height, width = gray.shape[:2]
            quality_metrics['resolution'] = (width, height)
            quality_metrics['total_pixels'] = width * height
            
            # Calculate blur metric (Laplacian variance)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            quality_metrics['blur_score'] = float(laplacian_var)
            quality_metrics['is_blurry'] = laplacian_var < 100
            
            # Calculate brightness metrics
            mean_brightness = np.mean(gray)
            quality_metrics['mean_brightness'] = float(mean_brightness)
            quality_metrics['is_too_dark'] = mean_brightness < 50
            quality_metrics['is_too_bright'] = mean_brightness > 200
            
            # Calculate contrast (standard deviation of pixel intensities)
            contrast = np.std(gray)
            quality_metrics['contrast'] = float(contrast)
            quality_metrics['is_low_contrast'] = contrast < 30
            
            # Overall quality assessment
            quality_issues = [
                quality_metrics['is_blurry'],
                quality_metrics['is_too_dark'],
                quality_metrics['is_too_bright'],
                quality_metrics['is_low_contrast']
            ]
            
            if not any(quality_issues):
                quality_metrics['overall_quality'] = 'Excellent'
            elif sum(quality_issues) == 1:
                quality_metrics['overall_quality'] = 'Good'
            elif sum(quality_issues) == 2:
                quality_metrics['overall_quality'] = 'Fair'
            else:
                quality_metrics['overall_quality'] = 'Poor'
            
            return quality_metrics
            
        except Exception as e:
            logger.error(f"Quality assessment failed: {str(e)}")
            return {'overall_quality': 'Unknown', 'error': str(e)}
    
    def batch_preprocess(self, image_paths: list, 
                        for_training: bool = True) -> np.ndarray:
        """
        Preprocess a batch of images
        
        Args:
            image_paths: List of image file paths
            for_training: Whether preprocessing is for training or prediction
            
        Returns:
            Array of preprocessed images
        """
        preprocessed_images = []
        
        for path in image_paths:
            try:
                # Load image
                image = cv2.imread(path)
                if image is None:
                    logger.warning(f"Could not load image: {path}")
                    continue
                
                # Preprocess based on use case
                if for_training:
                    processed = self.preprocess_for_training(image)
                else:
                    # Convert BGR to RGB for prediction preprocessing
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    processed = self.preprocess_for_prediction(image_rgb)
                    processed = processed[0]  # Remove batch dimension
                
                preprocessed_images.append(processed)
                
            except Exception as e:
                logger.error(f"Error processing image {path}: {str(e)}")
                continue
        
        return np.array(preprocessed_images)
