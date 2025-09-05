"""
Visualization utilities for Eye Disease Detection
Handles training visualization and Grad-CAM heatmap generation using Matplotlib
"""

import io
import base64
import logging
from typing import Optional, Tuple
import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from PIL import Image

logger = logging.getLogger(__name__)

class VisualizationGenerator:
    """
    Generates visualizations for model training and predictions
    Includes training metrics plots and Grad-CAM heatmaps
    """
    
    def __init__(self):
        """Initialize the visualization generator"""
        # Set matplotlib backend for server environments
        plt.switch_backend('Agg')
        
        # Set style for better-looking plots
        plt.style.use('default')
        sns.set_palette("husl")
    
    def generate_gradcam_heatmap(self, model: keras.Model, 
                               processed_image: np.ndarray, 
                               original_image: np.ndarray,
                               layer_name: str = 'conv2d_4') -> Optional[str]:
        """
        Generate enhanced Grad-CAM heatmap for model prediction visualization
        
        Args:
            model: Trained Keras model
            processed_image: Preprocessed image used for prediction
            original_image: Original image for overlay
            layer_name: Name of the convolutional layer for grad-cam
            
        Returns:
            Base64 encoded heatmap image, or None if generation fails
        """
        
        # Try Grad-CAM first, fallback to simple heatmap if it fails
        try:
            return self._generate_true_gradcam(model, processed_image, original_image, layer_name)
        except Exception as e:
            logger.warning(f"Grad-CAM failed, using simplified heatmap: {str(e)}")
            return self._generate_simple_heatmap(processed_image, original_image)
    
    def _generate_simple_heatmap(self, processed_image: np.ndarray, original_image: np.ndarray) -> Optional[str]:
        """
        Generate a simple heatmap visualization as fallback
        
        Args:
            processed_image: Preprocessed image
            original_image: Original image
            
        Returns:
            Base64 encoded heatmap image
        """
        try:
            # Create a simple attention map based on image gradients
            gray = cv2.cvtColor(original_image, cv2.COLOR_RGB2GRAY)
            
            # Calculate gradients to simulate attention
            grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            
            # Combine gradients
            gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
            
            # Normalize to 0-1
            heatmap = (gradient_magnitude - gradient_magnitude.min()) / (gradient_magnitude.max() - gradient_magnitude.min())
            
            # Apply Gaussian blur for smoother heatmap
            heatmap = cv2.GaussianBlur(heatmap, (15, 15), 0)
            
            # Create enhanced visualization
            return self._create_heatmap_visualization(heatmap, original_image)
            
        except Exception as e:
            logger.error(f"Error generating simple heatmap: {str(e)}")
            return None
    
    def _generate_true_gradcam(self, model: keras.Model, 
                              processed_image: np.ndarray, 
                              original_image: np.ndarray,
                              layer_name: str) -> Optional[str]:
        """
        Generate true Grad-CAM heatmap
        
        Args:
            model: Trained Keras model
            processed_image: Preprocessed image used for prediction
            original_image: Original image for overlay
            layer_name: Name of the convolutional layer for grad-cam
            
        Returns:
            Base64 encoded heatmap image, or None if generation fails
        """
        try:
            # First, make a prediction to ensure the model is built
            _ = model(processed_image)
            
            # Find the target layer
            target_layer = None
            for layer in model.layers:
                if layer.name == layer_name:
                    target_layer = layer
                    break
            
            if target_layer is None:
                # Try to find the last convolutional layer
                for layer in reversed(model.layers):
                    if 'conv' in layer.name.lower():
                        target_layer = layer
                        logger.info(f"Using layer {layer.name} for Grad-CAM")
                        break
            
            if target_layer is None:
                logger.warning("No suitable convolutional layer found for Grad-CAM")
                return None
            
            # Create a model that maps the input image to the activations of the target layer
            grad_model = keras.models.Model(
                inputs=model.inputs,
                outputs=[target_layer.output, model.output]
            )
            
            # Compute the gradient of the top predicted class
            with tf.GradientTape() as tape:
                conv_outputs, predictions = grad_model(processed_image)
                predicted_class = tf.argmax(predictions[0])
                class_output = predictions[:, predicted_class]
            
            # Get the gradients of the output w.r.t. the last conv layer
            grads = tape.gradient(class_output, conv_outputs)
            
            # Pool the gradients over all the axes leaving out the channel dimension
            pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
            
            # Weight the channels by the corresponding gradients
            conv_outputs = conv_outputs[0]
            heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
            heatmap = tf.squeeze(heatmap)
            
            # Normalize the heatmap
            heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
            heatmap = heatmap.numpy()
            
            # Resize heatmap to original image size
            original_height, original_width = original_image.shape[:2]
            heatmap_resized = cv2.resize(heatmap, (original_width, original_height))
            
            # Create enhanced visualization
            return self._create_heatmap_visualization(heatmap_resized, original_image, target_layer.name)
            
        except Exception as e:
            logger.error(f"Error generating true Grad-CAM heatmap: {str(e)}")
            raise
    
    def _create_heatmap_visualization(self, heatmap_resized: np.ndarray, original_image: np.ndarray, layer_name: str = 'gradient-based') -> Optional[str]:
        """
        Create enhanced heatmap visualization
        
        Args:
            heatmap_resized: Resized heatmap array
            original_image: Original image
            layer_name: Name of the layer used for analysis
            
        Returns:
            Base64 encoded visualization
        """
        try:
            # Create enhanced heatmap visualization with better layout
            fig = plt.figure(figsize=(16, 10))
            
            # Set dark background for better visibility
            fig.patch.set_facecolor('black')
            
            # Create a grid layout
            gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.2)
            
            # Original image (top left)
            ax1 = fig.add_subplot(gs[0, 0])
            ax1.imshow(original_image)
            ax1.set_title('Original Retinal Image', color='white', fontsize=14, fontweight='bold')
            ax1.axis('off')
            
            # Pure heatmap (top middle)
            ax2 = fig.add_subplot(gs[0, 1])
            heatmap_display = ax2.imshow(heatmap_resized, cmap='jet', alpha=0.8)
            ax2.set_title('AI Attention Heatmap', color='white', fontsize=14, fontweight='bold')
            ax2.axis('off')
            
            # Add colorbar for heatmap
            cbar = plt.colorbar(heatmap_display, ax=ax2, fraction=0.046, pad=0.04)
            cbar.set_label('Attention Intensity', color='white', fontsize=10)
            cbar.ax.yaxis.set_tick_params(color='white')
            plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')
            
            # Overlay (top right)
            ax3 = fig.add_subplot(gs[0, 2])
            ax3.imshow(original_image)
            overlay = ax3.imshow(heatmap_resized, alpha=0.5, cmap='jet')
            ax3.set_title('AI Focus Overlay', color='white', fontsize=14, fontweight='bold')
            ax3.axis('off')
            
            # Enhanced overlay with contours (bottom left)
            ax4 = fig.add_subplot(gs[1, 0])
            ax4.imshow(original_image)
            # Add contour lines to show attention boundaries
            contours = ax4.contour(heatmap_resized, levels=5, colors='cyan', linewidths=1.5, alpha=0.7)
            ax4.imshow(heatmap_resized, alpha=0.3, cmap='hot')
            ax4.set_title('Attention Boundaries', color='white', fontsize=14, fontweight='bold')
            ax4.axis('off')
            
            # Intensity histogram (bottom middle)
            ax5 = fig.add_subplot(gs[1, 1])
            ax5.hist(heatmap_resized.flatten(), bins=50, color='cyan', alpha=0.7, edgecolor='white')
            ax5.set_title('Attention Distribution', color='white', fontsize=14, fontweight='bold')
            ax5.set_xlabel('Attention Intensity', color='white')
            ax5.set_ylabel('Frequency', color='white')
            ax5.tick_params(colors='white')
            ax5.set_facecolor('black')
            
            # Statistics and info (bottom right)
            ax6 = fig.add_subplot(gs[1, 2])
            ax6.axis('off')
            
            # Calculate heatmap statistics
            max_attention = np.max(heatmap_resized)
            mean_attention = np.mean(heatmap_resized)
            std_attention = np.std(heatmap_resized)
            
            # Create info text
            info_text = f"""
Heatmap Analysis:

Max Attention: {max_attention:.3f}
Mean Attention: {mean_attention:.3f}
Std Deviation: {std_attention:.3f}

High attention areas (>0.7):
{np.sum(heatmap_resized > 0.7)} pixels

Analysis Method: {layer_name}

Interpretation:
• Red/Yellow: High influence
• Blue/Purple: Low influence
• Contours: Attention boundaries
            """
            
            ax6.text(0.05, 0.95, info_text, transform=ax6.transAxes, 
                    fontsize=11, color='white', verticalalignment='top',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='black', alpha=0.8))
            
            # Set overall figure background
            for ax in [ax1, ax2, ax3, ax4, ax5, ax6]:
                ax.set_facecolor('black')
                
            plt.suptitle('AI Model Attention Analysis - Heatmap Visualization', 
                        color='white', fontsize=16, fontweight='bold', y=0.95)
            
            # Save to base64 string
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
            buffer.seek(0)
            
            # Encode to base64
            image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            
            plt.close()
            
            return image_base64
            
        except Exception as e:
            logger.error(f"Error creating heatmap visualization: {str(e)}")
            return None
    
    def plot_training_history(self, history: dict, save_path: str = None) -> Optional[str]:
        """
        Plot training history with metrics
        
        Args:
            history: Training history dictionary
            save_path: Path to save the plot
            
        Returns:
            Base64 encoded plot image, or saved file path
        """
        try:
            # Create subplots
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('Training History', fontsize=16, fontweight='bold')
            
            epochs = range(1, len(history['loss']) + 1)
            
            # Plot training & validation accuracy
            if 'accuracy' in history and 'val_accuracy' in history:
                axes[0, 0].plot(epochs, history['accuracy'], 'bo-', label='Training Accuracy', linewidth=2)
                axes[0, 0].plot(epochs, history['val_accuracy'], 'ro-', label='Validation Accuracy', linewidth=2)
                axes[0, 0].set_title('Model Accuracy', fontsize=14, fontweight='bold')
                axes[0, 0].set_xlabel('Epoch')
                axes[0, 0].set_ylabel('Accuracy')
                axes[0, 0].legend()
                axes[0, 0].grid(True, alpha=0.3)
            
            # Plot training & validation loss
            axes[0, 1].plot(epochs, history['loss'], 'bo-', label='Training Loss', linewidth=2)
            if 'val_loss' in history:
                axes[0, 1].plot(epochs, history['val_loss'], 'ro-', label='Validation Loss', linewidth=2)
            axes[0, 1].set_title('Model Loss', fontsize=14, fontweight='bold')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Loss')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
            
            # Plot precision if available
            if 'precision' in history and 'val_precision' in history:
                axes[1, 0].plot(epochs, history['precision'], 'go-', label='Training Precision', linewidth=2)
                axes[1, 0].plot(epochs, history['val_precision'], 'mo-', label='Validation Precision', linewidth=2)
                axes[1, 0].set_title('Model Precision', fontsize=14, fontweight='bold')
                axes[1, 0].set_xlabel('Epoch')
                axes[1, 0].set_ylabel('Precision')
                axes[1, 0].legend()
                axes[1, 0].grid(True, alpha=0.3)
            else:
                axes[1, 0].text(0.5, 0.5, 'Precision data not available', 
                              ha='center', va='center', transform=axes[1, 0].transAxes)
                axes[1, 0].set_title('Model Precision', fontsize=14, fontweight='bold')
            
            # Plot recall if available
            if 'recall' in history and 'val_recall' in history:
                axes[1, 1].plot(epochs, history['recall'], 'co-', label='Training Recall', linewidth=2)
                axes[1, 1].plot(epochs, history['val_recall'], 'yo-', label='Validation Recall', linewidth=2)
                axes[1, 1].set_title('Model Recall', fontsize=14, fontweight='bold')
                axes[1, 1].set_xlabel('Epoch')
                axes[1, 1].set_ylabel('Recall')
                axes[1, 1].legend()
                axes[1, 1].grid(True, alpha=0.3)
            else:
                axes[1, 1].text(0.5, 0.5, 'Recall data not available', 
                              ha='center', va='center', transform=axes[1, 1].transAxes)
                axes[1, 1].set_title('Model Recall', fontsize=14, fontweight='bold')
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.close()
                logger.info(f"Training history plot saved to {save_path}")
                return save_path
            else:
                # Return as base64 string
                buffer = io.BytesIO()
                plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
                buffer.seek(0)
                image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
                plt.close()
                return image_base64
                
        except Exception as e:
            logger.error(f"Error plotting training history: {str(e)}")
            return None
    
    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray, 
                            class_names: list, save_path: str = None) -> Optional[str]:
        """
        Plot confusion matrix
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            class_names: List of class names
            save_path: Path to save the plot
            
        Returns:
            Base64 encoded plot image, or saved file path
        """
        try:
            from sklearn.metrics import confusion_matrix
            
            # Compute confusion matrix
            cm = confusion_matrix(y_true, y_pred)
            
            # Create plot
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=class_names, yticklabels=class_names)
            plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
            plt.xlabel('Predicted Label', fontsize=12)
            plt.ylabel('True Label', fontsize=12)
            plt.xticks(rotation=45)
            plt.yticks(rotation=0)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.close()
                logger.info(f"Confusion matrix saved to {save_path}")
                return save_path
            else:
                # Return as base64 string
                buffer = io.BytesIO()
                plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
                buffer.seek(0)
                image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
                plt.close()
                return image_base64
                
        except Exception as e:
            logger.error(f"Error plotting confusion matrix: {str(e)}")
            return None
    
    def plot_class_distribution(self, class_counts: dict, save_path: str = None) -> Optional[str]:
        """
        Plot class distribution
        
        Args:
            class_counts: Dictionary with class names as keys and counts as values
            save_path: Path to save the plot
            
        Returns:
            Base64 encoded plot image, or saved file path
        """
        try:
            classes = list(class_counts.keys())
            counts = list(class_counts.values())
            
            # Create plot
            plt.figure(figsize=(12, 6))
            bars = plt.bar(classes, counts, color=plt.cm.Set3(range(len(classes))))
            plt.title('Class Distribution', fontsize=16, fontweight='bold')
            plt.xlabel('Classes', fontsize=12)
            plt.ylabel('Number of Samples', fontsize=12)
            plt.xticks(rotation=45)
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(height)}', ha='center', va='bottom')
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.close()
                logger.info(f"Class distribution plot saved to {save_path}")
                return save_path
            else:
                # Return as base64 string
                buffer = io.BytesIO()
                plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
                buffer.seek(0)
                image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
                plt.close()
                return image_base64
                
        except Exception as e:
            logger.error(f"Error plotting class distribution: {str(e)}")
            return None
    
    def create_prediction_visualization(self, image: np.ndarray, 
                                      predictions: list, 
                                      true_label: str = None) -> Optional[str]:
        """
        Create visualization showing image with prediction probabilities
        
        Args:
            image: Input image
            predictions: List of prediction dictionaries with 'class' and 'confidence'
            true_label: True label if available
            
        Returns:
            Base64 encoded visualization image
        """
        try:
            # Create subplot
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Show image
            ax1.imshow(image)
            ax1.set_title('Input Image', fontsize=14, fontweight='bold')
            ax1.axis('off')
            
            # Show predictions
            classes = [pred['class'] for pred in predictions[:5]]  # Top 5 predictions
            confidences = [pred['confidence'] for pred in predictions[:5]]
            
            # Create horizontal bar chart
            y_pos = range(len(classes))
            bars = ax2.barh(y_pos, confidences, color=plt.cm.viridis(np.linspace(0, 1, len(classes))))
            
            ax2.set_yticks(y_pos)
            ax2.set_yticklabels(classes)
            ax2.set_xlabel('Confidence', fontsize=12)
            ax2.set_title('Prediction Probabilities', fontsize=14, fontweight='bold')
            ax2.set_xlim(0, 1)
            
            # Add confidence values on bars
            for i, (bar, conf) in enumerate(zip(bars, confidences)):
                ax2.text(conf + 0.01, bar.get_y() + bar.get_height()/2,
                        f'{conf:.3f}', va='center', fontweight='bold')
            
            # Highlight true label if provided
            if true_label and true_label in classes:
                true_idx = classes.index(true_label)
                bars[true_idx].set_color('red')
                bars[true_idx].set_alpha(0.8)
            
            plt.tight_layout()
            
            # Return as base64 string
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            plt.close()
            
            return image_base64
            
        except Exception as e:
            logger.error(f"Error creating prediction visualization: {str(e)}")
            return None
