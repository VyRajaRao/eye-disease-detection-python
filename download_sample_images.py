#!/usr/bin/env python3
"""
Download Sample Eye Disease Images
This script downloads sample images from public sources for testing the model
"""

import os
import requests
import time
from pathlib import Path

def create_sample_images():
    """Create simple sample images for testing"""
    try:
        from PIL import Image, ImageDraw, ImageFont
        import numpy as np
        
        print("🖼️ Creating sample synthetic images for testing...")
        
        # Colors for different conditions
        colors = {
            'Normal': [(255, 200, 150), (255, 180, 120)],
            'Diabetic Retinopathy': [(255, 150, 150), (200, 100, 100)],
            'Glaucoma': [(200, 255, 200), (150, 200, 150)],
            'Cataract': [(230, 230, 230), (200, 200, 200)],
            'Age-related Macular Degeneration': [(255, 255, 200), (230, 230, 150)],
            'Hypertensive Retinopathy': [(255, 180, 180), (220, 150, 150)]
        }
        
        dataset_path = Path("dataset")
        
        for condition, color_palette in colors.items():
            condition_path = dataset_path / condition
            condition_path.mkdir(parents=True, exist_ok=True)
            
            # Create 15 sample images per condition
            for i in range(15):
                # Create a 512x512 image
                img = Image.new('RGB', (512, 512), color_palette[0])
                draw = ImageDraw.Draw(img)
                
                # Draw a circular retinal boundary
                center = (256, 256)
                radius = 200
                
                # Main retinal area
                draw.ellipse([center[0]-radius, center[1]-radius, 
                             center[0]+radius, center[1]+radius], 
                            fill=color_palette[1])
                
                # Add some realistic features
                # Optic disc
                disc_center = (200 + i*10, 256)
                draw.ellipse([disc_center[0]-30, disc_center[1]-30,
                             disc_center[0]+30, disc_center[1]+30],
                            fill=(255, 220, 180))
                
                # Blood vessels (simplified)
                for j in range(4):
                    start_x = center[0] + (j-2)*50
                    start_y = center[1] + (j-2)*30
                    end_x = start_x + 100
                    end_y = start_y + (j-2)*20
                    draw.line([start_x, start_y, end_x, end_y], 
                             fill=(180, 100, 100), width=3)
                
                # Add condition-specific features
                if condition == 'Diabetic Retinopathy':
                    # Add some dots (microaneurysms)
                    for k in range(8):
                        dot_x = 200 + k*15 + i*5
                        dot_y = 200 + k*12 + i*3
                        draw.ellipse([dot_x-2, dot_y-2, dot_x+2, dot_y+2],
                                   fill=(150, 0, 0))
                
                elif condition == 'Glaucoma':
                    # Enlarged optic cup
                    draw.ellipse([disc_center[0]-20, disc_center[1]-20,
                                 disc_center[0]+20, disc_center[1]+20],
                                fill=(240, 200, 160))
                
                elif condition == 'Cataract':
                    # Add cloudy overlay
                    overlay = Image.new('RGB', (512, 512), (255, 255, 255))
                    overlay_draw = ImageDraw.Draw(overlay)
                    overlay_draw.ellipse([center[0]-radius, center[1]-radius,
                                         center[0]+radius, center[1]+radius],
                                        fill=(200, 200, 200))
                    img = Image.blend(img, overlay, 0.3)
                
                # Save the image
                img_path = condition_path / f"sample_{i+1}.jpg"
                img.save(img_path, "JPEG", quality=85)
            
            print(f"✅ Created 15 sample images for {condition}")
        
        print("🎉 Sample images created successfully!")
        return True
        
    except ImportError as e:
        print(f"❌ PIL not available: {e}")
        return False
    except Exception as e:
        print(f"❌ Error creating sample images: {e}")
        return False

def download_real_samples():
    """Download some real sample images from public sources"""
    print("📥 Downloading real sample images...")
    
    # Sample retinal images from public sources
    sample_urls = {
        'Normal': [
            'https://raw.githubusercontent.com/btcsuite/btcutil/master/example_test.go'  # Placeholder - you'd need actual image URLs
        ],
        # Add more URLs for different conditions
    }
    
    print("⚠️  For real images, please:")
    print("   1. Visit kaggle.com/competitions/aptos2019-blindness-detection")
    print("   2. Download the APTOS 2019 dataset")
    print("   3. Or use other publicly available datasets")
    print("   4. Place images in the respective dataset folders")
    
    return True

def main():
    """Main function"""
    print("🖼️ " + "="*50)
    print("   EyeZen Detect - Sample Image Downloader")
    print("🖼️ " + "="*50 + "\n")
    
    print("Creating sample synthetic images for testing...")
    
    # Create synthetic sample images
    success = create_sample_images()
    
    if success:
        print("\n✅ Sample images ready!")
        print("\n📋 Next Steps:")
        print("1. For better training, add real eye disease images to dataset folders")
        print("2. Use public datasets like APTOS 2019, IDRiD, or Messidor")
        print("3. Ensure you have permission to use any real medical images")
        print("4. Run training: python backend/train_model.py --data-dir dataset")
    else:
        print("\n❌ Failed to create sample images")
        print("Please install Pillow: pip install Pillow")
    
    return success

if __name__ == "__main__":
    main()
