# EyeZen Detect - AI-Powered Eye Disease Detection

A comprehensive web application for detecting eye diseases using advanced deep learning techniques. This application combines a modern React frontend with a powerful Python backend that implements custom CNN models using TensorFlow/Keras, OpenCV, and scikit-learn.

## Features

### Frontend (React + TypeScript)
- 🎨 Modern, responsive UI with drag-and-drop image upload
- 📊 Real-time prediction results with confidence scores
- 📈 Interactive visualizations and analysis details
- 🌙 Dark/light theme support
- 📱 Mobile-friendly design
- 🔄 Loading animations and status indicators

### Backend (Python + Flask)
- 🧠 Custom CNN architecture for eye disease classification
- 🔬 OpenCV-based image preprocessing pipeline
- 📊 Matplotlib visualizations for training metrics
- 🔥 Grad-CAM heatmap generation for prediction explanations
- 🎯 Support for multiple eye disease classes:
  - Normal
  - Diabetic Retinopathy
  - Glaucoma
  - Cataract
  - Age-related Macular Degeneration
  - Hypertensive Retinopathy

### Machine Learning Pipeline
- 🏗️ Custom CNN model with 4 convolutional blocks
- 🔄 Data augmentation techniques (rotation, zoom, flips)
- 📈 Training visualization and metrics tracking
- 💾 Model persistence in .h5 format
- 🎯 Transfer learning support (EfficientNet, ResNet, VGG)
- 🔍 Image quality assessment and preprocessing

## Tech Stack

### Frontend
- React 18 with TypeScript
- Vite for fast development and building
- Tailwind CSS for styling
- Radix UI components
- React Router for navigation
- React Query for state management

### Backend
- Flask for web framework
- TensorFlow/Keras for deep learning
- OpenCV for image processing
- NumPy for numerical computations
- Matplotlib/Seaborn for visualizations
- scikit-learn for machine learning utilities
- Pillow for image handling

## Installation and Setup

### Prerequisites
- Node.js (v16 or higher)
- Python 3.8 or higher
- Git

### 1. Clone the Repository
```bash
git clone https://github.com/VyRajaRao/eyezen-detect.git
cd eyezen-detect
```

### 2. Frontend Setup
```bash
# Install frontend dependencies
npm install

# Start the development server
npm run dev
```
The frontend will be available at `http://localhost:5173`

### 3. Backend Setup
```bash
# Navigate to backend directory
cd backend

# Create a virtual environment (recommended)
python -m venv venv

# Activate virtual environment
# On Windows:
venv\\Scripts\\activate
# On macOS/Linux:
source venv/bin/activate

# Install Python dependencies
pip install -r requirements.txt

# Start the Flask server
python app.py
```
The backend API will be available at `http://localhost:5000`

## Model Training

### 1. Prepare Your Dataset
Create a dataset directory with the following structure:
```
dataset/
├── Normal/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── Diabetic Retinopathy/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── Glaucoma/
│   └── ...
├── Cataract/
│   └── ...
├── Age-related Macular Degeneration/
│   └── ...
└── Hypertensive Retinopathy/
    └── ...
```

### 2. Create Sample Dataset Structure
```bash
cd backend
python train_model.py --create-sample-dataset --data-dir ./dataset
```

### 3. Train the Model
```bash
# Basic training
python train_model.py --data-dir ./dataset --epochs 50 --batch-size 32

# With transfer learning
python train_model.py --data-dir ./dataset --epochs 30 --use-transfer-learning

# Custom parameters
python train_model.py --data-dir ./dataset --epochs 100 --batch-size 16 --learning-rate 0.0001
```

### Training Parameters
- `--data-dir`: Path to the dataset directory
- `--epochs`: Number of training epochs (default: 50)
- `--batch-size`: Training batch size (default: 32)
- `--learning-rate`: Learning rate (default: 0.001)
- `--use-transfer-learning`: Use pre-trained models
- `--model-name`: Custom model filename

## API Endpoints

### Health Check
```
GET /api/health
```
Returns the health status of the API and model loading status.

### Image Prediction
```
POST /api/predict
```
Uploads an image and returns prediction results.

**Parameters:**
- `image`: Image file (JPG, PNG, JPEG)
- `generate_heatmap`: Boolean to generate Grad-CAM heatmap

**Response:**
```json
{
  "disease": "Normal",
  "confidence": 0.95,
  "all_predictions": [
    {"class": "Normal", "confidence": 0.95},
    {"class": "Glaucoma", "confidence": 0.03}
  ],
  "processing_time": 2.3,
  "image_quality": "Excellent",
  "heatmap": "base64_encoded_heatmap_image"
}
```

### Model Information
```
GET /api/model/info
```
Returns information about the current model.

### Model Retraining
```
POST /api/train
```
Triggers model retraining with new data.

## Usage

### 1. Start Both Servers
Make sure both the frontend (port 5173) and backend (port 5000) servers are running.

### 2. Upload an Image
- Drag and drop a retinal fundus image onto the upload area, or
- Click "Select Image" to browse for an image file
- Supported formats: JPG, PNG, JPEG
- Maximum file size: 10MB

### 3. View Results
- The AI will analyze the image and provide:
  - Disease classification
  - Confidence score
  - Image quality assessment
  - Processing time
- Click "View Heatmap" to see Grad-CAM visualization
- Download a detailed PDF report

### 4. Model Performance
The current model achieves:
- Training Accuracy: ~85%
- Validation Accuracy: ~82%
- Processing Time: ~2-3 seconds per image

## Image Preprocessing

The application includes advanced image preprocessing:

1. **Quality Enhancement**:
   - CLAHE (Contrast Limited Adaptive Histogram Equalization)
   - Gaussian blur noise reduction
   - Unsharp masking for edge enhancement

2. **Circular Region Detection**:
   - Hough Circle Transform for retinal boundary detection
   - Automatic cropping to focus on relevant areas

3. **Data Augmentation**:
   - Rotation (±20 degrees)
   - Width/height shifts (±20%)
   - Horizontal flips
   - Zoom variations (±20%)
   - Brightness adjustments

## Model Architecture

### Custom CNN Model
```
Input (224x224x3)
├── Conv2D(32) + BatchNorm + Conv2D(32) + MaxPool + Dropout
├── Conv2D(64) + BatchNorm + Conv2D(64) + MaxPool + Dropout
├── Conv2D(128) + BatchNorm + Conv2D(128) + MaxPool + Dropout
├── Conv2D(256) + BatchNorm + Conv2D(256) + MaxPool + Dropout
├── GlobalAveragePooling2D
├── Dense(512) + BatchNorm + Dropout
├── Dense(256) + BatchNorm + Dropout
└── Dense(6, softmax)
```

### Transfer Learning Options
- EfficientNetB0 (recommended)
- ResNet50
- VGG16

## Visualizations

### Training Metrics
- Loss curves (training vs validation)
- Accuracy curves (training vs validation)
- Precision and recall metrics
- Confusion matrix

### Prediction Explanations
- Grad-CAM heatmaps showing areas of focus
- Class probability distributions
- Image quality metrics

## File Structure
```
eyezen-detect/
├── src/
│   ├── components/
│   ├── pages/
│   ├── types/
│   └── ...
├── backend/
│   ├── utils/
│   │   ├── image_preprocessor.py
│   │   ├── model_manager.py
│   │   └── visualization.py
│   ├── models/
│   ├── uploads/
│   ├── app.py
│   ├── config.py
│   ├── train_model.py
│   └── requirements.txt
├── package.json
└── README.md
```

## Troubleshooting

### Common Issues

1. **Model not loading**:
   - Ensure the model file exists in `backend/models/`
   - Run training script to create a model
   - Check file permissions

2. **CORS errors**:
   - Verify backend is running on port 5000
   - Check CORS configuration in `config.py`

3. **Image upload fails**:
   - Check file size (max 10MB)
   - Ensure image format is supported
   - Verify backend API is accessible

4. **Training fails**:
   - Check dataset structure matches required format
   - Ensure sufficient disk space
   - Verify all dependencies are installed

### Performance Optimization

1. **For better accuracy**:
   - Use more training data (>1000 images per class)
   - Implement transfer learning
   - Increase training epochs
   - Use data augmentation

2. **For faster inference**:
   - Use smaller input image size
   - Optimize model architecture
   - Use GPU acceleration

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License.

## Disclaimer

This application is for educational and research purposes only. It should not be used as a substitute for professional medical diagnosis. Always consult with qualified healthcare professionals for medical advice and treatment.

## Contact

For questions or support, please open an issue on GitHub or contact the development team.

---

**Note**: This application demonstrates advanced machine learning techniques for medical image analysis. The model should be trained on validated medical datasets and thoroughly tested before any clinical use.
