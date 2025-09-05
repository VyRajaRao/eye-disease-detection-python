#!/usr/bin/env python3
"""
Setup and Quick Start Script for EyeZen Detect
Helps users set up the environment and test the application
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

def print_header():
    """Print application header"""
    print("🌟 " + "=" * 60 + " 🌟")
    print("   EyeZen Detect - AI-Powered Eye Disease Detection")
    print("   Setup and Quick Start Script")
    print("🌟 " + "=" * 60 + " 🌟\n")

def check_python_version():
    """Check if Python version is adequate"""
    version = sys.version_info
    print(f"🐍 Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8 or higher is required")
        return False
    
    print("✅ Python version is adequate")
    return True

def check_node():
    """Check if Node.js is installed"""
    try:
        result = subprocess.run(['node', '--version'], 
                              capture_output=True, text=True, check=True)
        version = result.stdout.strip()
        print(f"📦 Node.js version: {version}")
        print("✅ Node.js is available")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ Node.js not found. Please install Node.js v16 or higher")
        return False

def create_virtual_environment():
    """Create Python virtual environment"""
    venv_path = Path("backend/venv")
    
    if venv_path.exists():
        print("✅ Virtual environment already exists")
        return True
    
    try:
        print("📦 Creating virtual environment...")
        subprocess.run([sys.executable, '-m', 'venv', str(venv_path)], 
                      check=True)
        print("✅ Virtual environment created")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to create virtual environment: {e}")
        return False

def get_activation_command():
    """Get virtual environment activation command"""
    if platform.system() == "Windows":
        return "backend\\venv\\Scripts\\activate"
    else:
        return "source backend/venv/bin/activate"

def install_python_deps():
    """Install Python dependencies"""
    try:
        print("📦 Installing Python dependencies...")
        
        # Determine pip path
        if platform.system() == "Windows":
            pip_path = "backend/venv/Scripts/pip"
        else:
            pip_path = "backend/venv/bin/pip"
        
        # Install requirements
        subprocess.run([pip_path, 'install', '-r', 'backend/requirements.txt'], 
                      check=True)
        print("✅ Python dependencies installed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install Python dependencies: {e}")
        return False

def install_node_deps():
    """Install Node.js dependencies"""
    try:
        print("📦 Installing Node.js dependencies...")
        subprocess.run(['npm', 'install'], check=True)
        print("✅ Node.js dependencies installed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install Node.js dependencies: {e}")
        return False

def create_basic_model():
    """Create a basic model for testing"""
    try:
        print("🧠 Creating basic model for testing...")
        
        # Use the test script to create a model
        if platform.system() == "Windows":
            python_path = "backend/venv/Scripts/python"
        else:
            python_path = "backend/venv/bin/python"
        
        subprocess.run([python_path, 'backend/test_backend.py'], 
                      cwd='.', check=True)
        print("✅ Basic model created")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to create model: {e}")
        return False

def show_next_steps():
    """Show next steps to the user"""
    activation_cmd = get_activation_command()
    
    print("\n🚀 Setup Complete! Next Steps:")
    print("=" * 50)
    
    print("\n1️⃣ Start the Backend Server:")
    print(f"   {activation_cmd}")
    print("   cd backend")
    print("   python app.py")
    
    print("\n2️⃣ Start the Frontend Server (in a new terminal):")
    print("   npm run dev")
    
    print("\n3️⃣ Open Your Browser:")
    print("   Frontend: http://localhost:5173")
    print("   Backend API: http://localhost:5000")
    
    print("\n4️⃣ Test the Application:")
    print("   - Upload a retinal fundus image")
    print("   - View AI predictions")
    print("   - Check Grad-CAM heatmaps")
    
    print("\n📚 For more information:")
    print("   - Read README.md for detailed instructions")
    print("   - Check backend/requirements.txt for dependencies")
    print("   - Visit the GitHub repository for updates")

def create_sample_images_info():
    """Create information about sample images"""
    info_text = """
# Sample Images for Testing

To test the EyeZen Detect application, you'll need retinal fundus images.

## Where to get sample images:
1. **Public datasets:**
   - APTOS 2019 Blindness Detection (Kaggle)
   - IDRiD Dataset (IEEE DataPort)
   - Messidor Dataset
   - EyePACS Dataset

2. **For development/testing:**
   - Use synthetic or publicly available images
   - Ensure images are in RGB format
   - Recommended size: 224x224 pixels or larger

## Image requirements:
- Format: JPG, PNG, JPEG
- Size: Maximum 10MB
- Content: Retinal fundus photographs
- Quality: Clear, well-lit images work best

## Disclaimer:
- Only use images you have permission to use
- This tool is for educational purposes only
- Always consult healthcare professionals for medical advice
"""
    
    with open("SAMPLE_IMAGES.md", "w") as f:
        f.write(info_text)
    
    print("📄 Created SAMPLE_IMAGES.md with testing information")

def main():
    """Main setup function"""
    print_header()
    
    # Check requirements
    if not check_python_version():
        return False
    
    if not check_node():
        return False
    
    print("\n🔧 Setting up the environment...")
    
    # Setup steps
    steps = [
        ("Create Virtual Environment", create_virtual_environment),
        ("Install Python Dependencies", install_python_deps),
        ("Install Node.js Dependencies", install_node_deps),
        ("Create Basic Model", create_basic_model),
        ("Create Sample Images Info", create_sample_images_info),
    ]
    
    for step_name, step_func in steps:
        print(f"\n🔄 {step_name}...")
        if not step_func():
            print(f"❌ Setup failed at: {step_name}")
            return False
    
    show_next_steps()
    return True

if __name__ == "__main__":
    success = main()
    
    if success:
        print("\n🎉 Setup completed successfully!")
    else:
        print("\n💥 Setup failed. Please check the errors above.")
    
    sys.exit(0 if success else 1)
