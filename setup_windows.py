#!/usr/bin/env python3
"""
Windows Setup Script for Eye Disease Detection System
This script will set up the complete system including ML model and dataset
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

def run_command(command, cwd=None, shell=True, check_output=False):
    """Run a command and return the result"""
    try:
        if check_output:
            result = subprocess.run(command, cwd=cwd, shell=shell, capture_output=True, text=True, check=True)
            return result.stdout.strip()
        else:
            result = subprocess.run(command, cwd=cwd, shell=shell, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"❌ Command failed: {command}")
                print(f"Error: {result.stderr}")
                return False
            return True
    except Exception as e:
        print(f"❌ Error running command {command}: {e}")
        return False

def check_prerequisites():
    """Check if required software is installed"""
    print("🔍 Checking prerequisites...")
    
    # Check Node.js
    if not run_command("node --version", shell=True):
        print("❌ Node.js is not installed. Please install Node.js first.")
        print("Download from: https://nodejs.org/")
        return False
    
    # Check Python
    if not run_command("python --version", shell=True):
        print("❌ Python is not installed. Please install Python first.")
        print("Download from: https://python.org/")
        return False
    
    # Check npm
    if not run_command("npm --version", shell=True):
        print("❌ npm is not installed. Please install npm first.")
        return False
    
    print("✅ All prerequisites are installed")
    return True

def setup_frontend():
    """Set up the React frontend"""
    print("\n📱 Setting up Frontend...")
    
    frontend_dir = Path("frontend")
    if not frontend_dir.exists():
        print("❌ Frontend directory not found")
        return False
    
    # Install dependencies
    print("Installing frontend dependencies...")
    if not run_command("npm install --legacy-peer-deps", cwd=frontend_dir):
        print("❌ Failed to install frontend dependencies")
        return False
    
    print("✅ Frontend setup complete")
    return True

def setup_backend():
    """Set up the Node.js backend"""
    print("\n🔧 Setting up Backend...")
    
    backend_dir = Path("backend")
    if not backend_dir.exists():
        print("❌ Backend directory not found")
        return False
    
    # Install Node.js dependencies
    print("Installing backend dependencies...")
    if not run_command("npm install", cwd=backend_dir):
        print("❌ Failed to install backend dependencies")
        return False
    
    print("✅ Backend setup complete")
    return True

def setup_ml_model():
    """Set up the Python ML model and download dataset"""
    print("\n🤖 Setting up ML Model...")
    
    ml_dir = Path("backend/ml_model")
    if not ml_dir.exists():
        print("❌ ML model directory not found")
        return False
    
    # Create virtual environment
    venv_dir = ml_dir / "venv"
    if not venv_dir.exists():
        print("Creating Python virtual environment...")
        if not run_command("python -m venv venv", cwd=ml_dir):
            print("❌ Failed to create virtual environment")
            return False
    
    # Get Python and pip paths
    if platform.system() == "Windows":
        python_path = venv_dir / "Scripts" / "python.exe"
        pip_path = venv_dir / "Scripts" / "pip.exe"
    else:
        python_path = venv_dir / "bin" / "python"
        pip_path = venv_dir / "bin" / "pip"
    
    print("Installing Python dependencies...")
    if not run_command(f'"{pip_path}" install -r requirements.txt', cwd=ml_dir):
        print("❌ Failed to install Python dependencies")
        return False
    
    # Download dataset
    print("\n📥 Downloading Kaggle Dataset...")
    if not run_command(f'"{python_path}" download_dataset.py', cwd=ml_dir):
        print("❌ Failed to download dataset")
        print("Continuing with synthetic data...")
    
    print("✅ ML Model setup complete")
    return True

def create_env_file():
    """Create .env file for backend"""
    print("\n⚙️ Creating environment file...")
    
    backend_dir = Path("backend")
    env_file = backend_dir / ".env"
    
    if not env_file.exists():
        env_content = """PORT=5000
JWT_SECRET=your-secret-key-change-in-production
"""
        with open(env_file, 'w') as f:
            f.write(env_content)
        print("✅ .env file created")
    else:
        print("✅ .env file already exists")
    
    return True

def main():
    """Main setup function"""
    print("🚀 Setting up Eye Disease Detection System...")
    print("=" * 50)
    
    # Check prerequisites
    if not check_prerequisites():
        return
    
    # Setup frontend
    if not setup_frontend():
        return
    
    # Setup backend
    if not setup_backend():
        return
    
    # Setup ML model and download dataset
    if not setup_ml_model():
        return
    
    # Create environment file
    if not create_env_file():
        return
    
    print("\n🎉 Setup completed successfully!")
    print("=" * 50)
    print("\nTo start the application:")
    print("1. Start the backend:")
    print("   cd backend")
    print("   npm run dev")
    print("\n2. Start the frontend (in a new terminal):")
    print("   cd frontend")
    print("   npm start")
    print("\nThe application will be available at:")
    print("- Frontend: http://localhost:3000")
    print("- Backend API: http://localhost:5000")
    print("\nHappy coding! 👁️")

if __name__ == "__main__":
    main() 