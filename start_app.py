#!/usr/bin/env python3
"""
Simple startup script for Eye Disease Detection System
This will start the frontend and backend without requiring ML model setup
"""

import subprocess
import time
import sys
import os
from pathlib import Path

def run_command(command, cwd=None, shell=True):
    """Run a command in a new process"""
    try:
        process = subprocess.Popen(command, cwd=cwd, shell=shell)
        return process
    except Exception as e:
        print(f"❌ Error running command {command}: {e}")
        return None

def main():
    """Main startup function"""
    print("🚀 Starting Eye Disease Detection System...")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not Path("frontend").exists() or not Path("backend").exists():
        print("❌ Please run this script from the eye_disease_detection directory")
        return
    
    # Start backend
    print("🔧 Starting Backend Server...")
    backend_process = run_command("npm run dev", cwd="backend")
    
    if backend_process is None:
        print("❌ Failed to start backend")
        return
    
    # Wait a moment for backend to start
    print("⏳ Waiting for backend to start...")
    time.sleep(3)
    
    # Start frontend
    print("📱 Starting Frontend...")
    frontend_process = run_command("npm start", cwd="frontend")
    
    if frontend_process is None:
        print("❌ Failed to start frontend")
        backend_process.terminate()
        return
    
    print("\n🎉 Application started successfully!")
    print("=" * 50)
    print("The application is now running:")
    print("- Frontend: http://localhost:3000")
    print("- Backend API: http://localhost:5000")
    print("\nPress Ctrl+C to stop both servers")
    
    try:
        # Keep the script running
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 Stopping servers...")
        if backend_process:
            backend_process.terminate()
        if frontend_process:
            frontend_process.terminate()
        print("✅ Servers stopped")

if __name__ == "__main__":
    main() 