#!/usr/bin/env python3
"""
Status Check Script for Eye Disease Detection System
Checks if both frontend and backend servers are running
"""

import requests
import time
import sys

def check_backend():
    """Check if backend server is running"""
    try:
        response = requests.get('http://localhost:5000/api/health', timeout=5)
        if response.status_code == 200:
            print("✅ Backend server is running on http://localhost:5000")
            return True
        else:
            print("❌ Backend server responded with status:", response.status_code)
            return False
    except requests.exceptions.RequestException as e:
        print("❌ Backend server is not running or not accessible")
        print("   Error:", str(e))
        return False

def check_frontend():
    """Check if frontend server is running"""
    try:
        response = requests.get('http://localhost:3000', timeout=5)
        if response.status_code == 200:
            print("✅ Frontend server is running on http://localhost:3000")
            return True
        else:
            print("❌ Frontend server responded with status:", response.status_code)
            return False
    except requests.exceptions.RequestException as e:
        print("❌ Frontend server is not running or not accessible")
        print("   Error:", str(e))
        return False

def main():
    """Main function to check both servers"""
    print("🔍 Checking Eye Disease Detection System Status...")
    print("=" * 50)
    
    backend_ok = check_backend()
    frontend_ok = check_frontend()
    
    print("=" * 50)
    
    if backend_ok and frontend_ok:
        print("🎉 Both servers are running successfully!")
        print("\n📱 You can now access the application:")
        print("   Frontend: http://localhost:3000")
        print("   Backend API: http://localhost:5000")
        print("\n🚀 To use the application:")
        print("   1. Open http://localhost:3000 in your browser")
        print("   2. Register a new account or login")
        print("   3. Go to Detection page to analyze eye images")
        print("   4. View results and recommendations")
        print("   5. Check your analysis history on the dashboard")
    else:
        print("⚠️  Some servers are not running properly.")
        print("\n🔧 Troubleshooting:")
        if not backend_ok:
            print("   - Backend: Make sure you're in the backend directory and run 'npm run dev'")
        if not frontend_ok:
            print("   - Frontend: Make sure you're in the frontend directory and run 'npm start'")
        print("\n💡 You can also use the startup script:")
        print("   python start_app.py")

if __name__ == "__main__":
    main() 