# Quick Start Guide - Eye Disease Detection System

## 🚀 Get Started in 3 Steps

### Step 1: Install Dependencies

The frontend and backend dependencies are already installed. If you need to reinstall:

```bash
# Frontend dependencies
cd frontend
npm install --legacy-peer-deps

# Backend dependencies  
cd ../backend
npm install
```

### Step 2: Start the Application

Run the startup script to launch both frontend and backend:

```bash
python start_app.py
```

Or start them manually:

**Terminal 1 - Backend:**
```bash
cd backend
npm run dev
```

**Terminal 2 - Frontend:**
```bash
cd frontend
npm start
```

### Step 3: Access the Application

- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:5000

## 🎯 Features Available

### ✅ Working Features
- **User Authentication**: Sign up and login
- **Modern UI**: Beautiful React interface with Tailwind CSS
- **Image Upload**: Drag and drop or file selection
- **Camera Capture**: Real-time camera access
- **Disease Detection**: Mock ML predictions with realistic results
- **Dashboard**: User statistics and activity tracking
- **Responsive Design**: Works on mobile and desktop

### 🔧 Technical Details
- **Frontend**: React 18 with modern hooks and context
- **Backend**: Express.js with JWT authentication
- **Image Processing**: Sharp library for image preprocessing
- **ML Model**: Mock predictions (can be upgraded to real ML later)

## 📱 How to Use

1. **Register/Login**: Create an account or sign in
2. **Upload Image**: Use camera or upload an eye image
3. **Get Analysis**: Receive disease detection results
4. **View History**: Check your previous scans

## 🛠️ Troubleshooting

### Common Issues

1. **Port Already in Use**:
   ```bash
   # Kill processes on ports 3000 and 5000
   netstat -ano | findstr :3000
   netstat -ano | findstr :5000
   ```

2. **Node Modules Missing**:
   ```bash
   cd frontend && npm install --legacy-peer-deps
   cd ../backend && npm install
   ```

3. **Python Issues**:
   - The app uses a mock ML model, so Python dependencies are optional
   - For real ML, install: `pip install opencv-python scikit-learn`

### Windows-Specific
- Use PowerShell or Command Prompt
- Run `python start_app.py` from the project root
- Ensure Node.js and Python are in PATH

## 🔄 Upgrading to Real ML Model

To use a real machine learning model:

1. Install Python dependencies:
   ```bash
   cd backend/ml_model
   python -m venv venv
   .\venv\Scripts\activate.ps1
   pip install opencv-python scikit-learn numpy pandas
   ```

2. Train the model:
   ```bash
   python simple_model.py
   ```

3. Update server.js to use `predict.py` instead of `mock_predict.py`

## 📞 Support

If you encounter issues:
1. Check the console for error messages
2. Ensure all dependencies are installed
3. Verify ports 3000 and 5000 are available
4. Check that Node.js and Python are properly installed

## 🎉 Ready to Use!

The application is now ready for demonstration and development. The mock ML model provides realistic predictions while the full system architecture is in place for real machine learning integration. 