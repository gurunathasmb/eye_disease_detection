# Eye Disease Detection System

A comprehensive web application for detecting eye diseases using machine learning and computer vision. The system includes a React frontend with authentication, image capture/upload capabilities, and a Python-based ML model for disease classification.

## Features

### Frontend (React)
- **User Authentication**: Sign up and login functionality
- **Modern UI**: Beautiful, responsive design with Tailwind CSS
- **Image Capture**: Real-time camera capture for eye images
- **File Upload**: Drag-and-drop image upload functionality
- **Disease Detection**: Real-time analysis with detailed results
- **Dashboard**: User statistics and activity tracking
- **Responsive Design**: Works on desktop and mobile devices

### Backend (Node.js + Python)
- **RESTful API**: Express.js server with JWT authentication
- **Image Processing**: Sharp library for image preprocessing
- **ML Integration**: Python-based TensorFlow model
- **File Upload**: Multer for handling image uploads
- **Security**: Password hashing and JWT tokens

### Machine Learning Model
- **CNN Architecture**: MobileNetV2-based transfer learning
- **Multi-class Classification**: Detects 4 types of eye conditions
- **Data Augmentation**: Improves model robustness
- **Transfer Learning**: Pre-trained ImageNet weights
- **Real-time Prediction**: Fast inference for web application
- **Kaggle Dataset Integration**: Uses real eye disease images

## Supported Eye Diseases

1. **Cataract**: Clouding of the eye's natural lens
2. **Diabetic Retinopathy**: Diabetes-related retinal damage
3. **Glaucoma**: Optic nerve damage
4. **Normal**: Healthy eye condition

## Project Structure

```
eye_disease_detection/
├── frontend/                 # React application
│   ├── src/
│   │   ├── components/       # React components
│   │   │   ├── auth/         # Authentication components
│   │   │   ├── dashboard/    # Dashboard component
│   │   │   └── detection/    # Disease detection component
│   │   ├── context/          # React context for auth
│   │   └── App.js           # Main app component
│   ├── package.json
│   └── tailwind.config.js
├── backend/                  # Node.js server
│   ├── server.js            # Express server
│   ├── package.json
│   └── ml_model/            # Python ML model
│       ├── train_model.py   # Model training script
│       ├── predict.py       # Prediction script
│       ├── download_dataset.py # Dataset download script
│       ├── requirements.txt # Python dependencies
│       └── dataset/         # Eye disease images
├── setup_with_dataset.py    # Complete setup script
└── README.md
```

## Prerequisites

- Node.js (v16 or higher)
- Python (v3.8 or higher)
- npm or yarn
- Git

## Quick Setup (Recommended)

### Option 1: Automated Setup with Dataset

Run the complete setup script that will install dependencies and download the Kaggle dataset:

```bash
# Windows
python setup_with_dataset.py

# Linux/Mac
python3 setup_with_dataset.py
```

This script will:
- ✅ Check prerequisites (Node.js, Python, npm)
- ✅ Install frontend dependencies
- ✅ Install backend dependencies
- ✅ Set up Python virtual environment
- ✅ Install Python ML dependencies
- ✅ Download Kaggle eye disease dataset
- ✅ Create environment configuration

### Option 2: Manual Setup

If you prefer manual setup or the automated script fails:

#### 1. Frontend Setup

```bash
cd frontend
npm install
```

#### 2. Backend Setup

```bash
cd backend
npm install
```

#### 3. ML Model Setup

```bash
cd backend/ml_model

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Download dataset
python download_dataset.py
```

## Starting the Application

### 1. Start Backend Server

```bash
cd backend
npm run dev
```

The backend API will be available at `http://localhost:5000`

### 2. Start Frontend (in a new terminal)

```bash
cd frontend
npm start
```

The frontend will be available at `http://localhost:3000`

## Usage

### 1. User Registration/Login
- Navigate to the application
- Create a new account or login with existing credentials
- JWT tokens are automatically handled for authentication

### 2. Disease Detection
- **Camera Capture**: Use the camera tab to capture real-time eye images
- **File Upload**: Drag and drop or select eye images from your device
- **Analysis**: Click "Analyze" to process the image
- **Results**: View detailed disease information, symptoms, and treatment recommendations

### 3. Dashboard
- View your detection history
- Track statistics and activity
- Access quick actions for new scans

## Machine Learning Model

### Dataset

The system uses the **Eye Diseases Classification** dataset from Kaggle:
- **Source**: [gunavenkatdoddi/eye-diseases-classification](https://www.kaggle.com/datasets/gunavenkatdoddi/eye-diseases-classification)
- **Classes**: 4 eye disease categories
- **Images**: Real medical eye images
- **Automatic Download**: Handled by `download_dataset.py`

### Training the Model

1. **Automatic Setup**: The setup script automatically downloads and prepares the dataset
2. **Manual Training**:
   ```bash
   cd backend/ml_model
   # Activate virtual environment
   python train_model.py
   ```

3. **Model Outputs**:
   - `eye_disease_model.h5`: Trained model file
   - `class_names.json`: Class labels
   - `confusion_matrix.png`: Model performance visualization
   - `training_history.png`: Training progress plots

### Model Architecture

- **Base Model**: MobileNetV2 (pre-trained on ImageNet)
- **Input Size**: 224x224x3 RGB images
- **Output**: 4-class softmax probabilities
- **Optimizer**: Adam with learning rate scheduling
- **Loss Function**: Categorical crossentropy

## API Endpoints

### Authentication
- `POST /api/auth/signup` - User registration
- `POST /api/auth/login` - User login

### Disease Detection
- `POST /api/detect` - Upload and analyze eye image
- `GET /api/history` - Get user detection history

### Health Check
- `GET /api/health` - API status check

## Environment Variables

The setup script automatically creates a `.env` file in the backend directory:

```env
PORT=5000
JWT_SECRET=your-secret-key-change-in-production
```

## Troubleshooting

### Common Issues

1. **Setup Script Fails**:
   - Ensure Node.js and Python are installed and in PATH
   - Try running the manual setup steps

2. **Dataset Download Fails**:
   - Check internet connection
   - The system will fall back to synthetic data
   - You can manually add images to `backend/ml_model/dataset/`

3. **Port Already in Use**:
   - Change the PORT in backend/.env file
   - Kill processes using ports 3000 or 5000

4. **Python Dependencies Issues**:
   - Ensure you're using the virtual environment
   - Try: `pip install --upgrade pip`

### Windows-Specific Issues

- Use `python setup_with_dataset.py` instead of shell scripts
- Ensure Python and Node.js are added to PATH during installation
- Use PowerShell or Command Prompt (not Git Bash for some commands)

## Development

### Frontend Development
```bash
cd frontend
npm start
```

### Backend Development
```bash
cd backend
npm run dev
```

### Model Training
```bash
cd backend/ml_model
# Activate virtual environment first
python train_model.py
```

## Production Deployment

### Frontend
```bash
cd frontend
npm run build
```

### Backend
```bash
cd backend
npm start
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License.

## Disclaimer

This application is for educational and demonstration purposes. It should not be used as a substitute for professional medical diagnosis. Always consult with qualified healthcare professionals for medical advice and diagnosis.

## Support

For support and questions, please open an issue in the repository or contact the development team.

## Acknowledgments

- TensorFlow and Keras for machine learning capabilities
- React and Node.js communities for excellent documentation
- Kaggle for providing the eye disease dataset
- Medical imaging datasets and research papers
- Open-source computer vision libraries 