@echo off
echo 🚀 Setting up Eye Disease Detection System...
echo ==============================================

REM Check if Node.js is installed
node --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Node.js is not installed. Please install Node.js first.
    pause
    exit /b 1
)

REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Python is not installed. Please install Python first.
    pause
    exit /b 1
)

echo ✅ Node.js and Python are installed

REM Setup Frontend
echo.
echo 📱 Setting up Frontend...
cd frontend

if not exist "node_modules" (
    echo Installing frontend dependencies...
    npm install
) else (
    echo Frontend dependencies already installed
)

echo ✅ Frontend setup complete

REM Setup Backend
echo.
echo 🔧 Setting up Backend...
cd ..\backend

if not exist "node_modules" (
    echo Installing backend dependencies...
    npm install
) else (
    echo Backend dependencies already installed
)

REM Setup Python ML Model
echo.
echo 🤖 Setting up ML Model...
cd ml_model

if not exist "venv" (
    echo Creating Python virtual environment...
    python -m venv venv
)

echo Activating virtual environment...
call venv\Scripts\activate.bat

echo Installing Python dependencies...
pip install -r requirements.txt

echo ✅ ML Model setup complete

REM Create .env file if it doesn't exist
cd ..
if not exist ".env" (
    echo Creating .env file...
    (
        echo PORT=5000
        echo JWT_SECRET=your-secret-key-change-in-production
    ) > .env
    echo ✅ .env file created
) else (
    echo ✅ .env file already exists
)

echo.
echo 🎉 Setup completed successfully!
echo.
echo To start the application:
echo 1. Start the backend: cd backend ^&^& npm run dev
echo 2. Start the frontend: cd frontend ^&^& npm start
echo.
echo The application will be available at:
echo - Frontend: http://localhost:3000
echo - Backend API: http://localhost:5000
echo.
echo Happy coding! 👁️
pause 