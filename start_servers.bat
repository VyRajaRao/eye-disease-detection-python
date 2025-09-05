@echo off
echo 🚀 Starting EyeZen Detect - AI-Powered Eye Disease Detection
echo ============================================================

echo.
echo 🔧 Starting Backend Server (Python/Flask)...
start "EyeZen Backend" cmd /k "cd /d C:\23R21A05CD\Projects\eyezen-detect\backend && venv\Scripts\python app.py"

echo ⏳ Waiting for backend to initialize...
timeout /t 5 /nobreak >nul

echo.
echo 🎨 Starting Frontend Server (React/Vite)...
start "EyeZen Frontend" cmd /k "cd /d C:\23R21A05CD\Projects\eyezen-detect && npm run dev"

echo.
echo ✅ Both servers are starting!
echo.
echo 🌐 Application URLs:
echo    Frontend: http://localhost:5173
echo    Backend:  http://localhost:5000
echo.
echo 📋 Instructions:
echo 1. Wait for both servers to fully start
echo 2. Open your browser and go to http://localhost:5173
echo 3. Upload a retinal fundus image to test the AI
echo 4. View the prediction results and heatmaps
echo.
echo Press any key to open the application in your browser...
pause >nul
start http://localhost:5173
