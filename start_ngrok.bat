@echo off
echo ========================================
echo   XAUUSD Dashboard - Ngrok Tunnel
echo ========================================
echo.
echo Starting tunnel to http://localhost:8501
echo.
echo Your dashboard will be accessible via a public URL
echo Keep this window open to maintain the tunnel
echo.
echo ========================================
echo.
ngrok.exe http 8501
