@echo off
echo Starting Cloudflare Tunnel for XAUUSD Dashboard...
echo.
echo Your dashboard will be accessible via a public URL
echo Keep this window open to maintain the tunnel
echo.
cloudflared.exe tunnel --url http://localhost:8501
