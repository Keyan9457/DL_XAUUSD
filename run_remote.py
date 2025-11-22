import os
import sys
from pyngrok import ngrok
import subprocess
import time

def run_remote_dashboard():
    # 1. Kill existing streamlit processes (optional, to avoid conflicts)
    # os.system("taskkill /f /im streamlit.exe") 

    # 2. Start Ngrok Tunnel
    # Note: If you have an auth token, set it here: ngrok.set_auth_token("YOUR_TOKEN")
    # Otherwise it runs in anonymous mode (session expires in 2 hours)
    
    try:
        public_url = ngrok.connect(8501).public_url
        print(f"\n\n{'='*60}")
        print(f"🌍 REMOTE DASHBOARD URL: {public_url}")
        print(f"{'='*60}\n\n")
    except Exception as e:
        print(f"Error starting Ngrok: {e}")
        return

    # 3. Run Streamlit
    cmd = [sys.executable, "-m", "streamlit", "run", "dashboard.py", "--server.headless", "true"]
    
    try:
        subprocess.check_call(cmd)
    except KeyboardInterrupt:
        print("\nStopping remote dashboard...")
        ngrok.kill()

if __name__ == "__main__":
    run_remote_dashboard()
