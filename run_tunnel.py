import subprocess
import time
import sys

def run_ssh_tunnel():
    print("Starting SSH Tunnel to localhost.run...")
    # SSH command to forward port 8501 to localhost.run
    # -R 80:localhost:8501 : Forward remote port 80 to local 8501
    # nokey@localhost.run : The user and host
    # -o StrictHostKeyChecking=no : Avoid manual confirmation prompt
    cmd = ["ssh", "-R", "80:localhost:8501", "nokey@localhost.run", "-o", "StrictHostKeyChecking=no"]
    
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    
    print("Tunnel started. Waiting for URL...")
    
    # Read output line by line to find the URL
    while True:
        line = process.stdout.readline()
        if not line:
            break
        print(line.strip())
        if "tunneled with tls" in line.lower() or "localhost.run" in line.lower():
            print(f"\nFOUND URL IN OUTPUT: {line.strip()}")

if __name__ == "__main__":
    run_ssh_tunnel()
