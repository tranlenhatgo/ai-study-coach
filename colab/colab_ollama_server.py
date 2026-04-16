# ============================================================================
# AI Study Coach — Self-Hosted DeepSeek-R1 on Google Colab
# ============================================================================
#
# This script runs DeepSeek-R1-Distill-Llama-8B via Ollama on Google Colab's
# GPU and exposes it via ngrok so your local FastAPI server can connect to it.
#
# HOW TO USE:
#   1. Open Google Colab: https://colab.research.google.com/
#   2. Create a new notebook
#   3. Set runtime to GPU: Runtime → Change runtime type → T4 GPU
#   4. Copy each "CELL" section below into a separate Colab cell
#   5. Run cells in order
#   6. Copy the ngrok URL and paste it into your .env file
#
# MODEL: deepseek-ai/DeepSeek-R1-Distill-Llama-8B (via Ollama as deepseek-r1:8b)
#
# ============================================================================


# %%  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CELL 1: Check GPU & Install Ollama
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

import subprocess
import os

# Verify GPU is available
gpu_check = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
print("=" * 60)
print("🖥️  GPU Information")
print("=" * 60)
print(gpu_check.stdout)

if "NVIDIA" not in gpu_check.stdout:
    print("❌ No GPU detected! Go to Runtime → Change runtime type → T4 GPU")
    raise SystemExit("GPU required")

# Install Ollama (direct binary download — the install.sh script
# fails on Colab because it tries to set up systemd services)
print("\n📦 Installing Ollama...")
try:
    # Determine architecture
    arch = subprocess.run(["uname", "-m"], capture_output=True, text=True, check=True).stdout.strip()
    ollama_arch = ""
    if "x86_64" in arch:
        ollama_arch = "amd64"
    elif "aarch64" in arch:
        ollama_arch = "arm64"
    else:
        raise ValueError(f"Unsupported architecture: {arch}")

    ollama_url = f"https://ollama.com/download/ollama-linux-{ollama_arch}"
    install_path = "/usr/local/bin/ollama"

    print(f"   Downloading Ollama binary for {ollama_arch}...")
    subprocess.run(
        ["curl", "-L", ollama_url, "-o", install_path],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(["chmod", "+x", install_path], check=True)
    print("✅ Ollama installed successfully!")
except subprocess.CalledProcessError as e:
    print(f"❌ Error installing Ollama: {e}")
    if e.stdout:
        print(f"   stdout: {e.stdout}")
    if e.stderr:
        print(f"   stderr: {e.stderr}")
    raise SystemExit("Ollama installation failed.")
except Exception as e:
    print(f"❌ Unexpected error: {e}")
    raise SystemExit("Ollama installation failed.")


# %%  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CELL 2: Start Ollama Server
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

import subprocess
import time
import os

# Start Ollama server in the background
print("🚀 Starting Ollama server...")
ollama_process = subprocess.Popen(
    ["ollama", "serve"],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    env={**os.environ, "OLLAMA_HOST": "0.0.0.0:11434"},
)

# Wait for server to be ready
time.sleep(5)

# Verify it's running
import urllib.request
try:
    response = urllib.request.urlopen("http://localhost:11434")
    print("✅ Ollama server is running on port 11434")
except Exception as e:
    print(f"⚠️ Server may still be starting... Error: {e}")
    print("   Wait a few seconds and run this cell again")


# %%  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CELL 3: Pull DeepSeek-R1-Distill-Llama-8B
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
# deepseek-r1:8b is the Ollama tag for deepseek-ai/DeepSeek-R1-Distill-Llama-8B
# It's an 8B parameter model distilled from DeepSeek-R1, with strong reasoning
# capabilities — great for an AI Study Coach.
#
# Size: ~4.9 GB download, ~8 GB VRAM usage on T4
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

import subprocess

MODEL = "deepseek-r1:8b"

print(f"📥 Pulling model: {MODEL}")
print(f"   (deepseek-ai/DeepSeek-R1-Distill-Llama-8B)")
print("   This may take 3-5 minutes on first run...\n")

result = subprocess.run(
    ["ollama", "pull", MODEL],
    capture_output=False,  # Show progress in real-time
)

if result.returncode == 0:
    print(f"\n✅ Model '{MODEL}' is ready!")
else:
    print(f"\n❌ Failed to pull model. Check the output above for errors.")

# Verify model is loaded
print("\n📋 Available models:")
subprocess.run(["ollama", "list"])


# %%  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CELL 4: Quick Test — Verify DeepSeek-R1 Works
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

import json
import urllib.request

MODEL = "deepseek-r1:8b"

print(f"🧪 Testing model '{MODEL}' with a study coach prompt...\n")

test_payload = json.dumps({
    "model": MODEL,
    "messages": [
        {
            "role": "system",
            "content": "You are an AI Study Coach. You help students improve based on their quiz performance."
        },
        {
            "role": "user",
            "content": "I scored 60% on my History quiz. What should I focus on?"
        }
    ],
    "stream": False,
    "options": {"temperature": 0.7},
}).encode("utf-8")

req = urllib.request.Request(
    "http://localhost:11434/api/chat",
    data=test_payload,
    headers={"Content-Type": "application/json"},
)

try:
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.loads(resp.read().decode())
        reply = data["message"]["content"]
        print(f"🤖 DeepSeek-R1 says:\n")
        print(reply[:500])  # First 500 chars
        if len(reply) > 500:
            print(f"\n... ({len(reply)} total characters)")
        print(f"\n✅ Model is working correctly!")
except Exception as e:
    print(f"❌ Test failed: {e}")
    print("   Make sure Cell 2 (start server) and Cell 3 (pull model) succeeded.")


# %%  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CELL 5: Expose via ngrok
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
# Since you already have ngrok, just paste your auth token below.
# Get it at: https://dashboard.ngrok.com/get-started/your-authtoken
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

import subprocess

NGROK_AUTH_TOKEN = ""  # ← Paste your ngrok auth token here

if not NGROK_AUTH_TOKEN:
    print("❌ Please set your NGROK_AUTH_TOKEN above!")
    print("   Get it at: https://dashboard.ngrok.com/get-started/your-authtoken")
    raise SystemExit("ngrok token required")

# Install pyngrok
subprocess.run(["pip", "install", "pyngrok", "-q"], check=True)

from pyngrok import ngrok

# Set auth token
ngrok.set_auth_token(NGROK_AUTH_TOKEN)

# Create tunnel to Ollama port
tunnel = ngrok.connect(11434)
public_url = tunnel.public_url

MODEL = "deepseek-r1:8b"

print("=" * 60)
print("🎉 DEEPSEEK-R1 SERVER IS NOW PUBLIC!")
print("=" * 60)
print()
print(f"   🔗 Public URL:  {public_url}")
print(f"   🤖 Model:       {MODEL}")
print()
print("   Update your .env file with these values:")
print()
print(f"   COACH_OLLAMA_URL={public_url}")
print(f"   COACH_OLLAMA_MODEL={MODEL}")
print()
print("=" * 60)
print()
print("⚠️  Keep this Colab tab open! Closing it will stop the server.")
print("    Run Cell 6 below to prevent session timeout.")


# %%  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CELL 6: Keep Alive — Prevents Colab from Timing Out
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
# Runs an infinite keep-alive loop. Press Runtime → Interrupt to stop.
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

import time
import urllib.request
from datetime import datetime

print("♻️  Keep-alive loop started. Session will stay active.")
print(f"   Ollama URL: {public_url}")
print(f"   Model: deepseek-r1:8b")
print("   Press Runtime → Interrupt execution to stop.\n")

ping_count = 0
while True:
    try:
        urllib.request.urlopen("http://localhost:11434", timeout=10)
        ping_count += 1
        timestamp = datetime.now().strftime("%H:%M:%S")
        if ping_count % 10 == 0:  # Log every 5 minutes (10 * 30s)
            print(f"   [{timestamp}] ✅ Server healthy — {ping_count} pings sent")
    except Exception as e:
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"   [{timestamp}] ⚠️ Server health check failed: {e}")

    time.sleep(30)
