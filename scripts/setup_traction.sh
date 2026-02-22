#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

MODEL_NAME="${OLLAMA_MODEL:-llama3.1:8b}"

echo "[setup] Root: $ROOT_DIR"

if [ ! -d ".venv" ]; then
  echo "[setup] Creating virtual environment..."
  python3 -m venv .venv
fi

source .venv/bin/activate

echo "[setup] Installing Python dependencies..."
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

if [ -f ".streamlit/secrets.toml.example" ] && [ ! -f ".streamlit/secrets.toml" ]; then
  echo "[setup] Creating .streamlit/secrets.toml from example..."
  mkdir -p .streamlit
  cp .streamlit/secrets.toml.example .streamlit/secrets.toml
fi

if ! command -v ollama >/dev/null 2>&1; then
  echo "[setup] Ollama not found. Installing..."
  case "$(uname -s)" in
    Darwin)
      if command -v brew >/dev/null 2>&1; then
        brew install --cask ollama || brew install ollama
      else
        echo "[setup] Homebrew not found. Install Ollama manually from https://ollama.com/download"
      fi
      ;;
    Linux)
      curl -fsSL https://ollama.com/install.sh | sh
      ;;
    *)
      echo "[setup] Unsupported OS for automatic Ollama install."
      ;;
  esac
fi

if command -v ollama >/dev/null 2>&1; then
  echo "[setup] Starting Ollama server if needed..."
  if ! ollama list >/dev/null 2>&1; then
    nohup ollama serve >/tmp/ollama-traction.log 2>&1 &
    sleep 3
  fi
  echo "[setup] Pulling Ollama model: $MODEL_NAME"
  ollama pull "$MODEL_NAME" || true
else
  echo "[setup] Ollama CLI still not available. The app can run, but AI Procedure will use fallback output."
fi

echo "[setup] Done."
echo "[setup] Run the app with: streamlit run app.py"
