# TRACTION (Agri-Scout)

Streamlit dashboard for real-time crop disease detection, farm map monitoring, data visualization, and AI treatment procedures.

## Quick setup (recommended)

```bash
./setup.sh
```

What this does:
- Creates `.venv`
- Installs Python dependencies
- Installs `ollama` (macOS/Linux when possible)
- Starts `ollama serve` if needed
- Pulls the default model (`llama3.1:8b`)

Then run:

```bash
source .venv/bin/activate
streamlit run app.py
```

## Ollama-first AI Procedure

The `AI Procedure` tab uses Ollama as the primary LLM path by default.

Optional config via env vars or Streamlit secrets:
- `OLLAMA_BASE_URL` (default: `http://127.0.0.1:11434`)
- `OLLAMA_MODEL` (default: `llama3.1:8b`)

You can still configure fallback callables with:
- `LLM_ADVISOR_CALLABLE`
- `TRACTION_LLM_CALLABLE`
