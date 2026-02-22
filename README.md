# Agri-Scout v4 — Real-Time Crop Disease Detection Dashboard

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run app.py
```

## Offline AI Treatment Plans with Ollama

Agri-Scout now includes **fast, offline AI treatment plan generation** using [Ollama](https://ollama.ai/). When a disease is detected, the system automatically generates actionable agricultural advice without cloud dependencies.

### Setup Ollama

#### 1. Install Ollama
- **macOS/Linux**: Download from https://ollama.ai/download
- **Windows**: Download from https://ollama.ai/download (requires WSL2)

#### 2. Start the Ollama Server
```bash
ollama serve
```

This starts the Ollama API server on `http://localhost:11434` (default).

#### 3. Pull the Mistral Model
In a new terminal:
```bash
ollama pull mistral
```

The `mistral` model is lightweight (~7GB) and provides fast, accurate agricultural advice.

### How It Works

1. **Disease Detection**: Vision engine detects a disease with confidence ≥ 85%
2. **Treatment Generation**: Disease name + confidence sent to local Ollama
3. **AI Planning**: Mistral model generates treatment advice (~2-3 sentences) covering:
   - Immediate action (fungicide/rotation/isolation)
   - Prevention measures
   - Monitoring frequency
4. **Dashboard Display**: Treatment plan appears in the "AI Treatment Advisory" panel
5. **Fallback Mode**: If Ollama is unavailable, static USDA-guidelines-based advice is shown

### Example Output

```
💊 AI Treatment Plan:
Apply copper or sulfur-based fungicide every 7-10 days.
Remove infected leaves and improve air circulation.
Check daily; fungicide-resistant strains possible.
```

### Performance

- **Latency**: ~1-2 seconds per treatment plan (first call slower due to model load)
- **Offline**: No internet required after model download
- **Resource**: ~7GB disk for Mistral model, minimal RAM/CPU overhead
- **Cost**: Free and open-source

### Configuration

Edit `app.py` to customize:
```python
_OLLAMA_BASE_URL = "http://localhost:11434"  # Ollama server URL
_OLLAMA_MODEL = "mistral"                     # Model name
_OLLAMA_TIMEOUT = 10                          # Timeout in seconds
```

To use a different Ollama model:
```bash
ollama pull llama2  # or any other available model
```
Then update `_OLLAMA_MODEL = "llama2"` in `app.py`.

### Troubleshooting

**Q: "Ollama not found" error**
- Ensure Ollama server is running: `ollama serve`
- Check firewall isn't blocking `localhost:11434`

**Q: Slow treatment generation**
- First inference loads the model into memory (~10-20 sec)
- Subsequent calls are faster. Run a quick test to warm up

**Q: Want faster generation?**
- Try `ollama pull neural-chat` (faster, smaller model)
- Or use pre-generated static advice (automatic fallback)

---

## Other Features

- **Real-time Video Stream**: Live crop disease detection via WebRTC
- **GPS Integration**: Tractor simulator with disease clustering and logging
- **Interactive Maps**: Google Maps heatmaps with disease hotspots
- **ONNX Model**: Lightweight MobileNetV2 inference on edge devices
- **Visual Analytics**: Disease distribution graphs and risk bands
- **Settings Profiles**: Save and load farm configurations

## Dataset

Uses the PlantVillage corn disease dataset (Augmented):
- Corn Cercospora (Gray Leaf Spot)
- Corn Common Rust
- Corn Northern Leaf Blight
- Corn Healthy

## Model

- **Architecture**: MobileNetV2 (transfer learning)
- **Format**: ONNX (opset 14, Qualcomm AI Hub compatible)
- **Size**: ~9 MB
- **Inference**: ~50-100ms on Snapdragon X Elite NPU

## License & Credits

Built for Qualcomm Edge AI Hackathon.
