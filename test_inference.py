import onnxruntime as ort
import numpy as np
from PIL import Image

# ==============================================================================
#  CONFIGURATION
# ==============================================================================
MODEL_PATH = "models/agri_scout_mobilenetv2.onnx"
IMAGE_PATH = "/Users/siddhant/Downloads/Agri-Scout/dataset/archive/test/test/CornCommonRust1.JPG"

# These must perfectly match the alphabetical order of your training folders
CLASSES = [
    "Cercospora Leaf Spot / Gray Leaf Spot",
    "Common Rust",
    "Healthy"    
    "Northern Leaf Blight",
]

def preprocess_image(image_path: str) -> np.ndarray:
    """Replicates the exact mathematical transforms used during training."""
    # 1. Load and convert to RGB
    img = Image.open(image_path).convert('RGB')
    
    # 2. Resize to 256x256
    img = img.resize((256, 256), Image.BILINEAR)
    
    # 3. Center Crop to 224x224
    left = (256 - 224) / 2
    top = (256 - 224) / 2
    right = left + 224
    bottom = top + 224
    img = img.crop((left, top, right, bottom))
    
    # 4. Convert to numpy array and scale to [0, 1]
    img_data = np.array(img).astype(np.float32) / 255.0
    
    # 5. Normalize using ImageNet means and stds
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_data = (img_data - mean) / std
    
    # 6. Transpose from (Height, Width, Channels) to (Channels, Height, Width)
    img_data = np.transpose(img_data, (2, 0, 1))
    
    # 7. Add the batch dimension: (1, Channels, Height, Width)
    img_data = np.expand_dims(img_data, axis=0)
    
    return img_data

def main():
    print("=" * 50)
    print(" Agri-Scout | Local ONNX Inference Test")
    print("=" * 50)
    
    print(f"\n[1] Loading model: {MODEL_PATH}")
    session = ort.InferenceSession(MODEL_PATH)
    input_name = session.get_inputs()[0].name
    
    print(f"[2] Processing image: {IMAGE_PATH.split('/')[-1]}")
    img_tensor = preprocess_image(IMAGE_PATH)
    
    print("[3] Running AI inference...")
    outputs = session.run(None, {input_name: img_tensor})
    
    # The output is raw numbers (logits). We use softmax to get readable percentages.
    logits = outputs[0][0] 
    exp_logits = np.exp(logits - np.max(logits)) # Subtract max for numerical stability
    probabilities = exp_logits / np.sum(exp_logits)
    
    # Find the class with the highest probability
    best_idx = np.argmax(probabilities)
    best_class = CLASSES[best_idx]
    confidence = probabilities[best_idx] * 100
    
    print("\n" + "-" * 50)
    print(f" 🎯 DETECTION  : {best_class}")
    print(f" 📊 CONFIDENCE : {confidence:.2f}%")
    print("-" * 50 + "\n")

if __name__ == "__main__":
    main()