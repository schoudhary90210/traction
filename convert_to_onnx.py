#!/usr/bin/env python3
"""
================================================================================
 Agri-Scout — Standalone ONNX Conversion Script
================================================================================
 Purpose : Convert a trained MobileNetV2 .pth checkpoint to a FULL .onnx file
 Problem : PyTorch 2.x Dynamo exporter drops frozen (requires_grad=False)
           parameters during tracing, producing a ~0.26 MB file (head only).
 Fix     : 1. Rebuild the architecture & load weights from .pth
           2. Unfreeze ALL parameters (requires_grad = True)
           3. Force the LEGACY JIT/TorchScript tracer (not Dynamo)
           4. Validate exported file is > 8 MB before declaring success
================================================================================
 Usage   : python convert_to_onnx.py
================================================================================
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models


# ==============================================================================
#  CONFIGURATION
# ==============================================================================

PTH_PATH       = "./models/agri_scout_mobilenetv2.pth"
ONNX_PATH      = "./models/agri_scout_mobilenetv2.onnx"
NUM_CLASSES    = 4
INPUT_SIZE     = 224
OPSET_VERSION  = 14          # Qualcomm AI Hub / QNN compatible
MIN_FILE_MB    = 8.0         # Sanity threshold — full MobileNetV2 is ~9-14 MB


# ==============================================================================
#  STEP 1 — Rebuild the exact architecture used during training
# ==============================================================================

def rebuild_model(num_classes: int) -> nn.Module:
    """
    Instantiate MobileNetV2 with the SAME classifier head that was used
    during training, so the state_dict keys match exactly.
    """
    # Start from the stock architecture (no pretrained weights — we'll load ours)
    model = models.mobilenet_v2(weights=None)

    # Replace the classifier head to match the training script
    in_features = model.classifier[1].in_features  # 1280
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(in_features, num_classes),
    )

    return model


# ==============================================================================
#  STEP 2 — Load weights & unfreeze everything
# ==============================================================================

def load_and_unfreeze(model: nn.Module, pth_path: str) -> nn.Module:
    """
    Load the saved state_dict and ensure every single parameter has
    requires_grad=True so the tracer sees the full graph.
    """
    pth = Path(pth_path)
    if not pth.exists():
        print(f"❌  Checkpoint not found: {pth.resolve()}")
        sys.exit(1)

    state_dict = torch.load(str(pth), map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict, strict=True)
    print(f"[LOAD]  Weights loaded from: {pth}")

    # --- THE CRITICAL FIX ---
    # Unfreeze every parameter so the JIT tracer includes the full backbone.
    frozen_count = 0
    for param in model.parameters():
        if not param.requires_grad:
            param.requires_grad = True
            frozen_count += 1

    total_params = sum(p.numel() for p in model.parameters())
    print(f"[FIX]   Unfroze {frozen_count} parameter tensors")
    print(f"[FIX]   Total parameters now visible to tracer: {total_params:,}")

    return model


# ==============================================================================
#  STEP 3 — Export using the LEGACY JIT tracer (bypasses Dynamo entirely)
# ==============================================================================

def export_onnx(model: nn.Module, onnx_path: str) -> Path:
    """
    Export the full model to ONNX using the legacy TorchScript-based tracer.
    PyTorch 2.6+ defaults to the Dynamo exporter which drops frozen params.
    """
    out = Path(onnx_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    model.eval()
    model.to("cpu")

    dummy = torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE)

    print(f"\n[ONNX]  Exporting with LEGACY tracer (opset {OPSET_VERSION})...")

    # -------------------------------------------------------------------------
    # Force the legacy (TorchScript / JIT) ONNX exporter.
    #
    # PyTorch >= 2.6 switched the default torch.onnx.export to the new
    # Dynamo-based exporter, which traces only the "live" computation graph.
    # Frozen parameters (requires_grad=False) appear as constants that Dynamo
    # may fold or discard, producing a tiny .onnx file.
    #
    # Passing  dynamo=False  forces the old JIT-trace path which serialises
    # every nn.Parameter regardless of its grad flag.
    # -------------------------------------------------------------------------
    export_kwargs = dict(
        model=model,
        args=(dummy,),
        f=str(out),
        export_params=True,
        opset_version=OPSET_VERSION,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input":  {0: "batch_size"},
            "output": {0: "batch_size"},
        },
    )

    # PyTorch >= 2.6 accepts `dynamo=False` to opt out of the new exporter.
    # Older versions don't recognise the kwarg, so we try/except gracefully.
    try:
        torch.onnx.export(**export_kwargs, dynamo=False)
        print("[ONNX]  Used dynamo=False  (legacy JIT tracer)")
    except TypeError:
        # Fallback for PyTorch < 2.6 where dynamo kwarg doesn't exist.
        # The legacy tracer is already the default in those versions.
        torch.onnx.export(**export_kwargs)
        print("[ONNX]  Used default tracer (PyTorch < 2.6)")

    return out


# ==============================================================================
#  STEP 4 — Validate the export
# ==============================================================================

def validate_export(onnx_path: Path) -> None:
    """Check file size and optionally run onnx.checker for structural validity."""

    size_mb = onnx_path.stat().st_size / (1024 * 1024)
    print(f"\n{'=' * 60}")
    print(f"  Export path : {onnx_path.resolve()}")
    print(f"  File size   : {size_mb:.2f} MB")
    print(f"  Opset       : {OPSET_VERSION}")
    print(f"  Input shape : (1, 3, {INPUT_SIZE}, {INPUT_SIZE})")
    print(f"{'=' * 60}")

    if size_mb < MIN_FILE_MB:
        print(f"\n❌  FAILED — File is {size_mb:.2f} MB (expected > {MIN_FILE_MB} MB).")
        print("    The backbone was likely dropped again. Debug steps:")
        print("    1. Confirm all params are unfrozen before export.")
        print("    2. Ensure torch.onnx.export is using the JIT tracer.")
        sys.exit(1)

    # Optional: deep structural check (requires `onnx` package)
    try:
        import onnx
        onnx_model = onnx.load(str(onnx_path))
        onnx.checker.check_model(onnx_model)
        print("\n✅  ONNX structural check PASSED (onnx.checker)")
    except ImportError:
        print("\n⚠️   `onnx` package not installed — skipping structural check.")
        print("    Install with:  pip install onnx")
    except Exception as e:
        print(f"\n❌  ONNX structural check FAILED: {e}")
        sys.exit(1)

    print(f"\n✅  SUCCESS — Model is {size_mb:.2f} MB and ready for Qualcomm AI Hub.")
    print("    Next step:  qai-hub compile agri_scout_mobilenetv2.onnx --device \"Snapdragon X Elite\"")


# ==============================================================================
#  MAIN
# ==============================================================================

def main() -> None:
    print("=" * 60)
    print("  Agri-Scout  |  .pth → .onnx Conversion")
    print("  Fix for PyTorch 2.x Dynamo exporter dropping frozen weights")
    print("=" * 60 + "\n")

    model = rebuild_model(NUM_CLASSES)
    model = load_and_unfreeze(model, PTH_PATH)
    onnx_path = export_onnx(model, ONNX_PATH)
    validate_export(onnx_path)


if __name__ == "__main__":
    main()