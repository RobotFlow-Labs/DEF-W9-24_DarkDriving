"""AnimaNode for DarkDriving -- Docker serving endpoint.

Provides low-light image enhancement as a service:
- POST /predict: enhance a low-light driving image
- GET /health: service health
- GET /ready: model loaded check

ROS2 topics:
- Subscribe: /camera/night/image_raw
- Publish: /darkdriving/enhanced, /darkdriving/metrics
"""

from __future__ import annotations

import base64
import io
import time
from typing import Any

import numpy as np
import torch
from PIL import Image

from dark_driving.model import get_model


class DarkDrivingNode:
    """DarkDriving enhancement inference node.

    Loads an enhancement model and provides predict() for serving.
    Compatible with anima_serve.node.AnimaNode interface.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}
        self.model: torch.nn.Module | None = None
        self.device = torch.device("cpu")
        self.input_size = (512, 512)
        self._ready = False

    def setup_inference(self) -> None:
        """Load model weights and configure backend."""
        model_cfg = self.config.get("model", {})

        # Device detection
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        # Build model
        self.model = get_model(
            name=model_cfg.get("name", "retinexformer"),
            in_channels=model_cfg.get("in_channels", 3),
            out_channels=model_cfg.get("out_channels", 3),
            embed_dim=model_cfg.get("embed_dim", 32),
            num_blocks=model_cfg.get("num_blocks", 4),
            num_heads=model_cfg.get("num_heads", 4),
        ).to(self.device)

        # Load weights if available
        weight_path = self.config.get("weight_path")
        if weight_path:
            ckpt = torch.load(weight_path, map_location=self.device, weights_only=False)
            if "model" in ckpt:
                self.model.load_state_dict(ckpt["model"])
            else:
                self.model.load_state_dict(ckpt)

        self.model.eval()
        self._ready = True

    @torch.no_grad()
    def process(self, input_data: dict[str, Any]) -> dict[str, Any]:
        """Run enhancement inference.

        Args:
            input_data: Dict with 'image' key (base64 string or numpy array).

        Returns:
            Dict with 'enhanced' (base64), 'metrics' (dict), 'latency_ms' (float).
        """
        if not self._ready or self.model is None:
            return {"error": "Model not loaded. Call setup_inference() first."}

        t0 = time.time()

        # Decode input
        if "image" in input_data:
            img_data = input_data["image"]
            if isinstance(img_data, str):
                # Base64 encoded
                img_bytes = base64.b64decode(img_data)
                img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            elif isinstance(img_data, np.ndarray):
                img = Image.fromarray(img_data)
            else:
                return {"error": "Unsupported image format"}
        else:
            return {"error": "Missing 'image' key in input_data"}

        # Preprocess
        h, w = self.input_size
        img_resized = img.resize((w, h), Image.BILINEAR)
        img_np = np.array(img_resized, dtype=np.float32) / 255.0
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).to(self.device)

        # Enhance
        with torch.amp.autocast("cuda", enabled=self.device.type == "cuda"):
            enhanced_tensor = self.model(img_tensor)

        enhanced_tensor = enhanced_tensor.clamp(0.0, 1.0)

        # Post-process
        enhanced_np = (
            enhanced_tensor[0].cpu().permute(1, 2, 0).numpy() * 255.0
        ).astype(np.uint8)
        enhanced_img = Image.fromarray(enhanced_np)

        # Encode output
        buffer = io.BytesIO()
        enhanced_img.save(buffer, format="PNG")
        enhanced_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

        latency_ms = (time.time() - t0) * 1000.0

        return {
            "enhanced": enhanced_b64,
            "latency_ms": latency_ms,
            "input_size": list(self.input_size),
        }

    def get_status(self) -> dict[str, Any]:
        """Module-specific status fields."""
        return {
            "model_loaded": self._ready,
            "device": str(self.device),
            "input_size": list(self.input_size),
            "model_name": self.config.get("model", {}).get("name", "unknown"),
        }
