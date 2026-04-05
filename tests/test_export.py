"""Tests for DarkDriving export pipeline."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from dark_driving.model import get_model


class TestExportSafetensors:
    def test_export_and_reload(self, tmp_path):
        from safetensors.torch import load_file, save_file

        model = get_model("retinexformer", embed_dim=8, num_blocks=1, num_heads=1)
        path = tmp_path / "model.safetensors"
        save_file(model.state_dict(), str(path))
        assert path.exists()
        assert path.stat().st_size > 0

        # Reload
        loaded = load_file(str(path))
        model2 = get_model("retinexformer", embed_dim=8, num_blocks=1, num_heads=1)
        model2.load_state_dict(loaded)

        # Verify outputs match
        x = torch.randn(1, 3, 32, 32)
        model.eval()
        model2.eval()
        with torch.no_grad():
            y1 = model(x)
            y2 = model2(x)
        assert torch.allclose(y1, y2, atol=1e-6)


class TestExportONNX:
    def test_onnx_export(self, tmp_path):
        model = get_model("retinexformer", embed_dim=8, num_blocks=1, num_heads=1)
        model.eval()
        path = tmp_path / "model.onnx"
        dummy = torch.randn(1, 3, 64, 64)

        torch.onnx.export(
            model, dummy, str(path),
            opset_version=17,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        )
        assert path.exists()
        assert path.stat().st_size > 0

    def test_onnx_inference(self, tmp_path):
        import numpy as np

        model = get_model("retinexformer", embed_dim=8, num_blocks=1, num_heads=1)
        model.eval()
        path = tmp_path / "model.onnx"
        dummy = torch.randn(1, 3, 64, 64)

        torch.onnx.export(
            model, dummy, str(path),
            opset_version=17,
            input_names=["input"],
            output_names=["output"],
        )

        # Run with onnxruntime
        import onnxruntime as ort
        sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        result = sess.run(None, {"input": dummy.numpy()})
        assert result[0].shape == (1, 3, 64, 64)

        # Compare to PyTorch
        with torch.no_grad():
            pt_out = model(dummy).numpy()
        max_diff = np.max(np.abs(pt_out - result[0]))
        assert max_diff < 1e-3, f"ONNX vs PyTorch max_diff={max_diff}"


class TestExportPipeline:
    def test_full_model_save_load(self, tmp_path):
        """Test the complete checkpoint -> load -> verify cycle."""
        model = get_model("retinexformer", embed_dim=8, num_blocks=1, num_heads=1)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        # Save checkpoint
        state = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": 5,
            "step": 500,
            "metrics": {"val_psnr": 23.5, "val_ssim": 0.85},
        }
        ckpt_path = tmp_path / "best.pth"
        torch.save(state, ckpt_path)

        # Reload
        ckpt = torch.load(ckpt_path, weights_only=False)
        model2 = get_model("retinexformer", embed_dim=8, num_blocks=1, num_heads=1)
        model2.load_state_dict(ckpt["model"])

        assert ckpt["epoch"] == 5
        assert ckpt["metrics"]["val_psnr"] == 23.5

        # Verify outputs match
        x = torch.randn(1, 3, 32, 32)
        model.eval()
        model2.eval()
        with torch.no_grad():
            y1 = model(x)
            y2 = model2(x)
        assert torch.allclose(y1, y2, atol=1e-6)
