"""Tests for DarkDriving enhancement models."""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from dark_driving.model import (
    LLFormer,
    Retinexformer,
    SNRAwareEnhancer,
    UNetEnhancer,
    count_parameters,
    get_model,
)


@pytest.fixture
def dummy_input():
    """Create a dummy input tensor (B=2, C=3, H=64, W=64)."""
    return torch.randn(2, 3, 64, 64)


class TestRetinexformer:
    def test_forward_shape(self, dummy_input):
        model = Retinexformer(embed_dim=16, num_blocks=2, num_heads=2)
        out = model(dummy_input)
        assert out.shape == dummy_input.shape

    def test_output_not_equal_input(self, dummy_input):
        model = Retinexformer(embed_dim=16, num_blocks=2, num_heads=2)
        out = model(dummy_input)
        assert not torch.allclose(out, dummy_input, atol=1e-6)

    def test_gradient_flow(self, dummy_input):
        model = Retinexformer(embed_dim=16, num_blocks=2, num_heads=2)
        out = model(dummy_input)
        loss = out.mean()
        loss.backward()
        for p in model.parameters():
            if p.requires_grad:
                assert p.grad is not None


class TestSNRAware:
    def test_forward_shape(self, dummy_input):
        model = SNRAwareEnhancer(embed_dim=16, num_blocks=2, num_heads=2)
        out = model(dummy_input)
        assert out.shape == dummy_input.shape

    def test_gradient_flow(self, dummy_input):
        model = SNRAwareEnhancer(embed_dim=16, num_blocks=2, num_heads=2)
        out = model(dummy_input)
        loss = out.mean()
        loss.backward()
        has_grad = any(p.grad is not None for p in model.parameters() if p.requires_grad)
        assert has_grad


class TestLLFormer:
    def test_forward_shape(self, dummy_input):
        model = LLFormer(embed_dim=16, num_blocks=2, num_heads=2)
        out = model(dummy_input)
        assert out.shape == dummy_input.shape

    def test_gradient_flow(self, dummy_input):
        model = LLFormer(embed_dim=16, num_blocks=2, num_heads=2)
        out = model(dummy_input)
        loss = out.mean()
        loss.backward()
        has_grad = any(p.grad is not None for p in model.parameters() if p.requires_grad)
        assert has_grad


class TestUNet:
    def test_forward_shape(self, dummy_input):
        model = UNetEnhancer(embed_dim=16)
        out = model(dummy_input)
        assert out.shape == dummy_input.shape

    def test_gradient_flow(self, dummy_input):
        model = UNetEnhancer(embed_dim=16)
        out = model(dummy_input)
        loss = out.mean()
        loss.backward()
        has_grad = any(p.grad is not None for p in model.parameters() if p.requires_grad)
        assert has_grad


class TestModelRegistry:
    def test_get_retinexformer(self):
        model = get_model("retinexformer", embed_dim=16, num_blocks=2, num_heads=2)
        assert isinstance(model, Retinexformer)

    def test_get_snr_aware(self):
        model = get_model("snr_aware", embed_dim=16, num_blocks=2, num_heads=2)
        assert isinstance(model, SNRAwareEnhancer)

    def test_get_llformer(self):
        model = get_model("llformer", embed_dim=16, num_blocks=2, num_heads=2)
        assert isinstance(model, LLFormer)

    def test_get_unet(self):
        model = get_model("unet", embed_dim=16)
        assert isinstance(model, UNetEnhancer)

    def test_invalid_name_raises(self):
        with pytest.raises(ValueError, match="Unknown model"):
            get_model("nonexistent_model")

    def test_count_parameters(self):
        model = get_model("retinexformer", embed_dim=16, num_blocks=2, num_heads=2)
        n_params = count_parameters(model)
        assert n_params > 0
        assert isinstance(n_params, int)
