"""Tests for CUDA-accelerated kernels."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from dark_driving.cuda_kernels import (
    CUDAImageBatch,
    check_shared_kernels,
    fused_l1_ssim_loss_cuda,
    fused_preprocess_cuda,
    fused_psnr_ssim_cuda,
    fused_random_flip_cuda,
)


class TestFusedPreprocess:
    def test_output_shape(self):
        x = torch.randint(0, 255, (2, 3, 100, 100), dtype=torch.uint8)
        out = fused_preprocess_cuda(x, target_h=64, target_w=64)
        assert out.shape == (2, 3, 64, 64)
        assert out.dtype == torch.float32

    def test_normalization_range(self):
        x = torch.randint(0, 255, (2, 3, 64, 64), dtype=torch.uint8)
        out = fused_preprocess_cuda(x, target_h=64, target_w=64)
        assert out.min() >= 0.0
        assert out.max() <= 1.0

    def test_no_resize_needed(self):
        x = torch.randint(0, 255, (1, 3, 512, 512), dtype=torch.uint8)
        out = fused_preprocess_cuda(x, target_h=512, target_w=512)
        assert out.shape == (1, 3, 512, 512)


class TestFusedFlip:
    def test_flip_all(self):
        night = torch.randn(2, 3, 8, 8)
        day = torch.randn(2, 3, 8, 8)
        mask = torch.ones(2, dtype=torch.bool)
        n_out, d_out = fused_random_flip_cuda(night, day, mask)
        assert torch.allclose(n_out, night.flip(-1))
        assert torch.allclose(d_out, day.flip(-1))

    def test_flip_none(self):
        night = torch.randn(2, 3, 8, 8)
        day = torch.randn(2, 3, 8, 8)
        mask = torch.zeros(2, dtype=torch.bool)
        n_out, d_out = fused_random_flip_cuda(night, day, mask)
        assert torch.allclose(n_out, night)
        assert torch.allclose(d_out, day)


class TestFusedPSNRSSIM:
    def test_identical_images(self):
        x = torch.rand(2, 3, 32, 32)
        psnr, ssim = fused_psnr_ssim_cuda(x, x)
        assert psnr.item() > 30.0  # Very high PSNR for identical
        assert ssim.item() > 0.99

    def test_different_images(self):
        x = torch.rand(2, 3, 32, 32)
        y = torch.rand(2, 3, 32, 32)
        psnr, ssim = fused_psnr_ssim_cuda(x, y)
        assert psnr.item() < 30.0  # Lower PSNR for different
        assert ssim.item() < 0.5


class TestFusedLoss:
    def test_l1_only(self):
        pred = torch.randn(2, 3, 32, 32, requires_grad=True)
        target = torch.randn(2, 3, 32, 32)
        loss = fused_l1_ssim_loss_cuda(pred, target, ssim_weight=0.0)
        assert loss.shape == ()
        assert loss.item() > 0
        loss.backward()
        assert pred.grad is not None

    def test_l1_plus_ssim(self):
        pred = torch.randn(2, 3, 32, 32, requires_grad=True)
        target = torch.randn(2, 3, 32, 32)
        loss = fused_l1_ssim_loss_cuda(pred, target, ssim_weight=0.1)
        assert loss.shape == ()
        loss.backward()
        assert pred.grad is not None


class TestCUDAImageBatch:
    def test_put_get(self):
        device = torch.device("cpu")  # Test on CPU
        cache = CUDAImageBatch(device, max_cached=4)
        night = torch.randn(3, 64, 64)
        day = torch.randn(3, 64, 64)
        cache.put(0, night, day)
        result = cache.get(0)
        assert result is not None
        assert torch.allclose(result[0], night)

    def test_eviction(self):
        device = torch.device("cpu")
        cache = CUDAImageBatch(device, max_cached=2)
        for i in range(3):
            cache.put(i, torch.randn(3, 8, 8), torch.randn(3, 8, 8))
        # Oldest (idx=0) should be evicted
        assert cache.get(0) is None
        assert cache.get(1) is not None
        assert cache.get(2) is not None

    def test_clear(self):
        device = torch.device("cpu")
        cache = CUDAImageBatch(device, max_cached=4)
        cache.put(0, torch.randn(3, 8, 8), torch.randn(3, 8, 8))
        cache.clear()
        assert cache.get(0) is None


class TestSharedKernels:
    def test_check_returns_dict(self):
        result = check_shared_kernels()
        assert isinstance(result, dict)
