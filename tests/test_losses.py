"""Tests for DarkDriving loss functions."""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from dark_driving.losses import (
    CharbonnierLoss,
    CombinedLoss,
    L1Loss,
    SSIMLoss,
    build_loss,
)


@pytest.fixture
def pred_target():
    """Random pred/target pair (B=2, C=3, H=64, W=64)."""
    pred = torch.randn(2, 3, 64, 64, requires_grad=True)
    target = torch.randn(2, 3, 64, 64)
    return pred, target


class TestL1Loss:
    def test_scalar_output(self, pred_target):
        pred, target = pred_target
        loss = L1Loss()(pred, target)
        assert loss.shape == ()
        assert loss.item() > 0

    def test_gradient_flow(self, pred_target):
        pred, target = pred_target
        loss = L1Loss()(pred, target)
        loss.backward()
        assert pred.grad is not None

    def test_zero_on_identical(self):
        x = torch.randn(2, 3, 32, 32)
        loss = L1Loss()(x, x)
        assert loss.item() < 1e-6


class TestCharbonnierLoss:
    def test_scalar_output(self, pred_target):
        pred, target = pred_target
        loss = CharbonnierLoss()(pred, target)
        assert loss.shape == ()
        assert loss.item() > 0

    def test_gradient_flow(self, pred_target):
        pred, target = pred_target
        loss = CharbonnierLoss()(pred, target)
        loss.backward()
        assert pred.grad is not None

    def test_near_zero_on_identical(self):
        x = torch.randn(2, 3, 32, 32)
        loss = CharbonnierLoss(eps=1e-6)(x, x)
        assert loss.item() < 1e-3


class TestSSIMLoss:
    def test_scalar_output(self, pred_target):
        pred, target = pred_target
        loss = SSIMLoss(window_size=7)(pred, target)
        assert loss.shape == ()

    def test_range(self, pred_target):
        pred, target = pred_target
        loss = SSIMLoss(window_size=7)(pred, target)
        # SSIM loss = 1 - SSIM, should be in [0, 2]
        assert 0.0 <= loss.item() <= 2.0

    def test_identical_images(self):
        x = torch.rand(2, 3, 32, 32)
        loss = SSIMLoss(window_size=7)(x, x)
        # SSIM of identical images should be ~1, so loss ~0
        assert loss.item() < 0.01


class TestCombinedLoss:
    def test_l1_only(self, pred_target):
        pred, target = pred_target
        loss_fn = CombinedLoss(l1_weight=1.0, ssim_weight=0.0)
        total, loss_dict = loss_fn(pred, target)
        assert "l1" in loss_dict
        assert "total" in loss_dict
        assert total.item() > 0

    def test_l1_plus_ssim(self, pred_target):
        pred, target = pred_target
        loss_fn = CombinedLoss(l1_weight=1.0, ssim_weight=0.5)
        total, loss_dict = loss_fn(pred, target)
        assert "l1" in loss_dict
        assert "ssim" in loss_dict
        assert total.item() > 0

    def test_gradient_flow(self, pred_target):
        pred, target = pred_target
        loss_fn = CombinedLoss(l1_weight=1.0, ssim_weight=0.1)
        total, _ = loss_fn(pred, target)
        total.backward()
        assert pred.grad is not None


class TestBuildLoss:
    def test_default_config(self):
        config = {"l1_weight": 1.0}
        loss_fn = build_loss(config)
        assert isinstance(loss_fn, CombinedLoss)

    def test_custom_weights(self):
        config = {"l1_weight": 0.5, "ssim_weight": 0.3, "charbonnier_weight": 0.2}
        loss_fn = build_loss(config)
        pred = torch.randn(2, 3, 32, 32, requires_grad=True)
        target = torch.randn(2, 3, 32, 32)
        total, loss_dict = loss_fn(pred, target)
        assert "l1" in loss_dict
        assert "ssim" in loss_dict
        assert "charbonnier" in loss_dict
        total.backward()
        assert pred.grad is not None
