"""Tests for DarkDriving training pipeline."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from dark_driving.losses import build_loss
from dark_driving.model import get_model
from dark_driving.utils import (
    CheckpointManager,
    EarlyStopping,
    WarmupCosineScheduler,
    load_config,
    set_seed,
)


class TestCheckpointManager:
    def test_save_and_prune(self, tmp_path):
        mgr = CheckpointManager(save_dir=tmp_path, keep_top_k=2, metric="val_psnr", mode="max")
        for i, val in enumerate([10.0, 20.0, 15.0, 25.0]):
            state = {"model": {"w": torch.randn(3)}, "epoch": i}
            mgr.save(state, val, step=i * 100)

        # Should keep top 2 + best.pth
        pth_files = list(tmp_path.glob("checkpoint_*.pth"))
        assert len(pth_files) == 2
        assert (tmp_path / "best.pth").exists()
        assert mgr.best_metric == 25.0

    def test_best_pth_loadable(self, tmp_path):
        mgr = CheckpointManager(save_dir=tmp_path, keep_top_k=1)
        model = get_model("retinexformer", embed_dim=8, num_blocks=1, num_heads=1)
        state = {"model": model.state_dict(), "epoch": 1}
        mgr.save(state, 15.0, step=100)

        ckpt = torch.load(tmp_path / "best.pth", weights_only=False)
        assert "model" in ckpt
        model.load_state_dict(ckpt["model"])


class TestEarlyStopping:
    def test_no_stop_when_improving(self):
        es = EarlyStopping(patience=3, mode="max")
        assert not es.step(1.0)
        assert not es.step(2.0)
        assert not es.step(3.0)

    def test_stop_after_patience(self):
        es = EarlyStopping(patience=3, mode="max", min_delta=0.0)
        es.step(10.0)
        es.step(9.0)
        es.step(9.0)
        assert es.step(9.0)  # 3 epochs no improvement

    def test_min_mode(self):
        es = EarlyStopping(patience=2, mode="min", min_delta=0.0)
        es.step(1.0)
        es.step(1.1)
        assert es.step(1.1)


class TestWarmupCosineScheduler:
    def test_warmup_increases_lr(self):
        model = get_model("retinexformer", embed_dim=8, num_blocks=1, num_heads=1)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        sched = WarmupCosineScheduler(opt, warmup_steps=10, total_steps=100)

        lrs = []
        for _ in range(10):
            sched.step()
            lrs.append(sched.get_lr())

        # LR should increase during warmup
        assert lrs[-1] > lrs[0]

    def test_cosine_decreases_lr(self):
        model = get_model("retinexformer", embed_dim=8, num_blocks=1, num_heads=1)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        sched = WarmupCosineScheduler(opt, warmup_steps=5, total_steps=50)

        # Skip warmup
        for _ in range(5):
            sched.step()
        lr_after_warmup = sched.get_lr()

        # Run cosine phase
        for _ in range(40):
            sched.step()
        lr_end = sched.get_lr()

        assert lr_end < lr_after_warmup

    def test_state_dict_roundtrip(self):
        model = get_model("retinexformer", embed_dim=8, num_blocks=1, num_heads=1)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        sched = WarmupCosineScheduler(opt, warmup_steps=10, total_steps=100)

        for _ in range(15):
            sched.step()

        state = sched.state_dict()
        assert state["current_step"] == 15

        # Restore
        sched2 = WarmupCosineScheduler(opt, warmup_steps=10, total_steps=100)
        sched2.load_state_dict(state)
        assert sched2.current_step == 15


class TestTrainingSmoke:
    def test_single_step(self):
        """Verify a single training step works end-to-end."""
        set_seed(42)
        device = torch.device("cpu")

        model = get_model("retinexformer", embed_dim=8, num_blocks=1, num_heads=1).to(device)
        loss_fn = build_loss({"l1_weight": 1.0}).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        # Synthetic data
        night = torch.randn(2, 3, 64, 64, device=device)
        day = torch.randn(2, 3, 64, 64, device=device)

        model.train()
        enhanced = model(night)
        loss, loss_dict = loss_fn(enhanced, day)

        assert not torch.isnan(loss)
        assert loss.item() > 0

        loss.backward()
        optimizer.step()

        # Verify params changed
        model.eval()
        with torch.no_grad():
            enhanced2 = model(night)
        assert not torch.allclose(enhanced, enhanced2, atol=1e-6)

    def test_config_loading(self):
        """Verify config files parse correctly."""
        project_root = Path(__file__).resolve().parent.parent
        for cfg_name in ["paper.toml", "debug.toml"]:
            cfg_path = project_root / "configs" / cfg_name
            if cfg_path.exists():
                config = load_config(str(cfg_path))
                assert "model" in config
                assert "training" in config
                assert "loss" in config
