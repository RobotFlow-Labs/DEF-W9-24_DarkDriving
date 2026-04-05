"""Enhancement model implementations for DarkDriving benchmark.

Implements Retinexformer (primary), SNR-Aware, LLFormer wrappers,
and a generic UNet baseline. All models follow:
  Input: (B, 3, H, W) low-light image
  Output: (B, 3, H, W) enhanced image
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as f
from einops import rearrange

# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


class LayerNorm2d(nn.Module):
    """Channel-last LayerNorm adapted for (B, C, H, W) tensors."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[None, :, None, None] * x + self.bias[None, :, None, None]
        return x


class FeedForward(nn.Module):
    """GELU feed-forward block: Conv1x1 -> GELU -> Conv1x1."""

    def __init__(self, dim: int, mult: int = 4):
        super().__init__()
        hidden = dim * mult
        self.net = nn.Sequential(
            nn.Conv2d(dim, hidden, 1),
            nn.GELU(),
            nn.Conv2d(hidden, dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Retinexformer components
# ---------------------------------------------------------------------------


class IlluminationEstimator(nn.Module):
    """Shallow CNN that estimates an illumination map from the input image.

    Architecture: Conv -> ReLU -> Conv -> Conv (outputs illumination map).
    The lit image is computed as: input * illumination_map.
    """

    def __init__(self, in_channels: int = 3, embed_dim: int = 32):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, embed_dim, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(embed_dim, embed_dim, 3, padding=1)
        self.conv3 = nn.Conv2d(embed_dim, in_channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (illumination_map, features)."""
        feat = self.relu(self.conv1(x))
        feat = self.relu(self.conv2(feat))
        illu_map = torch.sigmoid(self.conv3(feat))
        return illu_map, feat


class IlluminationGuidedMSA(nn.Module):
    """Illumination-Guided Multi-head Self-Attention (IG-MSA).

    Uses illumination representations to modulate attention scores,
    directing the model to focus on regions with different lighting.
    """

    def __init__(self, dim: int, num_heads: int = 4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5

        self.qkv = nn.Conv2d(dim, dim * 3, 1)
        self.illu_proj = nn.Conv2d(dim, dim, 1)
        self.proj = nn.Conv2d(dim, dim, 1)

    def forward(
        self, x: torch.Tensor, illu_feat: torch.Tensor
    ) -> torch.Tensor:
        b, c, h, w = x.shape

        qkv = self.qkv(x)
        qkv = rearrange(
            qkv, "b (three heads d) h w -> three b heads (h w) d",
            three=3, heads=self.num_heads,
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Illumination guidance: modulate keys with illumination features
        illu = self.illu_proj(illu_feat)
        illu = rearrange(
            illu, "b (heads d) h w -> b heads (h w) d",
            heads=self.num_heads,
        )
        k = k + illu  # additive illumination guidance

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = f.softmax(attn, dim=-1)

        out = torch.matmul(attn, v)
        out = rearrange(
            out, "b heads (h w) d -> b (heads d) h w",
            h=h, w=w,
        )
        return self.proj(out)


class IGTBlock(nn.Module):
    """Single Illumination-Guided Transformer block.

    IG-MSA -> LayerNorm -> FeedForward -> LayerNorm with skip connections.
    """

    def __init__(self, dim: int, num_heads: int = 4, ff_mult: int = 4):
        super().__init__()
        self.norm1 = LayerNorm2d(dim)
        self.attn = IlluminationGuidedMSA(dim, num_heads)
        self.norm2 = LayerNorm2d(dim)
        self.ff = FeedForward(dim, ff_mult)

    def forward(
        self, x: torch.Tensor, illu_feat: torch.Tensor
    ) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), illu_feat)
        x = x + self.ff(self.norm2(x))
        return x


class Retinexformer(nn.Module):
    """Retinexformer: One-stage Retinex-based Transformer for Low-light Enhancement.

    Architecture:
      1. Illumination Estimator: estimates illumination map and features
      2. Light-up: element-wise multiply input by illumination map
      3. Feature extraction: project lit image to embedding dim
      4. IGT blocks: illumination-guided transformer blocks
      5. Reconstruction: project back to RGB

    Reference: Cai et al., ICCV 2023.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        embed_dim: int = 32,
        num_blocks: int = 4,
        num_heads: int = 4,
    ):
        super().__init__()
        self.illumination_estimator = IlluminationEstimator(in_channels, embed_dim)

        # Feature extraction from lit image
        self.input_proj = nn.Conv2d(in_channels, embed_dim, 3, padding=1)

        # IGT blocks
        self.blocks = nn.ModuleList([
            IGTBlock(embed_dim, num_heads) for _ in range(num_blocks)
        ])

        # Reconstruction
        self.output_proj = nn.Conv2d(embed_dim, out_channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Step 1: Estimate illumination
        illu_map, illu_feat = self.illumination_estimator(x)

        # Step 2: Light up the input
        lit = x * illu_map + x  # residual light-up

        # Step 3: Feature extraction
        feat = self.input_proj(lit)

        # Step 4: IGT blocks with illumination guidance
        for block in self.blocks:
            feat = block(feat, illu_feat)

        # Step 5: Reconstruct enhanced image
        enhanced = self.output_proj(feat) + x  # global skip connection
        return enhanced


# ---------------------------------------------------------------------------
# SNR-Aware model
# ---------------------------------------------------------------------------


class SNRAwareBlock(nn.Module):
    """SNR-aware processing block.

    Dynamically switches between long-range (transformer) and short-range
    (convolution) operations based on local SNR estimation.
    """

    def __init__(self, dim: int, num_heads: int = 4):
        super().__init__()
        # SNR estimation branch
        self.snr_conv = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, 1, 1),
            nn.Sigmoid(),
        )

        # Long-range path (for low-SNR regions)
        self.norm_long = LayerNorm2d(dim)
        self.qkv = nn.Conv2d(dim, dim * 3, 1)
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.proj_long = nn.Conv2d(dim, dim, 1)

        # Short-range path (for high-SNR regions)
        self.conv_short = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, 3, padding=1),
        )

        self.norm_out = LayerNorm2d(dim)
        self.ff = FeedForward(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape

        # Estimate SNR map (0=low SNR, 1=high SNR)
        snr_map = self.snr_conv(x)  # (B, 1, H, W)

        # Long-range path
        x_norm = self.norm_long(x)
        qkv = self.qkv(x_norm)
        qkv = rearrange(
            qkv, "b (three heads d) h w -> three b heads (h w) d",
            three=3, heads=self.num_heads,
        )
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = f.softmax(torch.matmul(q, k.transpose(-2, -1)) * self.scale, dim=-1)
        long_out = torch.matmul(attn, v)
        long_out = rearrange(long_out, "b heads (h w) d -> b (heads d) h w", h=h, w=w)
        long_out = self.proj_long(long_out)

        # Short-range path
        short_out = self.conv_short(x)

        # SNR-guided fusion
        out = x + (1.0 - snr_map) * long_out + snr_map * short_out
        out = out + self.ff(self.norm_out(out))
        return out


class SNRAwareEnhancer(nn.Module):
    """SNR-Aware Low-Light Image Enhancement.

    Uses signal-to-noise ratio estimation to dynamically blend
    transformer (long-range) and CNN (short-range) processing.

    Reference: Xu et al., CVPR 2022.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        embed_dim: int = 32,
        num_blocks: int = 4,
        num_heads: int = 4,
    ):
        super().__init__()
        self.input_proj = nn.Conv2d(in_channels, embed_dim, 3, padding=1)
        self.blocks = nn.ModuleList([
            SNRAwareBlock(embed_dim, num_heads) for _ in range(num_blocks)
        ])
        self.output_proj = nn.Conv2d(embed_dim, out_channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.input_proj(x)
        for block in self.blocks:
            feat = block(feat)
        return self.output_proj(feat) + x  # global residual


# ---------------------------------------------------------------------------
# LLFormer (axis-based transformer)
# ---------------------------------------------------------------------------


class AxisAttention(nn.Module):
    """Axis-based multi-head attention for efficient processing.

    Applies attention along height and width axes separately,
    reducing complexity from O(N^2) to O(N*sqrt(N)).
    """

    def __init__(self, dim: int, num_heads: int = 4, axis: str = "h"):
        super().__init__()
        self.num_heads = num_heads
        self.axis = axis
        self.scale = (dim // num_heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape

        if self.axis == "h":
            # Attention along height for each column
            x_r = rearrange(x, "b c h w -> (b w) h c")
        else:
            # Attention along width for each row
            x_r = rearrange(x, "b c h w -> (b h) w c")

        qkv = self.qkv(x_r)
        qkv = rearrange(
            qkv, "batch seq (three heads d) -> three batch heads seq d",
            three=3, heads=self.num_heads,
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = f.softmax(torch.matmul(q, k.transpose(-2, -1)) * self.scale, dim=-1)
        out = torch.matmul(attn, v)
        out = rearrange(out, "batch heads seq d -> batch seq (heads d)")
        out = self.proj(out)

        if self.axis == "h":
            out = rearrange(out, "(b w) h c -> b c h w", b=b, w=w)
        else:
            out = rearrange(out, "(b h) w c -> b c h w", b=b, h=h)
        return out


class LLFormerBlock(nn.Module):
    """LLFormer block: axis-based attention (H then W) + FFN."""

    def __init__(self, dim: int, num_heads: int = 4):
        super().__init__()
        self.norm1 = LayerNorm2d(dim)
        self.attn_h = AxisAttention(dim, num_heads, axis="h")
        self.norm2 = LayerNorm2d(dim)
        self.attn_w = AxisAttention(dim, num_heads, axis="w")
        self.norm3 = LayerNorm2d(dim)
        self.ff = FeedForward(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn_h(self.norm1(x))
        x = x + self.attn_w(self.norm2(x))
        x = x + self.ff(self.norm3(x))
        return x


class LLFormer(nn.Module):
    """LLFormer: axis-based multi-head attention for low-light enhancement.

    Decomposes 2D attention into sequential height-axis and width-axis
    attention for computational efficiency on high-resolution images.

    Reference: Wang et al., AAAI 2023.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        embed_dim: int = 32,
        num_blocks: int = 4,
        num_heads: int = 4,
    ):
        super().__init__()
        self.input_proj = nn.Conv2d(in_channels, embed_dim, 3, padding=1)
        self.blocks = nn.ModuleList([
            LLFormerBlock(embed_dim, num_heads) for _ in range(num_blocks)
        ])
        self.output_proj = nn.Conv2d(embed_dim, out_channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.input_proj(x)
        for block in self.blocks:
            feat = block(feat)
        return self.output_proj(feat) + x


# ---------------------------------------------------------------------------
# Generic UNet baseline
# ---------------------------------------------------------------------------


class DoubleConv(nn.Module):
    """Two 3x3 conv layers with BatchNorm and ReLU."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNetEnhancer(nn.Module):
    """Generic UNet for image enhancement (ablation baseline).

    Standard encoder-decoder with skip connections.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        embed_dim: int = 32,
        **kwargs: Any,
    ):
        super().__init__()
        dims = [embed_dim, embed_dim * 2, embed_dim * 4, embed_dim * 8]

        # Encoder
        self.enc1 = DoubleConv(in_channels, dims[0])
        self.enc2 = DoubleConv(dims[0], dims[1])
        self.enc3 = DoubleConv(dims[1], dims[2])
        self.enc4 = DoubleConv(dims[2], dims[3])

        self.pool = nn.MaxPool2d(2)

        # Decoder
        self.up3 = nn.ConvTranspose2d(dims[3], dims[2], 2, stride=2)
        self.dec3 = DoubleConv(dims[3], dims[2])
        self.up2 = nn.ConvTranspose2d(dims[2], dims[1], 2, stride=2)
        self.dec2 = DoubleConv(dims[2], dims[1])
        self.up1 = nn.ConvTranspose2d(dims[1], dims[0], 2, stride=2)
        self.dec1 = DoubleConv(dims[1], dims[0])

        self.out_conv = nn.Conv2d(dims[0], out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        # Decoder with skip connections
        d3 = self.up3(e4)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))
        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        return self.out_conv(d1) + x  # global residual


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

MODEL_REGISTRY: dict[str, type[nn.Module]] = {
    "retinexformer": Retinexformer,
    "snr_aware": SNRAwareEnhancer,
    "llformer": LLFormer,
    "unet": UNetEnhancer,
}


def get_model(
    name: str,
    in_channels: int = 3,
    out_channels: int = 3,
    embed_dim: int = 32,
    num_blocks: int = 4,
    num_heads: int = 4,
    **kwargs: Any,
) -> nn.Module:
    """Create an enhancement model by name.

    Args:
        name: Model name (retinexformer, snr_aware, llformer, unet).
        in_channels: Input channels (default 3).
        out_channels: Output channels (default 3).
        embed_dim: Embedding dimension.
        num_blocks: Number of transformer/processing blocks.
        num_heads: Number of attention heads.

    Returns:
        Initialized model.

    Raises:
        ValueError: If model name is not in registry.
    """
    name_lower = name.lower().replace("-", "_")
    if name_lower not in MODEL_REGISTRY:
        available = ", ".join(sorted(MODEL_REGISTRY.keys()))
        raise ValueError(f"Unknown model '{name}'. Available: {available}")

    cls = MODEL_REGISTRY[name_lower]
    return cls(
        in_channels=in_channels,
        out_channels=out_channels,
        embed_dim=embed_dim,
        num_blocks=num_blocks,
        num_heads=num_heads,
    )


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
