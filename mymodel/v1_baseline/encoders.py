"""Two-tower encoders for variant A.

AudioEncoder wraps MERT-v1-95M and mean-pools 75 Hz frames to a configurable
output rate (default 10 Hz). ImageEncoder wraps ViT-Base and processes the
strip as a sequence of 224x224 tiles with overlapping stride.

Both encoders default to frozen for variant A.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


class AudioEncoder(nn.Module):
    """MERT-v1-95M with mean-pooled output frames.

    Input:  audio waveform tensor (B, T_samples) at 24 kHz.
    Output: pooled frame embeddings (B, T_pooled, d_audio).
    """

    def __init__(
        self,
        model_id: str = "m-a-p/MERT-v1-95M",
        native_frame_rate: int = 75,
        pool_hz: int = 10,
        freeze: bool = True,
    ):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_id, trust_remote_code=True)
        self.d_audio = self.backbone.config.hidden_size
        self.native_frame_rate = native_frame_rate
        self.pool_hz = pool_hz
        # pool_kernel rounds to nearest int; exact only when 75 % pool_hz == 0.
        # For pool_hz=15, kernel=5; pool_hz=10, kernel=8 (75/10=7.5 → round up).
        self.pool_kernel = max(1, round(native_frame_rate / pool_hz))

        if freeze:
            for p in self.backbone.parameters():
                p.requires_grad = False
            self.backbone.eval()

    def train(self, mode: bool = True):
        # keep backbone in eval if frozen (no dropout, no BN updates)
        super().train(mode)
        if not any(p.requires_grad for p in self.backbone.parameters()):
            self.backbone.eval()
        return self

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        # audio: (B, T_samples)  float32 in [-1, 1]
        with torch.set_grad_enabled(
            any(p.requires_grad for p in self.backbone.parameters())
        ):
            out = self.backbone(audio).last_hidden_state           # (B, T_native, d_audio)

        # mean pool along the time axis with non-overlapping kernel.
        # avg_pool1d expects (B, C, T) so we transpose, pool, then transpose back.
        pooled = F.avg_pool1d(
            out.transpose(1, 2),
            kernel_size=self.pool_kernel,
            stride=self.pool_kernel,
            ceil_mode=False,
        ).transpose(1, 2)                                          # (B, T_pooled, d_audio)
        return pooled


class ImageEncoder(nn.Module):
    """Frozen ViT-Base applied to 224x224 tiles slid across the strip.

    Input:  image tensor (B, 3, 224, W). H is always 224 (strip builder pads it).
    Output: tile embeddings (B, N_tiles, d_image) and a (B, N_tiles) validity mask.

    If `image_mask` is provided (B, W) marking valid strip pixels, the output
    mask flags tiles whose centre column is invalid (i.e. fully inside padding).
    """

    def __init__(
        self,
        model_id: str = "google/vit-base-patch16-224-in21k",
        tile_size: int = 224,
        stride: int = 56,
        freeze: bool = True,
    ):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_id)
        self.d_image = self.backbone.config.hidden_size
        self.tile_size = tile_size
        self.stride = stride

        if freeze:
            for p in self.backbone.parameters():
                p.requires_grad = False
            self.backbone.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        if not any(p.requires_grad for p in self.backbone.parameters()):
            self.backbone.eval()
        return self

    @torch.no_grad()
    def _make_tile_mask(self, image_mask: torch.Tensor, n_tiles: int) -> torch.Tensor:
        # image_mask: (B, W) bool; tile is "valid" if its centre column is valid.
        B = image_mask.size(0)
        centres = (
            torch.arange(n_tiles, device=image_mask.device) * self.stride
            + self.tile_size // 2
        ).clamp(max=image_mask.size(1) - 1)                        # (N_tiles,)
        return image_mask[:, centres]                              # (B, N_tiles)

    def forward(
        self,
        image: torch.Tensor,
        image_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # image: (B, 3, 224, W)  uint8 or float
        if image.dtype == torch.uint8:
            image = image.float() / 255.0

        B, C, H, W = image.shape
        assert H == self.tile_size, f"image height {H} != tile_size {self.tile_size}"
        # unfold over width dimension into tiles
        tiles = image.unfold(3, self.tile_size, self.stride)        # (B, C, H, N, tile)
        N = tiles.size(3)
        tiles = tiles.permute(0, 3, 1, 2, 4).contiguous()           # (B, N, C, H, tile)
        tiles = tiles.view(B * N, C, self.tile_size, self.tile_size)

        with torch.set_grad_enabled(
            any(p.requires_grad for p in self.backbone.parameters())
        ):
            out = self.backbone(pixel_values=tiles).last_hidden_state[:, 0]   # (B*N, d_image)

        feats = out.view(B, N, -1)                                   # (B, N, d_image)

        if image_mask is None:
            tile_mask = torch.ones(B, N, dtype=torch.bool, device=image.device)
        else:
            tile_mask = self._make_tile_mask(image_mask.bool(), N)
        return feats, tile_mask
