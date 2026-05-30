"""Two-tower encoders for variant A.

AudioEncoder wraps MERT-v1-95M and mean-pools 75 Hz frames to a configurable
output rate (default 10 Hz). ImageEncoder wraps ViT-Base and processes the
strip as a sequence of 224x224 tiles with overlapping stride.

Pass lora_rank > 0 to inject LoRA adapters into attention layers instead of
fully freezing. Requires `peft` (pip install peft).
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


def _apply_lora(model: nn.Module, lora_rank: int, target_modules: list[str]) -> nn.Module:
    from peft import LoraConfig, get_peft_model
    cfg = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_rank * 2,
        target_modules=target_modules,
        lora_dropout=0.05,
        bias="none",
    )
    return get_peft_model(model, cfg)


class AudioEncoder(nn.Module):
    """MERT-v1-95M with mean-pooled output frames.

    Input:  audio waveform tensor (B, T_samples) at 24 kHz.
    Output: pooled frame embeddings (B, T_pooled, d_audio).

    freeze=True, lora_rank=0  → fully frozen (original v1 behaviour)
    freeze=False, lora_rank=0 → fully fine-tuned (expensive)
    freeze=True,  lora_rank>0 → LoRA adapters only (recommended)
    """

    # Attention projection names in MERT (Wav2Vec2-style transformer)
    LORA_TARGET_MODULES = ["q_proj", "v_proj"]

    def __init__(
        self,
        model_id: str = "m-a-p/MERT-v1-95M",
        native_frame_rate: int = 75,
        pool_hz: int = 10,
        freeze: bool = True,
        lora_rank: int = 0,
    ):
        super().__init__()
        backbone = AutoModel.from_pretrained(model_id, trust_remote_code=True)
        self.d_audio = backbone.config.hidden_size
        self.native_frame_rate = native_frame_rate
        self.pool_hz = pool_hz
        self.pool_kernel = max(1, round(native_frame_rate / pool_hz))

        if lora_rank > 0:
            # Freeze base weights; only LoRA adapters are trainable
            for p in backbone.parameters():
                p.requires_grad = False
            self.backbone = _apply_lora(backbone, lora_rank, self.LORA_TARGET_MODULES)
        elif freeze:
            for p in backbone.parameters():
                p.requires_grad = False
            backbone.eval()
            self.backbone = backbone
        else:
            self.backbone = backbone

        self._lora = lora_rank > 0

    def train(self, mode: bool = True):
        super().train(mode)
        # Keep backbone eval if nothing is trainable (no dropout/BN side effects)
        if not any(p.requires_grad for p in self.backbone.parameters()):
            self.backbone.eval()
        return self

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        with torch.set_grad_enabled(
            any(p.requires_grad for p in self.backbone.parameters())
        ):
            out = self.backbone(audio).last_hidden_state           # (B, T_native, d_audio)

        pooled = F.avg_pool1d(
            out.transpose(1, 2),
            kernel_size=self.pool_kernel,
            stride=self.pool_kernel,
            ceil_mode=False,
        ).transpose(1, 2)                                          # (B, T_pooled, d_audio)
        return pooled


class ImageEncoder(nn.Module):
    """ViT-Base applied to 224x224 tiles slid across the strip.

    Input:  image tensor (B, 3, 224, W). H is always 224.
    Output: tile embeddings (B, N_tiles, d_image) and a (B, N_tiles) validity mask.

    freeze=True, lora_rank=0  → fully frozen
    freeze=True,  lora_rank>0 → LoRA adapters only (recommended)
    """

    # Attention projection names in ViT (HuggingFace)
    LORA_TARGET_MODULES = ["query", "value"]

    def __init__(
        self,
        model_id: str = "google/vit-base-patch16-224-in21k",
        tile_size: int = 224,
        stride: int = 56,
        freeze: bool = True,
        lora_rank: int = 0,
    ):
        super().__init__()
        backbone = AutoModel.from_pretrained(model_id)
        self.d_image = backbone.config.hidden_size
        self.tile_size = tile_size
        self.stride = stride

        if lora_rank > 0:
            for p in backbone.parameters():
                p.requires_grad = False
            self.backbone = _apply_lora(backbone, lora_rank, self.LORA_TARGET_MODULES)
        elif freeze:
            for p in backbone.parameters():
                p.requires_grad = False
            backbone.eval()
            self.backbone = backbone
        else:
            self.backbone = backbone

        self._lora = lora_rank > 0

    def train(self, mode: bool = True):
        super().train(mode)
        if not any(p.requires_grad for p in self.backbone.parameters()):
            self.backbone.eval()
        return self

    @torch.no_grad()
    def _make_tile_mask(self, image_mask: torch.Tensor, n_tiles: int) -> torch.Tensor:
        B = image_mask.size(0)
        centres = (
            torch.arange(n_tiles, device=image_mask.device) * self.stride
            + self.tile_size // 2
        ).clamp(max=image_mask.size(1) - 1)
        return image_mask[:, centres]

    def forward(
        self,
        image: torch.Tensor,
        image_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if image.dtype == torch.uint8:
            image = image.float() / 255.0

        B, C, H, W = image.shape
        assert H == self.tile_size, f"image height {H} != tile_size {self.tile_size}"
        tiles = image.unfold(3, self.tile_size, self.stride)        # (B, C, H, N, tile)
        N = tiles.size(3)
        tiles = tiles.permute(0, 3, 1, 2, 4).contiguous()
        tiles = tiles.view(B * N, C, self.tile_size, self.tile_size)

        with torch.set_grad_enabled(
            any(p.requires_grad for p in self.backbone.parameters())
        ):
            out = self.backbone(pixel_values=tiles).last_hidden_state[:, 0]   # (B*N, d)

        feats = out.view(B, N, -1)

        if image_mask is None:
            tile_mask = torch.ones(B, N, dtype=torch.bool, device=image.device)
        else:
            tile_mask = self._make_tile_mask(image_mask.bool(), N)
        return feats, tile_mask
