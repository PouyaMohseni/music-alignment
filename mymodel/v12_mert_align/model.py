"""
v12 MERT-Align model.

Audio: MERT-v1-95M (frozen) -> 768-dim at 75Hz -> Linear(768, 256) + LN -> L2Norm
Score: ResNet18 (frozen) on 80px columns -> 512-dim -> Linear(512, 256) + LN -> L2Norm
Alignment: cosine similarity matrix -> DTW decode at inference
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm
from transformers import AutoModel


class MERTEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.mert = AutoModel.from_pretrained(
            'm-a-p/MERT-v1-95M', trust_remote_code=True)
        for p in self.mert.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        """wav: (T_samples,)  ->  (T_frames, 768)"""
        out = self.mert(wav.unsqueeze(0), output_hidden_states=True)
        # Average all 13 transformer layers (incl. embedding layer) — standard SUPERB protocol
        hidden = torch.stack(out.hidden_states, dim=0)  # (13, 1, T, 768)
        return hidden.mean(dim=0).squeeze(0)            # (T, 768)


class ResNetScoreEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = tvm.resnet18(weights=tvm.ResNet18_Weights.IMAGENET1K_V1)
        # Drop the final FC; keep up to avgpool -> (B, 512, 1, 1)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        for p in self.backbone.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def forward(self, cols: torch.Tensor) -> torch.Tensor:
        """cols: (N_cols, 3, 224, 224) -> (N_cols, 512)"""
        feats = self.backbone(cols)        # (N_cols, 512, 1, 1)
        return feats.squeeze(-1).squeeze(-1)  # (N_cols, 512)


class AlignmentHead(nn.Module):
    def __init__(self, audio_dim=768, score_dim=512, embed_dim=256):
        super().__init__()
        self.audio_proj = nn.Sequential(
            nn.Linear(audio_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )
        self.score_proj = nn.Sequential(
            nn.Linear(score_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )
        self.temperature = nn.Parameter(torch.tensor(0.07))

    def forward(self, audio_feats, score_feats):
        """
        audio_feats: (T, 768)
        score_feats: (N, 512)
        Returns sim: (T, N)
        """
        a = F.normalize(self.audio_proj(audio_feats), dim=-1)   # (T, 256)
        s = F.normalize(self.score_proj(score_feats), dim=-1)   # (N, 256)
        return a @ s.T                                           # (T, N)


class MERTAlignModel(nn.Module):
    def __init__(self, embed_dim: int = 256):
        super().__init__()
        self.audio_enc = MERTEncoder()
        self.score_enc = ResNetScoreEncoder()
        self.head = AlignmentHead(768, 512, embed_dim)

    def encode_audio(self, wav: torch.Tensor) -> torch.Tensor:
        """wav: (T_samples,)  ->  (T_frames, 768)"""
        return self.audio_enc(wav)

    def encode_score(self, cols: torch.Tensor, batch_sz: int = 64) -> torch.Tensor:
        """cols: (N_cols, 3, 224, 224)  ->  (N_cols, 512)"""
        chunks = cols.split(batch_sz)
        return torch.cat([self.score_enc(c) for c in chunks])

    def similarity(self, audio_feats, score_feats):
        return self.head(audio_feats, score_feats)

    def forward(self, wav, cols, col_batch=64):
        af = self.encode_audio(wav)
        sf_ = self.encode_score(cols, col_batch)
        return self.similarity(af, sf_)
