"""
v12 model variants sharing the same encoders.

v12b: MERT(frozen) + BiLSTM(768->1024) + AlignmentHead
v12c: MERT(LoRA r=8) + BiLSTM + AlignmentHead
v12d: MERT(LoRA r=8) + BiLSTM + CrossAttention(audio->score) + AlignmentHead

Score encoder is ResNet18 frozen in all variants.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm
from transformers import AutoModel
from peft import get_peft_model, LoraConfig, TaskType


# ── Encoders ──────────────────────────────────────────────────────────────────

class MERTEncoderFrozen(nn.Module):
    def __init__(self):
        super().__init__()
        self.mert = AutoModel.from_pretrained(
            'm-a-p/MERT-v1-95M', trust_remote_code=True)
        for p in self.mert.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def forward(self, wav):
        out = self.mert(wav.unsqueeze(0), output_hidden_states=True)
        hidden = torch.stack(out.hidden_states, dim=0)   # (13,1,T,768)
        return hidden.mean(0).squeeze(0)                 # (T, 768)


class MERTEncoderLoRA(nn.Module):
    def __init__(self, lora_rank: int = 8):
        super().__init__()
        base = AutoModel.from_pretrained(
            'm-a-p/MERT-v1-95M', trust_remote_code=True)
        cfg = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_rank * 2,
            target_modules=['q_proj', 'v_proj'],
            lora_dropout=0.05,
            bias='none',
        )
        self.mert = get_peft_model(base, cfg)

    def forward(self, wav):
        out = self.mert(wav.unsqueeze(0), output_hidden_states=True)
        hidden = torch.stack(out.hidden_states, dim=0)
        return hidden.mean(0).squeeze(0)                 # (T, 768)


class ResNetScoreEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = tvm.resnet18(weights=tvm.ResNet18_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        for p in self.backbone.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def forward(self, cols):
        feats = self.backbone(cols)
        return feats.squeeze(-1).squeeze(-1)             # (N, 512)


# ── Audio temporal head (BiLSTM) ──────────────────────────────────────────────

class AudioLSTM(nn.Module):
    def __init__(self, input_dim=768, hidden=512, layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden, layers,
                            batch_first=True, bidirectional=True)

    def forward(self, x):
        """x: (T, 768) -> (T, 1024)"""
        out, _ = self.lstm(x.unsqueeze(0))   # (1, T, 1024)
        return out.squeeze(0)                 # (T, 1024)


# ── Cross-attention alignment (v12d) ──────────────────────────────────────────

class CrossAttentionAlign(nn.Module):
    """Audio frames attend to score columns, refining the similarity signal."""
    def __init__(self, dim=256, heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True,
                                          dropout=0.1)
        self.norm = nn.LayerNorm(dim)

    def forward(self, audio_emb, score_emb):
        """
        audio_emb: (T, 256)
        score_emb: (N, 256)
        Returns refined: (T, 256)
        """
        q = audio_emb.unsqueeze(0)    # (1, T, 256)
        kv = score_emb.unsqueeze(0)   # (1, N, 256)
        out, _ = self.attn(q, kv, kv)
        return self.norm(audio_emb + out.squeeze(0))   # residual


# ── Shared projection head ─────────────────────────────────────────────────────

class ProjectionHead(nn.Module):
    def __init__(self, audio_in, score_in=512, dim=256):
        super().__init__()
        self.audio_proj = nn.Sequential(nn.Linear(audio_in, dim), nn.LayerNorm(dim))
        self.score_proj = nn.Sequential(nn.Linear(score_in, dim), nn.LayerNorm(dim))

    def forward(self, audio_feats, score_feats):
        a = F.normalize(self.audio_proj(audio_feats), dim=-1)
        s = F.normalize(self.score_proj(score_feats), dim=-1)
        return a @ s.T


# ── Full variant models ────────────────────────────────────────────────────────

class V12b(nn.Module):
    """MERT frozen + BiLSTM + projection."""
    def __init__(self, embed_dim=256, lstm_hidden=512, lstm_layers=2):
        super().__init__()
        self.audio_enc  = MERTEncoderFrozen()
        self.score_enc  = ResNetScoreEncoder()
        self.lstm       = AudioLSTM(768, lstm_hidden, lstm_layers)
        self.head       = ProjectionHead(lstm_hidden * 2, 512, embed_dim)

    def encode_audio(self, wav):
        with torch.no_grad():
            feats = self.audio_enc(wav)      # (T, 768)
        return self.lstm(feats)              # (T, 1024)

    def encode_score(self, cols, batch_sz=64):
        chunks = cols.split(batch_sz)
        return torch.cat([self.score_enc(c) for c in chunks])

    def forward(self, wav, cols, col_batch=64):
        af = self.encode_audio(wav)
        sf = self.encode_score(cols, col_batch)
        return self.head(af, sf)


class V12c(nn.Module):
    """MERT LoRA + BiLSTM + projection."""
    def __init__(self, embed_dim=256, lora_rank=8, lstm_hidden=512, lstm_layers=2):
        super().__init__()
        self.audio_enc  = MERTEncoderLoRA(lora_rank)
        self.score_enc  = ResNetScoreEncoder()
        self.lstm       = AudioLSTM(768, lstm_hidden, lstm_layers)
        self.head       = ProjectionHead(lstm_hidden * 2, 512, embed_dim)

    def encode_audio(self, wav):
        feats = self.audio_enc(wav)          # (T, 768) — gradients flow through LoRA
        return self.lstm(feats)              # (T, 1024)

    def encode_score(self, cols, batch_sz=64):
        chunks = cols.split(batch_sz)
        return torch.cat([self.score_enc(c) for c in chunks])

    def forward(self, wav, cols, col_batch=64):
        af = self.encode_audio(wav)
        sf = self.encode_score(cols, col_batch)
        return self.head(af, sf)


class V12d(nn.Module):
    """MERT LoRA + BiLSTM + cross-attention + projection."""
    def __init__(self, embed_dim=256, lora_rank=8, lstm_hidden=512,
                 lstm_layers=2, attn_heads=4):
        super().__init__()
        self.audio_enc  = MERTEncoderLoRA(lora_rank)
        self.score_enc  = ResNetScoreEncoder()
        self.lstm       = AudioLSTM(768, lstm_hidden, lstm_layers)
        self.head       = ProjectionHead(lstm_hidden * 2, 512, embed_dim)
        self.cross_attn = CrossAttentionAlign(embed_dim, attn_heads)

    def encode_audio(self, wav):
        feats = self.audio_enc(wav)
        return self.lstm(feats)

    def encode_score(self, cols, batch_sz=64):
        chunks = cols.split(batch_sz)
        return torch.cat([self.score_enc(c) for c in chunks])

    def forward(self, wav, cols, col_batch=64):
        af = self.encode_audio(wav)
        sf = self.encode_score(cols, col_batch)

        # Project both to shared space
        a = F.normalize(self.head.audio_proj(af), dim=-1)   # (T, 256)
        s = F.normalize(self.head.score_proj(sf), dim=-1)   # (N, 256)

        # Cross-attention: audio queries attend to score
        a_refined = self.cross_attn(a, s)                   # (T, 256)

        return a_refined @ s.T                               # (T, N)
