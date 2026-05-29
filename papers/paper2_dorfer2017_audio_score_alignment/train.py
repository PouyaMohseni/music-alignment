"""
Training — Dorfer et al. 2017.
Reference: paper Section 4.
"""
import torch
from model.audio_encoder import AudioEncoder
from model.score_encoder import ScoreEncoder
from model.loss import pairwise_ranking_loss


def train():
    audio_enc = AudioEncoder()
    score_enc = ScoreEncoder()
    optimizer = torch.optim.SGD(
        list(audio_enc.parameters()) + list(score_enc.parameters()),
        lr=0.01, momentum=0.9, weight_decay=1e-4
    )
    # TODO: build MSMD dataloader, implement training loop
    raise NotImplementedError("Implement training loop")


if __name__ == "__main__":
    train()
