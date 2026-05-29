"""
Training — Henkel et al. 2019 RL score following.
Algorithm: REINFORCE (policy gradient).
Reference: paper Section 4.
"""
import torch
from model.audio_encoder import AudioEncoder
from model.score_encoder import ScoreEncoder
from model.policy import PolicyNetwork


def train():
    audio_enc = AudioEncoder()
    score_enc = ScoreEncoder()
    policy = PolicyNetwork()
    optimizer = torch.optim.Adam(
        list(audio_enc.parameters()) +
        list(score_enc.parameters()) +
        list(policy.parameters()),
        lr=1e-4
    )
    # TODO: build environment (score position state), episode loop, REINFORCE update
    raise NotImplementedError("Implement RL training loop")


if __name__ == "__main__":
    train()
