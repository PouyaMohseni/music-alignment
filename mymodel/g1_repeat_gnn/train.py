"""G1 -- train a small GCN to embed notes such that heuristic-flagged repeat
groups (D2's transposition-invariant pitch-interval n-gram matches) land
close together in embedding space, learned from local melodic context via
message passing over SEQUENTIAL edges only (repeat-group membership is used
purely as the contrastive training target, never as an input edge -- see
graph_data.py's docstring for why that split matters).

Trains ONLY on the TRAIN split (conservative choice: even though this is a
score-structure-only task with no audio/alignment-GT involved, keeping GNN
WEIGHT TRAINING strictly on train-split pieces avoids any ambiguity about
train/test boundaries for a later paper writeup). At decode time the trained
model is applied -- inference only, no weight updates -- to embed notes in
held-out val/test pieces.

Score-only, no audio, no GPU strictly required (graphs are tiny), but run
via the same sbatch/module pattern as everything else this session for
consistency.

    python -m mymodel.g1_repeat_gnn.train --epochs 30
"""
from __future__ import annotations
import argparse, json, random
from pathlib import Path
from itertools import combinations

import numpy as np
import torch
import torch.nn.functional as F

from mymodel.g1_repeat_gnn.graph_data import build_note_graph
from mymodel.g1_repeat_gnn.model import RepeatGCN


def _piece_contrastive_loss(embed: torch.Tensor, repeat_groups: list[list[int]],
                            n_negatives: int = 8, tau: float = 0.1,
                            max_pos_per_group: int = 6, rng: random.Random = None) -> torch.Tensor | None:
    """embed: (N, D) L2-normalized. repeat_groups: list of note-index groups
    (each a weak/noisy positive set). InfoNCE per (anchor, positive) pair
    against random negatives drawn from outside the anchor's own group.

    Fully vectorized: gathers ALL (anchor, positive, negatives) index tuples
    across every group first (cheap Python-level bookkeeping using rejection
    sampling -- O(n_negatives) expected per pair, not O(N) -- since a
    handful of pieces have 100+ repeat groups, building an O(N) exclusion
    list per PAIR was the actual bottleneck: a smoke test on just 25 pieces
    for 15 epochs took 40+ minutes of CPU before this fix), then does ONE
    batched matmul + ONE batched cross_entropy for the whole piece instead of
    hundreds/thousands of individual scalar tensor ops."""
    rng = rng or random
    N = embed.shape[0]
    if N < 3 or not repeat_groups:
        return None

    anchors, positives, neg_lists = [], [], []
    for group in repeat_groups:
        group_set = set(group)
        pairs = list(combinations(group, 2))
        if len(pairs) > max_pos_per_group:
            pairs = rng.sample(pairs, max_pos_per_group)
        for a, p in pairs:
            negs = []
            tries = 0
            while len(negs) < n_negatives and tries < n_negatives * 20:
                cand = rng.randrange(N)
                tries += 1
                if cand not in group_set:
                    negs.append(cand)
            if len(negs) < n_negatives:
                continue
            anchors.append(a)
            positives.append(p)
            neg_lists.append(negs)

    if not anchors:
        return None

    sim = embed @ embed.T  # (N, N) cosine similarity, single matmul for the whole piece
    a_t = torch.tensor(anchors, device=embed.device)
    p_t = torch.tensor(positives, device=embed.device)
    n_t = torch.tensor(neg_lists, device=embed.device)  # (M, n_negatives)

    pos_sim = sim[a_t, p_t].unsqueeze(1)                      # (M, 1)
    neg_sim = sim[a_t.unsqueeze(1).expand(-1, n_negatives), n_t]  # (M, n_negatives)
    logits = torch.cat([pos_sim, neg_sim], dim=1) / tau       # (M, 1+n_negatives)
    target = torch.zeros(len(anchors), dtype=torch.long, device=embed.device)
    return F.cross_entropy(logits, target)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--processed', default='data/MSMD/processed')
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--pieces_per_step', type=int, default=8)
    p.add_argument('--out_dir', default='/scratch/pmohseni/results/g1_repeat_gnn')
    p.add_argument('--device', default=None)
    p.add_argument('--seed', type=int, default=0)
    a = p.parse_args()

    device = a.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    rng = random.Random(a.seed)
    torch.manual_seed(a.seed)

    proc = Path(a.processed)
    train_ids = json.load(open(proc / 'splits.json'))['train']

    print(f'Building graphs for {len(train_ids)} train pieces...', flush=True)
    graphs = []
    n_no_repeats = 0
    for pid in train_ids:
        d = np.load(proc / pid / 'noteheads.npz')
        if len(d['midi_pitch']) < 6:
            continue
        feats, adj, groups = build_note_graph(d['onset_sec'], d['midi_pitch'],
                                              d.get('measure_idx'))
        if not groups:
            n_no_repeats += 1
            continue
        graphs.append((feats.to(device), adj.to(device), groups))
    print(f'Usable pieces (>=1 repeat group): {len(graphs)}/{len(train_ids)} '
          f'({n_no_repeats} had none)', flush=True)

    model = RepeatGCN().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)

    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, a.epochs + 1):
        rng.shuffle(graphs)
        epoch_losses = []
        for i in range(0, len(graphs), a.pieces_per_step):
            batch = graphs[i:i + a.pieces_per_step]
            opt.zero_grad()
            step_losses = []
            for feats, adj, groups in batch:
                embed = model(feats, adj)
                loss = _piece_contrastive_loss(embed, groups, rng=rng)
                if loss is not None:
                    step_losses.append(loss)
            if not step_losses:
                continue
            total = torch.stack(step_losses).mean()
            total.backward()
            opt.step()
            epoch_losses.append(total.item())
        mean_loss = float(np.mean(epoch_losses)) if epoch_losses else float('nan')
        print(f'epoch {epoch:3d}/{a.epochs}  loss={mean_loss:.4f}', flush=True)
        torch.save({'state_dict': model.state_dict(), 'epoch': epoch},
                  out_dir / 'best_model.pt')

    print('Training finished.', flush=True)


if __name__ == '__main__':
    main()
