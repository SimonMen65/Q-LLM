import torch
import torch.nn as nn
import torch.nn.functional as F

class MLPChunkScorer(nn.Module):
    def __init__(self, dim, hidden_dim=128):
        super().__init__()
        self.scorer = nn.Sequential(
            nn.Linear(dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, query: torch.Tensor, block_k: torch.Tensor):
        """
        query: (num_units, dim)
        block_k: (num_units, num_blocks, dim)
        """
        num_units, num_blocks, dim = block_k.shape
        query_expanded = query.unsqueeze(1).expand(-1, num_blocks, -1)  # (U, B, D)
        pairwise = torch.cat([query_expanded, block_k], dim=-1)  # (U, B, 2D)
        score = self.scorer(pairwise).squeeze(-1)  # (U, B)
        return score