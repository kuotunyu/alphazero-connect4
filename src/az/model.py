"""Policy+value ResNet and the torch-backed batch evaluator.

The network returns RAW policy logits; legal-move masking (with -1e9, not
-inf, to avoid NaNs) happens in the evaluator and in the training loss so
that training and inference see the same distribution.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from az import game
from az.mcts import Evaluator

MASK_VALUE = -1e9


class ResBlock(nn.Module):
    def __init__(self, filters: int):
        super().__init__()
        self.conv1 = nn.Conv2d(filters, filters, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(filters)
        self.conv2 = nn.Conv2d(filters, filters, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(filters)

    def forward(self, x):
        y = torch.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        return torch.relu(x + y)


class PolicyValueNet(nn.Module):
    def __init__(self, blocks: int = 6, filters: int = 96):
        super().__init__()
        self.blocks = blocks
        self.filters = filters
        self.stem = nn.Sequential(
            nn.Conv2d(3, filters, 3, padding=1, bias=False),
            nn.BatchNorm2d(filters),
            nn.ReLU(inplace=True),
        )
        self.tower = nn.Sequential(*[ResBlock(filters) for _ in range(blocks)])
        flat = 32 * game.ROWS * game.COLS
        self.policy_head = nn.Sequential(
            nn.Conv2d(filters, 32, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(flat, game.COLS),
        )
        self.value_head = nn.Sequential(
            nn.Conv2d(filters, 32, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(flat, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Tanh(),
        )

    def forward(self, x):
        h = self.tower(self.stem(x))
        return self.policy_head(h), self.value_head(h).squeeze(-1)

    def config(self) -> dict:
        return {"blocks": self.blocks, "filters": self.filters}


def create_model(config: dict) -> PolicyValueNet:
    return PolicyValueNet(blocks=config["blocks"], filters=config["filters"])


class NetEvaluator(Evaluator):
    """Batches states into one forward pass; one H2D/D2H round-trip per call.

    Values are the value head's output: the expected outcome for the player
    to move, matching the search's perspective convention.
    """

    def __init__(self, model: PolicyValueNet, device=None, autocast: bool = False):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device).eval()
        self.autocast = autocast and self.device == "cuda"

    def evaluate_batch(self, states):
        planes = np.stack([game.encode(s) for s in states])
        legal = np.stack([game.legal_mask(s) for s in states])
        x = torch.from_numpy(planes).to(self.device)
        with torch.inference_mode():
            if self.autocast:
                with torch.autocast("cuda", dtype=torch.float16):
                    logits, values = self.model(x)
                logits, values = logits.float(), values.float()
            else:
                logits, values = self.model(x)
            logits = logits.masked_fill(
                ~torch.from_numpy(legal).to(self.device), MASK_VALUE
            )
            priors = torch.softmax(logits, dim=1)
        return priors.cpu().numpy(), values.cpu().numpy()
