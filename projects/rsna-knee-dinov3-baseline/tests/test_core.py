from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

from rsna_dinov3.config import ModelConfig
from rsna_dinov3.engine import macro_auc, masked_bce_loss
from rsna_dinov3.model import DinoStudyClassifier


class DummyBackbone(nn.Module):
    def __init__(self, hidden_size: int = 8) -> None:
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden_size)
        self.projection = nn.Linear(3, hidden_size)

    def forward(self, pixel_values: torch.Tensor):
        pooled = pixel_values.mean(dim=(-2, -1))
        token = self.projection(pooled)
        hidden = torch.stack([token, token * 0.5, token * 1.5], dim=1)
        return SimpleNamespace(last_hidden_state=hidden)


def test_masked_loss_ignores_unknown_targets():
    logits = torch.tensor([[0.0, 0.0]])
    targets = torch.tensor([[1.0, 0.0]])
    mask = torch.tensor([[True, False]])
    loss = masked_bce_loss(logits, targets, mask, torch.ones(2))
    assert torch.isclose(loss, torch.tensor(np.log(2.0), dtype=torch.float32))


def test_macro_auc_skips_single_class_columns():
    targets = np.array([[0.0, 1.0], [1.0, 1.0]])
    predictions = np.array([[0.1, 0.2], [0.9, 0.8]])
    score, per_label = macro_auc(targets, predictions)
    assert score == 1.0
    assert per_label[0] == 1.0
    assert np.isnan(per_label[1])


def test_study_model_ignores_padded_images():
    model = DinoStudyClassifier(ModelConfig(num_labels=12), backbone=DummyBackbone())
    model.eval()
    images = torch.rand(2, 4, 3, 16, 16)
    valid = torch.tensor([[True, True, False, False], [True, False, False, False]])
    plane_ids = torch.tensor([[0, 1, 3, 3], [2, 3, 3, 3]])
    metadata = torch.zeros(2, 4, 2)
    output = model(images, plane_ids, metadata, valid)
    assert output.shape == (2, 12)
    assert torch.isfinite(output).all()

