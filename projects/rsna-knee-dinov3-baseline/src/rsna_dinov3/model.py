from __future__ import annotations

import torch
from torch import nn
from transformers import AutoModel

from .config import ModelConfig


class GatedAttentionPool(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        attention_size = max(hidden_size // 2, 64)
        self.tanh = nn.Linear(hidden_size, attention_size)
        self.sigmoid = nn.Linear(hidden_size, attention_size)
        self.score = nn.Linear(attention_size, 1)

    def forward(self, features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        scores = self.score(torch.tanh(self.tanh(features)) * torch.sigmoid(self.sigmoid(features)))
        scores = scores.squeeze(-1).masked_fill(~mask, torch.finfo(features.dtype).min)
        weights = scores.softmax(dim=1)
        return torch.sum(features * weights.unsqueeze(-1), dim=1)


class DinoStudyClassifier(nn.Module):
    """Frozen 2D DINOv3 encoder plus metadata-aware multiple-instance pooling."""

    def __init__(self, config: ModelConfig, backbone: nn.Module | None = None) -> None:
        super().__init__()
        self.config = config
        self.backbone = backbone or AutoModel.from_pretrained(config.model_path)
        backbone_config = self.backbone.config
        hidden_size = getattr(backbone_config, "hidden_size", None)
        if hidden_size is None:
            hidden_sizes = getattr(backbone_config, "hidden_sizes", None)
            hidden_size = hidden_sizes[-1] if hidden_sizes else None
        if hidden_size is None:
            raise ValueError("Could not determine DINOv3 backbone hidden size")

        for parameter in self.backbone.parameters():
            parameter.requires_grad = False
        self.backbone.eval()

        self.slice_projection = nn.Sequential(
            nn.LayerNorm(2 * hidden_size),
            nn.Linear(2 * hidden_size, hidden_size),
            nn.GELU(),
        )
        self.plane_embedding = nn.Embedding(4, hidden_size)
        self.series_projection = nn.Sequential(
            nn.Linear(2, hidden_size), nn.GELU(), nn.Linear(hidden_size, hidden_size)
        )
        self.pool = GatedAttentionPool(hidden_size)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Dropout(config.dropout),
            nn.Linear(hidden_size, config.num_labels),
        )

    def train(self, mode: bool = True) -> "DinoStudyClassifier":
        super().train(mode)
        self.backbone.eval()
        return self

    def _encode_valid(self, images: torch.Tensor) -> torch.Tensor:
        # The backbone is intentionally frozen for a strong, inexpensive first baseline.
        with torch.no_grad():
            output = self.backbone(pixel_values=images)
        hidden = output.last_hidden_state
        if hidden.ndim == 3:  # ViT: CLS token + patch tokens (and possibly registers).
            cls = hidden[:, 0]
            register_count = int(getattr(self.backbone.config, "num_register_tokens", 0))
            patch_mean = hidden[:, 1 + register_count :].mean(dim=1)
        elif hidden.ndim == 4:  # ConvNeXt compatibility.
            patch_mean = hidden.mean(dim=(-2, -1))
            cls = getattr(output, "pooler_output", patch_mean)
        else:
            raise ValueError(f"Unexpected backbone output shape: {tuple(hidden.shape)}")
        return self.slice_projection(torch.cat([cls, patch_mean], dim=-1))

    def forward(
        self,
        images: torch.Tensor,
        plane_ids: torch.Tensor,
        series_meta: torch.Tensor,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, image_count = images.shape[:2]
        flat_images = images.flatten(0, 1)
        flat_valid = valid.flatten()
        valid_features = self._encode_valid(flat_images[flat_valid])
        hidden_size = valid_features.shape[-1]
        features = valid_features.new_zeros(batch_size * image_count, hidden_size)
        features[flat_valid] = valid_features
        features = features.view(batch_size, image_count, hidden_size)
        features = (
            features
            + self.plane_embedding(plane_ids)
            + self.series_projection(series_meta)
        )
        features = features.masked_fill(~valid.unsqueeze(-1), 0.0)
        return self.head(self.pool(features, valid))

    def trainable_state_dict(self) -> dict[str, torch.Tensor]:
        return {
            key: value
            for key, value in self.state_dict().items()
            if not key.startswith("backbone.")
        }

    def load_trainable_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        result = self.load_state_dict(state, strict=False)
        missing = [key for key in result.missing_keys if not key.startswith("backbone.")]
        if missing or result.unexpected_keys:
            raise RuntimeError(
                f"Invalid head checkpoint; missing={missing}, unexpected={result.unexpected_keys}"
            )
