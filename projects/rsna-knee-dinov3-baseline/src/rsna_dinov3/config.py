from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class ModelConfig:
    model_path: str = "facebook/dinov3-vits16-pretrain-lvd1689m"
    num_labels: int = 12
    dropout: float = 0.2

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class DataConfig:
    image_size: int = 224
    max_series: int = 6
    slices_per_series: int = 4
    percentile_low: float = 0.5
    percentile_high: float = 99.5

    @property
    def max_images(self) -> int:
        return self.max_series * self.slices_per_series

    def to_dict(self) -> dict:
        return asdict(self)

