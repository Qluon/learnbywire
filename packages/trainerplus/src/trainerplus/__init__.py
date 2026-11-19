"""TrainerPlus: open-harness candidate, optional proprietary control integration."""
from .config import DataConfig, TrainingConfig
from .data import TokenDataset, prepare_dataset
from .trainer import TrainerPlus
from .artifacts import TrainResult, VERSION as __version__

__all__ = ["TrainerPlus", "TrainingConfig", "DataConfig", "TokenDataset", "TrainResult", "prepare_dataset"]
