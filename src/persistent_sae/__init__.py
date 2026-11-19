"""Persistent sparse autoencoders: learning feature timescales in language models."""

from .model import PersistentSAE, convert_legacy_state_dict
from .trainer import PersistentSAETrainer

__all__ = ["PersistentSAE", "PersistentSAETrainer", "convert_legacy_state_dict"]
__version__ = "0.1.0"
