from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List
from dataclasses import dataclass

import numpy as np


class SignalModel(ABC):
    @abstractmethod
    def fit(self, training_data: np.array, training_labels: np.array) -> SignalModel:
        """
        Train the model using the provided data and labels.
        Return self for convenience.
        """
        raise NotImplementedError

    @abstractmethod
    def evaluate(self, test_data: np.array, test_labels: np.array) -> ModelEvaluationReport:
        """Compute model performance characteristics on the provided test data and labels."""
        raise NotImplementedError

    @abstractmethod
    def predict(self, bio_data: np.array, presented_symbols: List[str], all_symbols: List[str]) -> np.array:
        """Using the provided data, compute log likelihoods over the entire symbol set."""
        raise NotImplementedError

    @abstractmethod
    def save(self, checkpoint_path: Path) -> None:
        """Save model state to the provided checkpoint"""
        raise NotImplementedError

    @abstractmethod
    def load(self, checkpoint_path: Path) -> None:
        """Restore model state from the provided checkpoint"""
        raise NotImplementedError

    @abstractmethod
    def __eq__(self, other) -> bool:
        raise NotImplementedError


@dataclass
class ModelEvaluationReport:
    """
    Describes model performance characteristics.

    TODO - add more attributes as needed. Note that the model
    may not know about how its outputs are applied to compute a decision rule,
    so the model alone cannot compute AUC unless `model.evaluate` is also invoked
    with a callback or something. (This would get a bit finnicky).
    """

    top_1_acc: float
