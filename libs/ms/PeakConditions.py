from __future__ import annotations
import numpy as np
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .PeakSeries import PeakSeries

class PeakCondition(ABC):
    """
    Abstract base class for a condition to evaluate on PeakSeries.
    """

    @abstractmethod
    def evaluate(self, peaks: PeakSeries) -> bool:
        pass

    def __and__(self, other: PeakCondition) -> PeakCondition:
        return AndCondition(self, other)

    def __or__(self, other: PeakCondition) -> PeakCondition:
        return OrCondition(self, other)

    def __invert__(self) -> PeakCondition:
        return NotCondition(self)
    
class AndCondition(PeakCondition):
    """
    Condition that evaluates to True if both conditions are True.

    Example:
        combined = cond1 & cond2  # both conditions must be satisfied
    """
    def __init__(self, cond1: PeakCondition, cond2: PeakCondition):
        self.cond1 = cond1
        self.cond2 = cond2

    def evaluate(self, peaks: PeakSeries) -> bool:
        return self.cond1.evaluate(peaks) and self.cond2.evaluate(peaks)


class OrCondition(PeakCondition):
    """
    Condition that evaluates to True if either condition is True.

    Example:
        either = cond1 | cond2  # at least one condition must be satisfied
    """
    def __init__(self, cond1: PeakCondition, cond2: PeakCondition):
        self.cond1 = cond1
        self.cond2 = cond2

    def evaluate(self, peaks: PeakSeries) -> bool:
        return self.cond1.evaluate(peaks) or self.cond2.evaluate(peaks)


class NotCondition(PeakCondition):
    """
    Condition that evaluates to True if the inner condition is False.

    Example:
        negated = ~cond1  # the condition is NOT satisfied
    """
    def __init__(self, cond: PeakCondition):
        self.cond = cond

    def evaluate(self, peaks: PeakSeries) -> bool:
        return not self.cond.evaluate(peaks)


class ExistsMz(PeakCondition):
    def __init__(self, target_mz: float, mz_tol: float, min_intensity: float = 0.0):
        self.target_mz = target_mz
        self.mz_tol = mz_tol
        self.min_intensity = min_intensity

    def evaluate(self, peaks: PeakSeries) -> bool:
        mz_intensities = peaks.np
        mz_array = mz_intensities[:, 0]
        intensity_array = mz_intensities[:, 1]
        mask = (np.abs(mz_array - self.target_mz) <= self.mz_tol) & (intensity_array >= self.min_intensity)
        return np.any(mask)

