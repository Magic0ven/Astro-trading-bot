"""
Score History — Rolling buffer for composite scores.
Used to compute the Medium (rolling mean) and the Slope (delta or regression).
"""
from collections import deque
from typing import Optional
import numpy as np


class ScoreHistory:
    """
    Ring buffer that stores the last N composite scores.
    Computes Medium (mean) and Slope (linear regression slope or simple delta).
    """

    def __init__(self, window: int = 5):
        self.window = window
        self._western: deque = deque(maxlen=window)
        self._vedic: deque = deque(maxlen=window)

    def push(self, western_score: float, vedic_score: float):
        self._western.append(western_score)
        self._vedic.append(vedic_score)

    def medium(self, system: str = "western") -> Optional[float]:
        buf = self._western if system == "western" else self._vedic
        if not buf:
            return None
        return float(np.mean(buf))

    def slope(self, system: str = "western") -> Optional[float]:
        """
        Compute slope of the score series.
        With >=3 points: uses linear regression slope (more robust).
        With 2 points:   simple delta (current - previous).
        With <2 points:  None (not enough data).
        """
        buf = list(self._western if system == "western" else self._vedic)
        if len(buf) < 2:
            return None
        if len(buf) >= 3:
            x = np.arange(len(buf), dtype=float)
            coeffs = np.polyfit(x, buf, 1)
            return float(coeffs[0])  # slope of best-fit line
        return float(buf[-1] - buf[-2])

    def latest(self, system: str = "western") -> Optional[float]:
        buf = self._western if system == "western" else self._vedic
        return buf[-1] if buf else None

    def is_ready(self) -> bool:
        """True once we have at least 2 data points in both buffers."""
        return len(self._western) >= 2 and len(self._vedic) >= 2

    def summary(self) -> dict:
        return {
            "western": {
                "latest": self.latest("western"),
                "medium": self.medium("western"),
                "slope": self.slope("western"),
                "history": list(self._western),
            },
            "vedic": {
                "latest": self.latest("vedic"),
                "medium": self.medium("vedic"),
                "slope": self.slope("vedic"),
                "history": list(self._vedic),
            },
            "ready": self.is_ready(),
        }
