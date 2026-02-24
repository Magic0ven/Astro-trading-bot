"""
Score History — Rolling buffer for composite scores.
Used to compute the Medium (rolling mean) and the Slope (delta or regression).

Persistence: save() / load() write the buffer to a JSON file in logs/ so
history survives bot restarts. Without this, the bot would need to run for
SCORE_HISTORY_WINDOW × CHECK_INTERVAL_MINUTES before generating its first
signal after every restart.
"""
import json
from collections import deque
from pathlib import Path
from typing import Optional

import numpy as np


class ScoreHistory:
    """
    Ring buffer that stores the last N composite scores.
    Computes Medium (mean) and Slope (linear regression slope or simple delta).
    """

    def __init__(self, window: int = 5, persist_path: Optional[Path] = None):
        self.window = window
        self._western: deque = deque(maxlen=window)
        self._vedic:   deque = deque(maxlen=window)
        self._path = persist_path
        if self._path:
            self.load()

    # ── Mutation ──────────────────────────────────────────────────────────────

    def push(self, western_score: float, vedic_score: float):
        self._western.append(western_score)
        self._vedic.append(vedic_score)
        if self._path:
            self.save()

    # ── Queries ───────────────────────────────────────────────────────────────

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
            return float(np.polyfit(x, buf, 1)[0])
        return float(buf[-1] - buf[-2])

    def latest(self, system: str = "western") -> Optional[float]:
        buf = self._western if system == "western" else self._vedic
        return buf[-1] if buf else None

    def is_ready(self) -> bool:
        """True once we have at least 2 data points in both buffers."""
        return len(self._western) >= 2 and len(self._vedic) >= 2

    def bars_collected(self) -> int:
        """Number of bars pushed so far (up to window size)."""
        return len(self._western)

    def summary(self) -> dict:
        return {
            "western": {
                "latest":  self.latest("western"),
                "medium":  self.medium("western"),
                "slope":   self.slope("western"),
                "history": list(self._western),
            },
            "vedic": {
                "latest":  self.latest("vedic"),
                "medium":  self.medium("vedic"),
                "slope":   self.slope("vedic"),
                "history": list(self._vedic),
            },
            "ready": self.is_ready(),
        }

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self):
        """Write current buffer to disk so it survives restarts."""
        if not self._path:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._path, "w") as f:
            json.dump({
                "window":  self.window,
                "western": list(self._western),
                "vedic":   list(self._vedic),
            }, f, indent=2)

    def load(self):
        """Load buffer from disk if the file exists."""
        if not self._path or not self._path.exists():
            return
        try:
            with open(self._path) as f:
                data = json.load(f)
            for w in data.get("western", []):
                self._western.append(w)
            for v in data.get("vedic", []):
                self._vedic.append(v)
        except Exception:
            pass  # corrupted file — start fresh
