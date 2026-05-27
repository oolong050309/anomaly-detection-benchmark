"""图专属算法包。"""

from __future__ import annotations

from models.graph.cola import CoLADetector
from models.graph.dominant import DOMINANTDetector

__all__ = [
    "DOMINANTDetector",
    "CoLADetector",
]
