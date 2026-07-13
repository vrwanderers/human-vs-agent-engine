"""Human vs Agent evaluation-first game engine."""

from .engine import GameEngine, build_default_engine
from .stimulus import RealityStatus, StimulusModality, StimulusPrivacy

__all__ = [
    "GameEngine",
    "RealityStatus",
    "StimulusModality",
    "StimulusPrivacy",
    "build_default_engine",
]
__version__ = "0.1.0"
