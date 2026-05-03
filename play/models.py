from __future__ import annotations

try:
    from domain.models import BOARD_SIZE, FeatureType, GameState, Move, PlacedTile, ScoreEvent, TurnResult
except ImportError:  # pragma: no cover - package import fallback
    from .domain.models import BOARD_SIZE, FeatureType, GameState, Move, PlacedTile, ScoreEvent, TurnResult

__all__ = [
    "BOARD_SIZE",
    "FeatureType",
    "GameState",
    "Move",
    "PlacedTile",
    "ScoreEvent",
    "TurnResult",
]
