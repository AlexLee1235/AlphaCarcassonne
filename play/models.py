from __future__ import annotations

try:
    from domain.models import BOARD_SIZE, BotValue, FeatureType, GameState, Move, MoveRecord, PlacedTile, ScoreEvent, TurnResult
except ImportError:  # pragma: no cover - package import fallback
    from .domain.models import BOARD_SIZE, BotValue, FeatureType, GameState, Move, MoveRecord, PlacedTile, ScoreEvent, TurnResult

__all__ = [
    "BOARD_SIZE",
    "FeatureType",
    "GameState",
    "Move",
    "BotValue",
    "MoveRecord",
    "PlacedTile",
    "ScoreEvent",
    "TurnResult",
]
