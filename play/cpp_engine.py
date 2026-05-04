from __future__ import annotations

try:
    from engine.adapter import (
        BOARD_SIZE,
        ENGINE_BOARD_SIZE,
        PHASE_CHANCE,
        PHASE_MEEPLE,
        PHASE_TERMINAL,
        PHASE_TILE,
        PHYSICAL_TO_CANONICAL_TYPE,
        OPPONENT_MODES,
        START_POS,
        CppCarcassonneAdapter,
    )
except ImportError:  # pragma: no cover - package import fallback
    from .engine.adapter import (
        BOARD_SIZE,
        ENGINE_BOARD_SIZE,
        PHASE_CHANCE,
        PHASE_MEEPLE,
        PHASE_TERMINAL,
        PHASE_TILE,
        PHYSICAL_TO_CANONICAL_TYPE,
        OPPONENT_MODES,
        START_POS,
        CppCarcassonneAdapter,
    )

__all__ = [
    "BOARD_SIZE",
    "ENGINE_BOARD_SIZE",
    "PHASE_CHANCE",
    "PHASE_MEEPLE",
    "PHASE_TERMINAL",
    "PHASE_TILE",
    "PHYSICAL_TO_CANONICAL_TYPE",
    "OPPONENT_MODES",
    "START_POS",
    "CppCarcassonneAdapter",
]
