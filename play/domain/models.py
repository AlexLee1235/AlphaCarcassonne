from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple


BOARD_SIZE = 15


class FeatureType(str, Enum):
    CITY = "city"
    ROAD = "road"
    MONASTERY = "monastery"


@dataclass(frozen=True)
class Move:
    x: int
    y: int
    rotation: int


@dataclass
class PlacedTile:
    tile_id: int
    rotation: int
    meeple_owner: Optional[int] = None
    meeple_pos: Optional[int] = None  # 0-3 edges, 4 center
    meeple_markers: List[Tuple[int, int]] = field(default_factory=list)  # (owner, pos)


@dataclass
class ScoreEvent:
    player: int
    points: int
    reason: str
    positions: List[Tuple[int, int]] = field(default_factory=list)


@dataclass
class GameState:
    board: Dict[Tuple[int, int], PlacedTile]
    current_player: int
    holding_tile_id: Optional[int]
    draw_pile: List[int]
    deck_counts: Dict[int, int]
    scores: Dict[int, int]
    meeples_remaining: Dict[int, int]
    game_over: bool = False
    turn: int = 1


@dataclass
class TurnResult:
    state: GameState
    score_events: List[ScoreEvent]
    next_valid_moves: List[Move]
