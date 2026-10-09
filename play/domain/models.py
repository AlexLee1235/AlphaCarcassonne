from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple


BOARD_SIZE = 27  # must match the engine's VIEW_SIZE (game.hpp)


class FeatureType(str, Enum):
    CITY = "city"
    ROAD = "road"
    MONASTERY = "monastery"


@dataclass(frozen=True)
class Move:
    x: int
    y: int
    rotation: int


@dataclass(frozen=True)
class MoveRecord:
    player: int
    tile_id: int
    x: int
    y: int
    rotation: int
    meeple_pos: int
    score_deltas: Dict[int, int] = field(default_factory=dict)
    # An AlphaZero move's estimate of the game from the mover's side, -1..1: the
    # search's mean return and the network's raw value. None for other players.
    value: Optional[float] = None
    raw_value: Optional[float] = None
    # The turn's moves after its tile, as the actor logs write them: the meeple
    # move, or a choice by cell and its spot, then the dragon's steps.
    actions: Tuple[str, ...] = ()


@dataclass(frozen=True)
class BotValue:
    """What an AlphaZero player thought of the position when it placed its tile."""

    player: int
    value: float
    raw_value: float
    simulations: int


@dataclass
class PlacedTile:
    tile_id: int
    rotation: int
    tile_owner: Optional[int] = None
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
    # The expansions' pieces this game plays with ("big", "builder", "pig"), and
    # how many of each every player still holds: {player: {kind: count}}.
    piece_kinds: Tuple[str, ...] = ()
    pieces_remaining: Dict[int, Dict[str, int]] = field(default_factory=dict)
    # Traders & Builders: each player's goods tokens (wine, wheat, cloth); empty without its rules.
    goods: Dict[int, Tuple[int, int, int]] = field(default_factory=dict)
    # The player to move is on the extra tile their builder gave them.
    builder_extra_tile: bool = False
    # The engine's phase (PHASE_* of the native module).
    phase: int = 0
    # The Princess & the Dragon: the choices by cell this game plays ("portal",
    # "princess", "fairy"); the dragon's board cell (None until the first
    # volcano), the cells it has visited in the move under way and its steps so
    # far; the fairy's board cell and the spot of the meeple it stands next to
    # (-1 for none; None while in the supply), and that meeple's owner.
    choice_kinds: Tuple[str, ...] = ()
    dragon: Optional[Tuple[int, int]] = None
    dragon_visited: List[Tuple[int, int]] = field(default_factory=list)
    dragon_steps: int = 0
    fairy: Optional[Tuple[int, int, int]] = None
    fairy_owner: Optional[int] = None


@dataclass
class TurnResult:
    state: GameState
    score_events: List[ScoreEvent]
    next_valid_moves: List[Move]
