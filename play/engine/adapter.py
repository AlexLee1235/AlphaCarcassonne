from __future__ import annotations

import random
import secrets
from typing import Dict, List, Optional, Tuple

try:
    from domain import GameState, Move, PlacedTile
except ImportError:  # pragma: no cover - package import fallback
    from ..domain import GameState, Move, PlacedTile

try:
    from .. import _carcassonne_cpp
except ImportError:
    try:
        import _carcassonne_cpp
    except ImportError as exc:  # pragma: no cover - native extension setup guard
        raise ImportError(
            "Could not import the Carcassonne native engine. Build it with "
            "`python -m pip install -e play` or `python play/setup.py build_ext --inplace`."
        ) from exc


BOARD_SIZE = 15
ENGINE_BOARD_SIZE = int(getattr(_carcassonne_cpp, "BOARD_SIZE", BOARD_SIZE))
START_POS = (ENGINE_BOARD_SIZE // 2, ENGINE_BOARD_SIZE // 2)
PHASE_CHANCE = int(_carcassonne_cpp.PHASE_CHANCE)
PHASE_TILE = int(_carcassonne_cpp.PHASE_TILE)
PHASE_MEEPLE = int(_carcassonne_cpp.PHASE_MEEPLE)
PHASE_TERMINAL = int(_carcassonne_cpp.PHASE_TERMINAL)
PHYSICAL_TO_CANONICAL_TYPE = list(getattr(_carcassonne_cpp, "PHYSICAL_TO_CANONICAL_TYPE", []))


def _clamp(value: int, lower: int, upper: int) -> int:
    return max(lower, min(upper, value))


def _physical_to_art_id(physical_id: int) -> int:
    if physical_id <= 0:
        return 0
    return PHYSICAL_TO_CANONICAL_TYPE[physical_id]


class CppCarcassonneAdapter:
    def __init__(self, seed: Optional[int] = None):
        if seed is None:
            seed = secrets.randbits(32)
        self._rng = random.Random(seed)
        self._engine = _carcassonne_cpp.Carcassonne()
        self._turn = 1
        self._pending_meeple_options: List[int] = []
        self._viewport_origin = self._default_viewport_origin()
        self._resolve_chance_phase()
        self.state = self._build_state()

    @property
    def view_origin(self) -> Tuple[int, int]:
        return self._viewport_origin

    def to_engine_coords(self, x: int, y: int) -> Tuple[int, int]:
        origin_x, origin_y = self._viewport_origin
        return origin_x + x, origin_y + y

    def _default_viewport_origin(self) -> Tuple[int, int]:
        max_origin = max(0, ENGINE_BOARD_SIZE - BOARD_SIZE)
        return (
            _clamp(START_POS[0] - BOARD_SIZE // 2, 0, max_origin),
            _clamp(START_POS[1] - BOARD_SIZE // 2, 0, max_origin),
        )

    def _to_ui_coords(self, x: int, y: int) -> Optional[Tuple[int, int]]:
        origin_x, origin_y = self._viewport_origin
        ui_x = x - origin_x
        ui_y = y - origin_y
        if 0 <= ui_x < BOARD_SIZE and 0 <= ui_y < BOARD_SIZE:
            return ui_x, ui_y
        return None

    def _sample_draw_type(self, draws: List[Tuple[int, float]]) -> int:
        ticket = self._rng.random()
        cumulative = 0.0
        for type_id, probability in draws:
            cumulative += probability
            if ticket <= cumulative:
                return type_id
        return draws[-1][0]

    def _resolve_chance_phase(self) -> None:
        while self._engine.current_phase == PHASE_CHANCE and not self._engine.is_game_over:
            draws = list(self._engine.get_available_draws())
            if not draws:
                break
            self._engine.draw_tile(self._sample_draw_type(draws))

    def can_pan(self, dx: int, dy: int) -> bool:
        origin_x, origin_y = self._viewport_origin
        max_origin = max(0, ENGINE_BOARD_SIZE - BOARD_SIZE)
        next_x = _clamp(origin_x + dx, 0, max_origin)
        next_y = _clamp(origin_y + dy, 0, max_origin)
        return next_x != origin_x or next_y != origin_y

    def pan(self, dx: int, dy: int) -> bool:
        origin_x, origin_y = self._viewport_origin
        max_origin = max(0, ENGINE_BOARD_SIZE - BOARD_SIZE)
        next_origin = (
            _clamp(origin_x + dx, 0, max_origin),
            _clamp(origin_y + dy, 0, max_origin),
        )
        if next_origin == self._viewport_origin:
            return False
        self._viewport_origin = next_origin
        return True

    def get_valid_moves(self) -> List[Move]:
        if self._pending_meeple_options or self._engine.current_phase != PHASE_TILE:
            return []

        visible_moves: List[Move] = []
        for x, y, r in self._engine.get_legal_tile_moves():
            ui_pos = self._to_ui_coords(x, y)
            if ui_pos is None:
                continue
            visible_moves.append(Move(x=ui_pos[0], y=ui_pos[1], rotation=r))
        return visible_moves

    def confirm_tile(self, move: Move) -> List[int]:
        if self.state.game_over:
            return []

        engine_x, engine_y = self.to_engine_coords(move.x, move.y)
        legal = {(x, y, r) for (x, y, r) in self._engine.get_legal_tile_moves()}
        if (engine_x, engine_y, move.rotation) not in legal:
            raise ValueError(f"Invalid move: ({move.x}, {move.y}, r={move.rotation})")

        self._engine.place_tile(engine_x, engine_y, move.rotation)
        self._pending_meeple_options = list(self._engine.get_legal_meeple_moves())
        self.state = self._build_state()
        return list(self._pending_meeple_options)

    def apply_meeple(self, meeple_pos: int) -> None:
        if meeple_pos not in self._pending_meeple_options:
            raise ValueError(f"Invalid meeple position: {meeple_pos}")

        self._engine.place_meeple(meeple_pos)
        self._pending_meeple_options = []
        self._turn += 1
        self._resolve_chance_phase()
        self.state = self._build_state()

    def _build_board(self) -> Dict[Tuple[int, int], PlacedTile]:
        board: Dict[Tuple[int, int], PlacedTile] = {}
        for x, y, physical_id, rotation in self._engine.get_placed_tiles():
            board[(x, y)] = PlacedTile(
                tile_id=_physical_to_art_id(physical_id),
                rotation=rotation,
            )
        for player, x, y, pos in self._engine.get_meeple_tokens():
            tile = board.get((x, y))
            if tile is None:
                raise RuntimeError(f"Native meeple token ({player}, {x}, {y}, {pos}) has no matching tile snapshot")
            owner = player + 1
            tile.meeple_markers.append((owner, pos))
            if tile.meeple_owner is None:
                tile.meeple_owner = owner
                tile.meeple_pos = pos
        return board

    def _build_state(self) -> GameState:
        game_over = bool(self._engine.is_game_over)
        holding_tile_id = None
        if self._engine.current_phase == PHASE_TILE:
            holding_tile_id = _physical_to_art_id(self._engine.current_tile_in_hand)

        scores = self._engine.player_scores
        meeples = self._engine.holding_meeples
        current_player_ui = self._engine.current_player + 1

        return GameState(
            board=self._build_board(),
            current_player=current_player_ui,
            holding_tile_id=holding_tile_id,
            draw_pile=[],
            deck_counts={},
            scores={1: scores[0], 2: scores[1]},
            meeples_remaining={1: meeples[0], 2: meeples[1]},
            game_over=game_over,
            turn=self._turn,
        )
