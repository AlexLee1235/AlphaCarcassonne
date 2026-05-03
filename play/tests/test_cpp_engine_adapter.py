from __future__ import annotations

import pytest

from play import _carcassonne_cpp
from play.cpp_engine import BOARD_SIZE, CppCarcassonneAdapter, ENGINE_BOARD_SIZE, PHASE_TILE, START_POS
from play.models import Move


def _resolve_native_to_tile_phase(engine: _carcassonne_cpp.Carcassonne) -> None:
    while engine.current_phase == _carcassonne_cpp.PHASE_CHANCE and not engine.is_game_over:
        draws = list(engine.get_available_draws())
        assert draws
        engine.draw_tile(draws[0][0])


def test_native_binding_initial_snapshot_and_chance_distribution() -> None:
    engine = _carcassonne_cpp.Carcassonne()

    placed = list(engine.get_placed_tiles())
    assert (START_POS[0], START_POS[1], placed[0][2], 0) in placed
    assert _carcassonne_cpp.PHYSICAL_TO_CANONICAL_TYPE[placed[0][2]] == 20

    draws = list(engine.get_available_draws())
    assert draws
    assert sum(probability for _, probability in draws) == pytest.approx(1.0)


def test_native_binding_tile_and_meeple_flow_smoke() -> None:
    engine = _carcassonne_cpp.Carcassonne()
    _resolve_native_to_tile_phase(engine)

    moves = list(engine.get_legal_tile_moves())
    assert moves
    x, y, rot = moves[0]
    engine.place_tile(x, y, rot)
    assert engine.current_phase == _carcassonne_cpp.PHASE_MEEPLE

    meeple_moves = list(engine.get_legal_meeple_moves())
    assert -1 in meeple_moves
    playable = [pos for pos in meeple_moves if pos != -1]
    engine.place_meeple(playable[0] if playable else -1)

    for player, token_x, token_y, pos in engine.get_meeple_tokens():
        assert player in (0, 1)
        assert 0 <= token_x < ENGINE_BOARD_SIZE
        assert 0 <= token_y < ENGINE_BOARD_SIZE
        assert 0 <= pos <= 4


def test_adapter_initial_state_and_viewport_center() -> None:
    adapter = CppCarcassonneAdapter(seed=42)

    assert adapter.state.current_player == 1
    assert adapter.state.turn == 1
    assert adapter.state.holding_tile_id is not None
    assert len(adapter.get_valid_moves()) > 0
    expected_origin = (
        max(0, min(START_POS[0] - BOARD_SIZE // 2, max(0, ENGINE_BOARD_SIZE - BOARD_SIZE))),
        max(0, min(START_POS[1] - BOARD_SIZE // 2, max(0, ENGINE_BOARD_SIZE - BOARD_SIZE))),
    )
    assert adapter.view_origin == expected_origin


def test_adapter_confirm_tile_then_apply_meeple_advances_turn() -> None:
    adapter = CppCarcassonneAdapter(seed=42)
    valid_moves = adapter.get_valid_moves()
    assert valid_moves

    move = valid_moves[0]
    meeple_options = adapter.confirm_tile(move)
    assert adapter.state.holding_tile_id is None
    assert meeple_options

    adapter.apply_meeple(-1)
    assert adapter.state.turn == 2
    assert adapter.state.current_player == 2
    assert adapter._engine.current_phase == PHASE_TILE


def test_adapter_rejects_invalid_tile_confirmation() -> None:
    adapter = CppCarcassonneAdapter(seed=42)
    with pytest.raises(ValueError):
        adapter.confirm_tile(Move(x=0, y=0, rotation=0))


def test_adapter_pan_is_manual_and_clears_selection() -> None:
    adapter = CppCarcassonneAdapter(seed=42)
    original_origin = adapter.view_origin
    valid_moves = adapter.get_valid_moves()
    assert valid_moves

    adapter.selected_move = valid_moves[0]
    moved = adapter.pan(1, 0)
    if ENGINE_BOARD_SIZE == BOARD_SIZE:
        assert not moved
        assert adapter.view_origin == original_origin
    else:
        assert moved
        assert adapter.view_origin == (original_origin[0] + 1, original_origin[1])


def test_adapter_tiles_are_mapped_to_ui_canonical_ids() -> None:
    adapter = CppCarcassonneAdapter(seed=42)
    center = START_POS
    placed = adapter.state.board[center]

    assert placed.tile_id == 20
    assert placed.rotation == 0
    assert placed.meeple_owner is None
    assert placed.meeple_pos is None


def test_adapter_maps_active_meeple_into_board_snapshot() -> None:
    adapter = CppCarcassonneAdapter(seed=42)
    move = adapter.get_valid_moves()[0]

    meeple_options = adapter.confirm_tile(move)
    playable_options = [pos for pos in meeple_options if pos != -1]
    assert playable_options

    adapter.apply_meeple(playable_options[0])

    marked_tiles = [tile for tile in adapter.state.board.values() if tile.meeple_owner == 1]
    assert marked_tiles
    assert all(tile.meeple_pos is not None and 0 <= tile.meeple_pos <= 4 for tile in marked_tiles)
