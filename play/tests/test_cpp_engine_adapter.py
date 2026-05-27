from __future__ import annotations

import pytest

from play import _carcassonne_cpp
from play.cpp_engine import BOARD_SIZE, CppCarcassonneAdapter, ENGINE_BOARD_SIZE, PHASE_TILE, START_POS, PlayerSpec
from play.engine.adapter import BotCliClient
from play.models import Move
from play.ui.app import parse_ui_config


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


def test_bot_cli_random_and_mcts_return_legal_actions() -> None:
    engine = _carcassonne_cpp.Carcassonne()
    cli = BotCliClient()
    try:
        draw_type = list(engine.get_available_draws())[0][0]
        engine.draw_tile(draw_type)
        cli.request({"cmd": "apply_draw", "type": draw_type})

        legal_tile_moves = set(engine.get_legal_tile_moves())
        random_response = cli.request({"cmd": "choose", "bot": "random", "seed": 123})
        random_tile = (random_response["x"], random_response["y"], random_response["rot"])
        assert random_response["kind"] == "tile"
        assert random_tile in legal_tile_moves

        mcts_response = cli.request({"cmd": "choose", "bot": "mcts", "seed": 123, "simulations": 10})
        mcts_tile = (mcts_response["x"], mcts_response["y"], mcts_response["rot"])
        assert mcts_response["kind"] == "tile"
        assert mcts_tile in legal_tile_moves

        engine.place_tile(*random_tile)
        cli.request({"cmd": "apply_tile", "x": random_tile[0], "y": random_tile[1], "rot": random_tile[2]})
        legal_meeple_moves = set(engine.get_legal_meeple_moves())
        random_meeple = cli.request({"cmd": "choose", "bot": "random", "seed": 123})
        assert random_meeple["kind"] == "meeple"
        assert random_meeple["pos"] in legal_meeple_moves

        mcts_meeple = cli.request({"cmd": "choose", "bot": "mcts", "seed": 123, "simulations": 10})
        assert mcts_meeple["kind"] == "meeple"
        assert mcts_meeple["pos"] in legal_meeple_moves
    finally:
        cli.close()


def test_bot_cli_reports_latest_observation_shape() -> None:
    cli = BotCliClient()
    try:
        response = cli.request({"cmd": "info"})
    finally:
        cli.close()

    assert response["observation_shape"] == [80, ENGINE_BOARD_SIZE, ENGINE_BOARD_SIZE]
    assert response["observation_tensor_size"] == 80 * ENGINE_BOARD_SIZE * ENGINE_BOARD_SIZE
    assert response["num_distinct_actions"] == ENGINE_BOARD_SIZE * ENGINE_BOARD_SIZE * 4 + 6


def test_player_spec_builds_per_player_az_env_without_device() -> None:
    spec = PlayerSpec(
        type="az",
        az_path="/tmp/model",
        az_checkpoint=64,
        az_graph_def="vpnet.pb",
        max_simulations=800,
    )

    assert spec.type == "alphazero"
    assert spec.bot_env() == {
        "CARCASSONNE_AZ_PATH": "/tmp/model",
        "CARCASSONNE_AZ_CHECKPOINT": "64",
        "CARCASSONNE_AZ_GRAPH_DEF": "vpnet.pb",
        "CARCASSONNE_AZ_SIMULATIONS": "800",
    }


def test_ui_parser_uses_alpha_zero_style_player_args() -> None:
    config = parse_ui_config(
        [
            "--game=carcassonne",
            "--p1_type=human",
            "--p2_type=az",
            "--p2_az_path=/tmp/model",
            "--p2_az_checkpoint=64",
            "--p2_az_graph_def=vpnet.pb",
            "--p2_max_simulations=800",
            "--seed=123",
        ]
    )

    assert config.seed == 123
    assert config.p1_spec.type == "human"
    assert config.p2_spec.type == "alphazero"
    assert config.p2_spec.az_path == "/tmp/model"
    assert config.p2_spec.az_checkpoint == 64
    assert config.p2_spec.max_simulations == 800


def test_ui_parser_rejects_az_device_args() -> None:
    with pytest.raises(SystemExit):
        parse_ui_config(["--p2_az_device=/cuda:0"])


def test_bot_cli_alphazero_requires_model_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CARCASSONNE_AZ_PATH", raising=False)
    engine = _carcassonne_cpp.Carcassonne()
    cli = BotCliClient()
    try:
        draw_type = list(engine.get_available_draws())[0][0]
        cli.request({"cmd": "apply_draw", "type": draw_type})
        with pytest.raises(RuntimeError, match="CARCASSONNE_AZ_PATH"):
            cli.request({"cmd": "choose", "bot": "alphazero", "seed": 123, "simulations": 1})
    finally:
        cli.close()


def test_bot_cli_az_alias_requires_model_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CARCASSONNE_AZ_PATH", raising=False)
    engine = _carcassonne_cpp.Carcassonne()
    cli = BotCliClient()
    try:
        draw_type = list(engine.get_available_draws())[0][0]
        cli.request({"cmd": "apply_draw", "type": draw_type})
        with pytest.raises(RuntimeError, match="CARCASSONNE_AZ_PATH"):
            cli.request({"cmd": "choose", "bot": "az", "seed": 123, "simulations": 1})
    finally:
        cli.close()


def test_bot_cli_alphazero_rejects_legacy_observation_shape(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    monkeypatch.setenv("CARCASSONNE_AZ_PATH", str(tmp_path))
    monkeypatch.setenv("CARCASSONNE_AZ_GRAPH_DEF", "vpnet.pb")
    (tmp_path / "vpnet.pb").write_text("51 15 15 906 12 128 0.0001 0.0001 resnet")

    engine = _carcassonne_cpp.Carcassonne()
    cli = BotCliClient()
    try:
        draw_type = list(engine.get_available_draws())[0][0]
        cli.request({"cmd": "apply_draw", "type": draw_type})
        with pytest.raises(RuntimeError, match="model observation shape.*current game shape"):
            cli.request({"cmd": "choose", "bot": "alphazero", "seed": 123, "simulations": 1})
    finally:
        cli.close()


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
    engine_pos = adapter.to_engine_coords(move.x, move.y)
    assert adapter.state.board[engine_pos].tile_owner == 1
    assert sum(tile.tile_owner is not None for tile in adapter.state.board.values()) == 1
    assert adapter.state.holding_tile_id is None
    assert meeple_options

    adapter.apply_meeple(-1)
    assert adapter.state.turn == 2
    assert adapter.state.current_player == 2
    assert adapter._engine.current_phase == PHASE_TILE

    next_move = adapter.get_valid_moves()[0]
    adapter.confirm_tile(next_move)
    next_engine_pos = adapter.to_engine_coords(next_move.x, next_move.y)
    assert adapter.state.board[engine_pos].tile_owner is None
    assert adapter.state.board[next_engine_pos].tile_owner == 2
    assert sum(tile.tile_owner is not None for tile in adapter.state.board.values()) == 1


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
    assert placed.tile_owner is None
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


def test_adapter_merges_overlapping_player_meeple_markers_to_owner_zero() -> None:
    class FakeEngine:
        def get_placed_tiles(self):
            return [(START_POS[0], START_POS[1], 1, 0)]

        def get_meeple_tokens(self):
            return [(0, START_POS[0], START_POS[1], 2), (1, START_POS[0], START_POS[1], 2)]

    adapter = CppCarcassonneAdapter(seed=42)
    adapter._engine = FakeEngine()

    tile = adapter._build_board()[START_POS]

    assert tile.meeple_markers == [(0, 2)]
    assert tile.meeple_owner == 0
    assert tile.meeple_pos == 2


def test_adapter_random_opponent_auto_plays_back_to_human() -> None:
    adapter = CppCarcassonneAdapter(seed=42, opponent_mode="random")
    move = adapter.get_valid_moves()[0]

    adapter.confirm_tile(move)
    adapter.apply_meeple(-1)

    assert adapter.state.game_over or adapter.state.current_player == 1
    assert adapter.ai_status
    assert adapter.state.game_over or adapter.get_valid_moves()
    marked_tile_owners = [tile.tile_owner for tile in adapter.state.board.values() if tile.tile_owner is not None]
    assert len(marked_tile_owners) == 1
    assert adapter.state.game_over or marked_tile_owners == [2]


def test_adapter_p1_bot_p2_human_auto_plays_to_human() -> None:
    adapter = CppCarcassonneAdapter(
        seed=42,
        player_specs=(PlayerSpec(type="random"), PlayerSpec(type="human")),
    )
    try:
        assert adapter.state.game_over or adapter.state.current_player == 2
        assert adapter.ai_status
        assert adapter.state.game_over or adapter.get_valid_moves()
        marked_tile_owners = [tile.tile_owner for tile in adapter.state.board.values() if tile.tile_owner is not None]
        assert len(marked_tile_owners) == 1
        assert adapter.state.game_over or marked_tile_owners == [1]
    finally:
        adapter.close()


def test_adapter_random_vs_random_auto_finishes() -> None:
    adapter = CppCarcassonneAdapter(
        seed=42,
        player_specs=(PlayerSpec(type="random"), PlayerSpec(type="random")),
    )
    try:
        assert adapter.state.game_over
        assert adapter.ai_status
        assert not adapter.get_valid_moves()
    finally:
        adapter.close()


def test_adapter_mcts_opponent_auto_plays_back_to_human(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CARCASSONNE_MCTS_SIMULATIONS", "20")
    adapter = CppCarcassonneAdapter(seed=42, opponent_mode="mcts")
    move = adapter.get_valid_moves()[0]

    adapter.confirm_tile(move)
    adapter.apply_meeple(-1)

    assert adapter.state.game_over or adapter.state.current_player == 1
    assert adapter.ai_status
    assert adapter.state.game_over or adapter.get_valid_moves()


def test_adapter_alphazero_reports_missing_model_without_crashing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CARCASSONNE_AZ_PATH", raising=False)
    adapter = CppCarcassonneAdapter(seed=42, opponent_mode="alphazero")
    move = adapter.get_valid_moves()[0]

    adapter.confirm_tile(move)
    adapter.apply_meeple(-1)

    assert "CARCASSONNE_AZ_PATH" in adapter.ai_status
    adapter.close()


def test_adapter_accepts_az_alias() -> None:
    adapter = CppCarcassonneAdapter(seed=42, opponent_mode="az")
    try:
        assert adapter.opponent_mode == "alphazero"
    finally:
        adapter.close()
