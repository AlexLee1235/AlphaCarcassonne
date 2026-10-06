from __future__ import annotations

import pytest

from play import _carcassonne_cpp
from play.cpp_engine import (
    BOARD_SIZE,
    ENGINE_BOARD_SIZE,
    HALF_EDGE_COUNT,
    MEEPLE_POS_FIELD,
    MEEPLE_POS_INNER_FIELD,
    PHASE_TILE,
    START_POS,
    CppCarcassonneAdapter,
    PlayerSpec,
)
from play.engine import adapter as adapter_module
from play.engine.adapter import BotCliClient
from pathlib import Path

from play.models import BotValue, Move, MoveRecord
from play.ui.app import (
    build_player_specs,
    format_bot_value,
    format_move_record,
    human_seat,
    is_farmer,
    meeple_alignment,
    meeple_button_labels,
    parse_ui_config,
    should_show_start_game,
    summarize_ai_status,
)


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

    # 101 spatial planes (fields and expansion terrain included) and one global
    # plane; meeple positions -1..13.
    assert response["observation_shape"] == [115, ENGINE_BOARD_SIZE, ENGINE_BOARD_SIZE]
    assert response["observation_tensor_size"] == 115 * ENGINE_BOARD_SIZE * ENGINE_BOARD_SIZE
    assert response["num_distinct_actions"] == ENGINE_BOARD_SIZE * ENGINE_BOARD_SIZE * 4 + 29


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


def test_format_move_record() -> None:
    assert (
        format_move_record(
            MoveRecord(player=1, tile_id=20, x=7, y=8, rotation=2, meeple_pos=-1, score_deltas={1: 0, 2: 0})
        )
        == "P1(20,7,8,2,-1) +0(得分)"
    )
    assert (
        format_move_record(
            MoveRecord(player=1, tile_id=20, x=7, y=8, rotation=2, meeple_pos=4, score_deltas={1: 4, 2: 0})
        )
        == "P1(20,7,8,2,4) +4(得分)"
    )
    assert (
        format_move_record(
            MoveRecord(player=1, tile_id=20, x=7, y=8, rotation=2, meeple_pos=4, score_deltas={1: 0, 2: 4})
        )
        == "P1(20,7,8,2,4) P2+4(得分)"
    )
    assert (
        format_move_record(
            MoveRecord(player=1, tile_id=20, x=7, y=8, rotation=2, meeple_pos=4, score_deltas={1: 4, 2: 3})
        )
        == "P1(20,7,8,2,4) P1+4/P2+3(得分)"
    )


def test_should_show_start_game_only_for_unstarted_bot_vs_bot() -> None:
    assert should_show_start_game((PlayerSpec(type="random"), PlayerSpec(type="mcts")), started=False)
    assert not should_show_start_game((PlayerSpec(type="random"), PlayerSpec(type="mcts")), started=True)
    assert not should_show_start_game((PlayerSpec(type="human"), PlayerSpec(type="mcts")), started=False)
    assert not should_show_start_game((PlayerSpec(type="random"), PlayerSpec(type="human")), started=False)


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
    placed_tile_id = adapter.state.board[engine_pos].tile_id
    assert sum(tile.tile_owner is not None for tile in adapter.state.board.values()) == 1
    assert adapter.state.holding_tile_id is None
    assert meeple_options

    adapter.apply_meeple(-1)
    assert adapter.state.turn == 2
    assert adapter.state.current_player == 2
    assert adapter._engine.current_phase == PHASE_TILE
    assert len(adapter.move_records) == 1
    first_record = adapter.move_records[0]
    assert first_record.player == 1
    assert first_record.tile_id == placed_tile_id
    assert (first_record.x, first_record.y, first_record.rotation, first_record.meeple_pos) == (
        engine_pos[0],
        engine_pos[1],
        move.rotation,
        -1,
    )
    assert set(first_record.score_deltas) == {1, 2}

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
    assert adapter.state.meeples_remaining[1] == 6  # still on the board, not scored back

    record = adapter.move_records[0]
    marked = {pos: tile for pos, tile in adapter.state.board.items() if tile.meeple_markers}
    assert list(marked) == [(record.x, record.y)]
    assert marked[(record.x, record.y)].meeple_markers == [(1, playable_options[0])]


class FakeMeepleEngine:
    def __init__(self, tiles, tokens):
        self.tiles = tiles
        self.tokens = tokens

    def get_placed_tiles(self):
        return [(x, y, 1, 0) for x, y in self.tiles]

    def get_meeple_tokens(self):
        return self.tokens


def _meeple_record(player: int, x: int, y: int, meeple_pos: int) -> MoveRecord:
    return MoveRecord(player=player, tile_id=1, x=x, y=y, rotation=0, meeple_pos=meeple_pos)


def test_adapter_draws_meeple_only_where_it_was_placed() -> None:
    sx, sy = START_POS
    road = [(sx, sy), (sx + 1, sy), (sx + 2, sy)]
    adapter = CppCarcassonneAdapter(seed=42)
    # The engine marks the claimed road on every tile it runs through.
    adapter._engine = FakeMeepleEngine(road, [(0, x, y, 1) for x, y in road] + [(0, x, y, 3) for x, y in road])
    adapter.move_records = [_meeple_record(1, sx + 1, sy, 3), _meeple_record(2, sx + 2, sy, -1)]

    board = adapter._build_board()

    assert {pos: tile.meeple_markers for pos, tile in board.items() if tile.meeple_markers} == {(sx + 1, sy): [(1, 3)]}
    assert (board[(sx + 1, sy)].meeple_owner, board[(sx + 1, sy)].meeple_pos) == (1, 3)


def test_adapter_hides_meeple_returned_by_a_completed_feature() -> None:
    sx, sy = START_POS
    adapter = CppCarcassonneAdapter(seed=42)
    adapter._engine = FakeMeepleEngine([(sx, sy), (sx + 1, sy)], [])
    adapter.move_records = [_meeple_record(1, sx + 1, sy, 0), _meeple_record(2, sx, sy, 4)]

    assert not any(tile.meeple_markers for tile in adapter._build_board().values())


def test_adapter_keeps_farmers_although_they_have_no_token() -> None:
    sx, sy = START_POS
    adapter = CppCarcassonneAdapter(seed=42)
    # get_meeple_tokens() only covers roads, cities and monasteries; farmers never leave.
    adapter._engine = FakeMeepleEngine([(sx, sy), (sx + 1, sy)], [])
    farmer = MEEPLE_POS_FIELD + 3
    adapter.move_records = [_meeple_record(2, sx + 1, sy, farmer), _meeple_record(1, sx, sy, 2)]

    board = adapter._build_board()

    assert board[(sx + 1, sy)].meeple_markers == [(2, farmer)]
    assert board[(sx, sy)].meeple_markers == []  # the road meeple was scored and returned


def test_adapter_keeps_each_player_meeple_on_a_shared_feature() -> None:
    sx, sy = START_POS
    city = [(sx, sy), (sx + 1, sy)]
    adapter = CppCarcassonneAdapter(seed=42)
    tokens = [(player, x, y, pos) for player in (0, 1) for x, y in city for pos in (1, 3)]
    adapter._engine = FakeMeepleEngine(city, tokens)
    adapter.move_records = [_meeple_record(2, sx + 1, sy, 3), _meeple_record(1, sx, sy, 1)]

    board = adapter._build_board()

    assert board[(sx, sy)].meeple_markers == [(1, 1)]
    assert board[(sx + 1, sy)].meeple_markers == [(2, 3)]


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
    assert len(adapter.move_records) >= 2
    assert adapter.move_records[0].player == 2
    assert adapter.move_records[1].player == 1


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
        assert adapter.move_records
        assert adapter.move_records[0].player == 1
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
        assert len(adapter.move_records) > 1
        assert adapter.move_records[0].player in (1, 2)
    finally:
        adapter.close()


def test_adapter_random_vs_random_can_wait_for_start() -> None:
    adapter = CppCarcassonneAdapter(
        seed=42,
        player_specs=(PlayerSpec(type="random"), PlayerSpec(type="random")),
        auto_run_bots=False,
    )
    try:
        assert not adapter.state.game_over
        assert not adapter.move_records

        adapter.run_ai_turns()

        assert adapter.state.game_over
        assert len(adapter.move_records) > 1
    finally:
        adapter.close()


def test_adapter_random_vs_random_can_step_one_bot_turn() -> None:
    adapter = CppCarcassonneAdapter(
        seed=42,
        player_specs=(PlayerSpec(type="random"), PlayerSpec(type="random")),
        auto_run_bots=False,
    )
    try:
        played_turns = adapter.run_ai_turns(max_turns=1)

        assert played_turns == 1
        assert not adapter.state.game_over
        assert len(adapter.move_records) == 1
        assert adapter.move_records[0].player == 1
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


class FakeBotCli:
    """Stands in for the bot CLI process: mirrors the game and plays the first legal move."""

    def __init__(self, path=None, env=None):
        self.env = env or {}
        self.mirror = _carcassonne_cpp.Carcassonne()

    def close(self) -> None:
        pass

    def request(self, payload: dict) -> dict:
        command = payload["cmd"]
        if command == "apply_draw":
            self.mirror.draw_tile(payload["type"])
        elif command == "apply_tile":
            self.mirror.place_tile(payload["x"], payload["y"], payload["rot"])
        elif command == "apply_meeple":
            self.mirror.place_meeple(payload["pos"])
        elif command == "choose":
            if self.mirror.current_phase == _carcassonne_cpp.PHASE_TILE:
                x, y, rot = list(self.mirror.get_legal_tile_moves())[0]
                return {"ok": True, "kind": "tile", "x": x, "y": y, "rot": rot}
            return {"ok": True, "kind": "meeple", "pos": list(self.mirror.get_legal_meeple_moves())[0]}
        return {"ok": True}


def test_player_spec_passes_value_perspective_to_bot_env() -> None:
    assert "CARCASSONNE_AZ_VALUE_IS_CURRENT_PLAYER" not in PlayerSpec(type="az", az_path="/m").bot_env()
    on = PlayerSpec(type="az", az_path="/m", value_is_current_player=True).bot_env()
    off = PlayerSpec(type="az", az_path="/m", value_is_current_player=False).bot_env()
    assert on["CARCASSONNE_AZ_VALUE_IS_CURRENT_PLAYER"] == "true"
    assert off["CARCASSONNE_AZ_VALUE_IS_CURRENT_PLAYER"] == "false"


def test_setup_panel_seats_human_against_bot() -> None:
    first = build_player_specs(1, "az", az_path=" /runs/0919 ", az_checkpoint=34, max_simulations=800)
    assert first[0].is_human
    assert (first[1].type, first[1].az_path, first[1].az_checkpoint, first[1].max_simulations) == (
        "alphazero",
        "/runs/0919",
        34,
        800,
    )
    assert human_seat(first) == 1

    second = build_player_specs(2, "random", az_path="/ignored", az_checkpoint=34, max_simulations=800)
    assert second[0].type == "random" and second[0].az_path == "" and second[0].max_simulations is None
    assert human_seat(second) == 2

    with pytest.raises(ValueError, match="model directory"):
        build_player_specs(1, "az", az_path="  ")


def test_ui_shows_the_whole_engine_board() -> None:
    adapter = CppCarcassonneAdapter(seed=7)
    assert BOARD_SIZE == ENGINE_BOARD_SIZE
    assert adapter.view_origin == (0, 0)
    # Every legal move, edge cells included, is offered to the human for a whole game.
    while not adapter.state.game_over:
        engine_moves = sorted(adapter._engine.get_legal_tile_moves())
        ui_moves = sorted((move.x, move.y, move.rotation) for move in adapter.get_valid_moves())
        assert ui_moves == engine_moves
        adapter.confirm_tile(adapter.get_valid_moves()[-1])
        adapter.apply_meeple(-1)


def test_adapter_without_auto_run_leaves_bot_turn_to_caller(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(adapter_module, "BotCliClient", FakeBotCli)
    adapter = CppCarcassonneAdapter(
        seed=42,
        player_specs=(PlayerSpec(type="human"), PlayerSpec(type="random")),
        auto_run_bots=False,
    )
    adapter.confirm_tile(adapter.get_valid_moves()[0])
    adapter.apply_meeple(-1)

    assert adapter.state.current_player == 2
    assert adapter.is_ai_turn()
    assert len(adapter.move_records) == 1

    assert adapter.run_ai_turns(1) == 1
    assert adapter.state.current_player == 1
    assert not adapter.is_ai_turn()
    assert [record.player for record in adapter.move_records] == [2, 1]


class FakeValueBotCli(FakeBotCli):
    """An AlphaZero-like bot: its tile choice also reports the search and network value."""

    def request(self, payload: dict) -> dict:
        response = super().request(payload)
        if response.get("kind") == "tile":
            response.update({"value": 0.42, "raw_value": -0.1, "simulations": 800})
        elif response.get("kind") == "meeple":
            response.update({"value": 0.99, "raw_value": 0.99, "simulations": 1})
        return response


def test_adapter_records_the_bot_tile_search_value(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(adapter_module, "BotCliClient", FakeValueBotCli)
    adapter = CppCarcassonneAdapter(
        seed=42,
        player_specs=(PlayerSpec(type="human"), PlayerSpec(type="random")),
        auto_run_bots=False,
    )
    adapter.confirm_tile(adapter.get_valid_moves()[0])
    adapter.apply_meeple(-1)
    assert adapter.last_bot_value is None

    assert adapter.run_ai_turns(1) == 1

    bot_record, human_record = adapter.move_records
    # The tile search's value is kept, not the meeple search's.
    assert (bot_record.value, bot_record.raw_value) == (0.42, -0.1)
    assert (human_record.value, human_record.raw_value) == (None, None)
    assert adapter.last_bot_value == BotValue(player=2, value=0.42, raw_value=-0.1, simulations=800)


def test_format_bot_value_and_record_suffix() -> None:
    assert format_bot_value(BotValue(player=2, value=0.42, raw_value=-0.1, simulations=800)) == (
        "AZ (P2) value +0.42 · net -0.10 · 800 sims"
    )
    record = MoveRecord(player=2, tile_id=20, x=7, y=8, rotation=2, meeple_pos=-1, score_deltas={}, value=-0.375)
    assert format_move_record(record) == "P2(20,7,8,2,-1) +0(得分) v-0.38"


AZ_MODEL_1005 = Path("/mnt/c/achieve/Carcassonne/1005")


@pytest.mark.skipif(not (AZ_MODEL_1005 / "vpnet.pb").exists(), reason="needs the 1005 model")
def test_bot_cli_alphazero_reports_value(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CARCASSONNE_AZ_PATH", str(AZ_MODEL_1005))
    monkeypatch.setenv("CARCASSONNE_AZ_CHECKPOINT", "75")
    engine = _carcassonne_cpp.Carcassonne()
    cli = BotCliClient()
    try:
        draw_type = list(engine.get_available_draws())[0][0]
        cli.request({"cmd": "apply_draw", "type": draw_type})
        response = cli.request({"cmd": "choose", "bot": "alphazero", "seed": 123, "simulations": 4})
    finally:
        cli.close()

    assert response["kind"] == "tile"
    assert -1.0 <= response["value"] <= 1.0
    assert -1.0 <= response["raw_value"] <= 1.0
    assert response["simulations"] >= 1


def test_bot_cli_rejects_binary_built_for_another_board() -> None:
    client = BotCliClient.__new__(BotCliClient)
    closed = []
    client.request = lambda payload: {"ok": True, "observation_shape": [80, 15, 15]}
    client.close = lambda: closed.append(True)

    with pytest.raises(RuntimeError, match="build_ext --inplace"):
        client._check_board_size()
    assert closed


def test_bot_cli_reads_value_perspective_from_training_config(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    monkeypatch.delenv("CARCASSONNE_AZ_VALUE_IS_CURRENT_PLAYER", raising=False)
    (tmp_path / "config.json").write_text('{"value_is_current_player": true}')

    cli = BotCliClient(env={"CARCASSONNE_AZ_PATH": str(tmp_path)})
    try:
        assert cli.request({"cmd": "info"})["value_is_current_player"] is True
    finally:
        cli.close()

    cli = BotCliClient(
        env={"CARCASSONNE_AZ_PATH": str(tmp_path), "CARCASSONNE_AZ_VALUE_IS_CURRENT_PLAYER": "false"}
    )
    try:
        assert cli.request({"cmd": "info"})["value_is_current_player"] is False
    finally:
        cli.close()


def test_summarize_ai_status_keeps_first_line_only() -> None:
    trace = "P2 az: open file failed, file path: /m/checkpoint--1.pt\nframe #0: c10::Error::Error(...)\nframe #1: ..."

    assert summarize_ai_status(trace) == "P2 az: open file failed, file path: /m/checkpoint--1.pt"
    assert summarize_ai_status("x" * 500).endswith("...")
    assert len(summarize_ai_status("x" * 500)) == 240


def test_meeple_buttons_cover_every_engine_position() -> None:
    labels = meeple_button_labels()

    assert sorted(labels) == list(range(MEEPLE_POS_INNER_FIELD + 1))
    assert [labels[MEEPLE_POS_FIELD + e] for e in (0, 3, 7)] == [
        "Farmer: Top-left",
        "Farmer: Right-bottom",
        "Farmer: Left-top",
    ]
    assert not any(is_farmer(pos) for pos in range(5))
    assert all(is_farmer(pos) for pos in range(MEEPLE_POS_FIELD, MEEPLE_POS_INNER_FIELD + 1))


def test_meeple_alignment_follows_sides_and_half_edges() -> None:
    assert [meeple_alignment(side) for side in range(4)] == [(0, -0.75), (0.75, 0), (0, 0.75), (-0.75, 0)]
    assert meeple_alignment(4) == (0, 0)
    assert meeple_alignment(MEEPLE_POS_INNER_FIELD) == (0, 0)
    expected = [(-0.5, -0.7), (0.5, -0.7), (0.7, -0.5), (0.7, 0.5), (0.5, 0.7), (-0.5, 0.7), (-0.7, 0.5), (-0.7, -0.5)]
    for half_edge in range(HALF_EDGE_COUNT):
        assert meeple_alignment(MEEPLE_POS_FIELD + half_edge) == pytest.approx(expected[half_edge])
