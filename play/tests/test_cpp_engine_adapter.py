from __future__ import annotations

import json

import pytest

from play import _carcassonne_cpp
from play.cpp_engine import (
    BOARD_SIZE,
    ENGINE_BOARD_SIZE,
    HALF_EDGE_COUNT,
    MEEPLE_POS_BIG,
    MEEPLE_POS_BUILDER,
    MEEPLE_POS_FIELD,
    MEEPLE_POS_INNER_FIELD,
    MEEPLE_POS_PIG,
    PHASE_TILE,
    START_POS,
    CppCarcassonneAdapter,
    PlayerSpec,
    meeple_piece,
    meeple_spot,
)
from play.engine import adapter as adapter_module
from play.engine.adapter import BotCliClient, expansion_masks, expansion_parameters, game_log_file
from pathlib import Path

from play.models import BotValue, Move, MoveRecord
from play.ui.app import (
    BIG_MEEPLE_SIZE,
    MEEPLE_SIZE,
    build_player_specs,
    format_bot_value,
    format_goods,
    format_move_record,
    human_seat,
    is_farmer,
    meeple_alignment,
    meeple_button_labels,
    meeple_marker,
    offered_pieces,
    parse_ui_config,
    pieces_in_hand,
    should_show_start_game,
    summarize_ai_status,
    view_shift_pan,
)
from play.ui.app import CELL_SIZE


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

    # 202 spatial planes and one global plane; seven actions per cell (four tile
    # rotations, a magic portal's cell, the princess's cell, the fairy's cell),
    # 41 meeple moves and 4 dragon steps. The observation and the actions by cell
    # cover the view, not the whole board.
    assert response["observation_shape"] == [203, BOARD_SIZE, BOARD_SIZE]
    assert response["observation_tensor_size"] == 203 * BOARD_SIZE * BOARD_SIZE
    assert response["num_distinct_actions"] == BOARD_SIZE * BOARD_SIZE * 7 + 41 + 4


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


def test_meeple_action_strings_match_the_actor_logs() -> None:
    assert adapter_module._meeple_action_string(-1) == "place_meeple(skip)"
    assert adapter_module._meeple_action_string(2) == "place_meeple(edge=2)"
    assert adapter_module._meeple_action_string(4) == "place_meeple(monastery)"
    assert adapter_module._meeple_action_string(MEEPLE_POS_FIELD + 3) == "place_meeple(field=3)"
    assert adapter_module._meeple_action_string(MEEPLE_POS_INNER_FIELD) == "place_meeple(inner_field)"
    big, builder, pig = adapter_module.MEEPLE_POS_BIG, adapter_module.MEEPLE_POS_BUILDER, adapter_module.MEEPLE_POS_PIG
    assert adapter_module._meeple_action_string(big + 1) == "place_big_meeple(edge=1)"
    assert adapter_module._meeple_action_string(big + 4) == "place_big_meeple(monastery)"
    assert adapter_module._meeple_action_string(big + MEEPLE_POS_FIELD + 2) == "place_big_meeple(field=2)"
    assert adapter_module._meeple_action_string(builder + 3) == "place_builder(edge=3)"
    assert adapter_module._meeple_action_string(pig + 5) == "place_pig(field=5)"


def test_adapter_saves_a_finished_game(_game_log_dir) -> None:
    adapter = CppCarcassonneAdapter(
        seed=7,
        player_specs=(PlayerSpec(type="random"), PlayerSpec(type="random")),
    )
    try:
        assert adapter.state.game_over
        assert adapter.save_error == ""
        path = adapter.saved_game_path
        assert path is not None and path.parent == _game_log_dir
        record = json.loads(path.read_text(encoding="utf-8"))
        assert record["seed"] == 7
        assert [p["type"] for p in record["players"]] == ["random", "random"]
        assert record["scores"] == [adapter.state.scores[1], adapter.state.scores[2]]
        assert record["turns"][-1]["scores"] == record["scores"]
        assert len(record["turns"]) == len(adapter.move_records)

        actions = record["actions"]
        draws = [a for a in actions if a.startswith("draw_type(")]
        tiles = [a for a in actions if a.startswith("place_tile(")]
        meeples = [a for a in actions if a.startswith("place_meeple(")]
        assert len(draws) + len(tiles) + len(meeples) == len(actions)
        assert len(tiles) == len(meeples) == len(record["turns"])
        assert len(draws) >= len(tiles)

        lines = (_game_log_dir / adapter_module.GAME_LOG_FILE).read_text(encoding="utf-8").splitlines()
        returns = record["returns"]
        assert len(lines) == 1 and record["log_game"] == 1
        assert lines[0].startswith("[")
        assert f"] Game 1: Returns: {returns[0]} {returns[1]}; Actions: " in lines[0]
        assert lines[0].endswith("; Actions: " + " ".join(actions))
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

    def __init__(self, path=None, env=None, expansion_masks=None):
        self.env = env or {}
        if expansion_masks is None:
            self.mirror = _carcassonne_cpp.Carcassonne()
        else:
            # Like the real bot CLI, play the same expansions as the UI.
            self.mirror = _carcassonne_cpp.Carcassonne(expansions=expansion_masks[0], rules=expansion_masks[1])

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


def test_ui_follows_the_engine_view() -> None:
    adapter = CppCarcassonneAdapter(seed=7)
    assert BOARD_SIZE < ENGINE_BOARD_SIZE
    # The UI shows the engine's view, which holds every legal move: each one, on
    # the view's edge cells too, is offered to the human for a whole game.
    origins = set()
    while not adapter.state.game_over:
        assert adapter.view_origin == tuple(adapter._engine.view_origin)
        origins.add(adapter.view_origin)
        engine_moves = sorted(adapter._engine.get_legal_tile_moves())
        ui_moves = sorted((*adapter.to_engine_coords(move.x, move.y), move.rotation) for move in adapter.get_valid_moves())
        assert ui_moves == engine_moves
        adapter.confirm_tile(adapter.get_valid_moves()[-1])
        adapter.apply_meeple(-1)
    assert len(origins) > 1


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
    client.close = lambda: closed.append(True)

    with pytest.raises(RuntimeError, match="build_ext --inplace"):
        client._check_board_size({"ok": True, "observation_shape": [80, 15, 15]})
    assert closed


def test_bot_cli_rejects_binary_that_ignores_the_expansions() -> None:
    client = BotCliClient.__new__(BotCliClient)
    closed = []
    client.close = lambda: closed.append(True)
    river = expansion_masks({"river": "on"})

    # A bot CLI built before CARCASSONNE_EXPANSIONS reports no masks: fine for a base game only.
    client._check_expansions({"ok": True}, expansion_masks({}))
    with pytest.raises(RuntimeError, match="did not pick up the expansions"):
        client._check_expansions({"ok": True}, river)
    with pytest.raises(RuntimeError, match="did not pick up the expansions"):
        client._check_expansions({"ok": True, "expansions": 1, "rules": 0}, river)
    assert len(closed) == 2


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

    assert sorted(labels) == list(range(MEEPLE_POS_PIG + HALF_EDGE_COUNT))
    # A tab per piece: meeple and big meeple on every spot, builder on sides, pig on fields.
    pieces = [meeple_piece(pos) for pos in sorted(labels)]
    assert [pieces.count(kind) for kind in ("meeple", "big", "builder", "pig")] == [14, 14, 4, 8]
    assert [labels[pos] for pos in (0, 3, 4)] == ["Up", "Left", "Center"]
    assert [labels[MEEPLE_POS_FIELD + e] for e in (0, 3, 7)] == [
        "Farmer: Top-left",
        "Farmer: Right-bottom",
        "Farmer: Left-top",
    ]
    assert [labels[MEEPLE_POS_BIG + pos] for pos in range(MEEPLE_POS_INNER_FIELD + 1)] == [
        labels[pos] for pos in range(MEEPLE_POS_INNER_FIELD + 1)
    ]
    assert [labels[MEEPLE_POS_BUILDER + side] for side in range(4)] == ["Up", "Right", "Down", "Left"]
    assert [labels[MEEPLE_POS_PIG + e] for e in (0, 7)] == ["Field: Top-left", "Field: Left-top"]
    assert not any(is_farmer(pos) for pos in range(5))
    assert all(is_farmer(pos) for pos in range(MEEPLE_POS_FIELD, MEEPLE_POS_INNER_FIELD + 1))
    assert is_farmer(MEEPLE_POS_BIG + MEEPLE_POS_FIELD) and not is_farmer(MEEPLE_POS_BIG + 1)
    assert not any(is_farmer(pos) for pos in range(MEEPLE_POS_BUILDER, MEEPLE_POS_PIG + HALF_EDGE_COUNT))


def test_meeple_markers_pick_each_piece_image() -> None:
    assert meeple_marker(2) == ("standing", MEEPLE_SIZE)
    assert meeple_marker(MEEPLE_POS_FIELD + 3) == ("farmer", MEEPLE_SIZE)
    assert meeple_marker(MEEPLE_POS_BIG + 4) == ("standing", BIG_MEEPLE_SIZE)
    assert meeple_marker(MEEPLE_POS_BIG + MEEPLE_POS_INNER_FIELD) == ("farmer", BIG_MEEPLE_SIZE)
    assert meeple_marker(MEEPLE_POS_BUILDER + 1) == ("builder", MEEPLE_SIZE)
    assert meeple_marker(MEEPLE_POS_PIG + 7) == ("pig", MEEPLE_SIZE)
    for image in ("standing", "farmer", "builder", "pig"):
        for owner in (1, 2):
            assert (adapter_module.REPO_ROOT / "meeples" / f"{image}_p{owner}.png").exists()


def test_pieces_in_hand_and_goods_text() -> None:
    assert pieces_in_hand(7, {}) == "■" * 7
    assert pieces_in_hand(3, {"big": 1, "builder": 0, "pig": 1}) == "■■■ ◆●"
    assert pieces_in_hand(0, {"big": 1, "builder": 1}) == "◆▲"
    assert pieces_in_hand(0, {"big": 0}) == "-"
    assert format_goods((2, 0, 1)) == "wine 2 · wheat 0 · cloth 1"


def test_offered_pieces_name_the_tabs_with_a_move() -> None:
    assert offered_pieces([-1]) == []
    assert offered_pieces([-1, MEEPLE_POS_PIG + 1, 2, MEEPLE_POS_BIG + 2]) == ["meeple", "big", "pig"]
    assert offered_pieces([MEEPLE_POS_BUILDER]) == ["builder"]


def test_meeple_alignment_follows_sides_and_half_edges() -> None:
    assert [meeple_alignment(side) for side in range(4)] == [(0, -0.75), (0.75, 0), (0, 0.75), (-0.75, 0)]
    assert meeple_alignment(4) == (0, 0)
    assert meeple_alignment(MEEPLE_POS_INNER_FIELD) == (0, 0)
    expected = [(-0.5, -0.7), (0.5, -0.7), (0.7, -0.5), (0.7, 0.5), (0.5, 0.7), (-0.5, 0.7), (-0.7, 0.5), (-0.7, -0.5)]
    for half_edge in range(HALF_EDGE_COUNT):
        assert meeple_alignment(MEEPLE_POS_FIELD + half_edge) == pytest.approx(expected[half_edge])


RIVER_TYPES = range(25, 35)
RIVER_LAKE_TYPE = 26


def _expansion_bit(name: str) -> int:
    return int(_carcassonne_cpp.expansion_bit(list(_carcassonne_cpp.EXPANSION_NAMES).index(name)))


def test_expansion_masks_follow_the_game_parameters() -> None:
    base = int(_carcassonne_cpp.BASE_ONLY)
    river = _expansion_bit("river")
    inns = _expansion_bit("inns_cathedrals")

    assert expansion_masks({}) == (base, 0)
    assert expansion_masks({"river": "on"}) == (base | river, river)
    assert expansion_masks({"inns_cathedrals": "tiles"}) == (base | inns, 0)
    assert expansion_masks({"river": "off"}) == (base, 0)
    for bad in ({"river": "tiles"}, {"river": "x"}, {"lakes": "on"}):
        with pytest.raises(ValueError):
            expansion_masks(bad)
    assert expansion_parameters({"river": "on"}) == "river=on"
    assert game_log_file({}) == "log-actor-gui.txt"
    assert game_log_file({"river": "on"}) == "log-actor-gui-river.txt"


def _river_game(seed: int) -> CppCarcassonneAdapter:
    return CppCarcassonneAdapter(seed=seed, expansions={"river": "on"})


def test_river_game_starts_at_the_spring_and_lays_the_river_first() -> None:
    adapter = _river_game(5)
    assert adapter.expansions == {"river": "on"}
    assert adapter.state.board[START_POS].tile_id == 25  # the spring replaces the start tile
    assert "River: on" in adapter.mode_label()

    while len(adapter.move_records) < 12 and not adapter.state.game_over:
        adapter.confirm_tile(adapter.get_valid_moves()[0])
        adapter.apply_meeple(-1)

    placed = [record.tile_id for record in reversed(adapter.move_records)]
    # 12 river tiles with the spring: the other 11 come first and the lake closes them.
    assert all(tile in RIVER_TYPES for tile in placed[:11])
    assert placed[10] == RIVER_LAKE_TYPE
    assert placed[11] not in RIVER_TYPES
    adapter.close()


def test_river_game_plays_to_the_end() -> None:
    import random

    rng = random.Random(7)
    adapter = _river_game(7)
    while not adapter.state.game_over:
        options = adapter.confirm_tile(rng.choice(adapter.get_valid_moves()))
        adapter.apply_meeple(rng.choice(options))
    assert adapter.state.game_over
    assert sum(adapter.state.scores.values()) > 0
    adapter.close()


def test_bot_cli_plays_with_the_river() -> None:
    masks = expansion_masks({"river": "on"})
    engine = _carcassonne_cpp.Carcassonne(expansions=masks[0], rules=masks[1])
    cli = BotCliClient(env={"CARCASSONNE_EXPANSIONS": "river=on"}, expansion_masks=masks)
    try:
        info = cli.request({"cmd": "info"})
        assert (info["expansions"], info["rules"]) == masks
        draw_type = list(engine.get_available_draws())[0][0]
        assert draw_type in RIVER_TYPES
        engine.draw_tile(draw_type)
        cli.request({"cmd": "apply_draw", "type": draw_type})
        response = cli.request({"cmd": "choose", "bot": "random", "seed": 1})
    finally:
        cli.close()

    assert response["kind"] == "tile"
    assert (response["x"], response["y"], response["rot"]) in set(engine.get_legal_tile_moves())


def test_bot_cli_reports_a_base_game_without_expansions() -> None:
    cli = BotCliClient(expansion_masks=expansion_masks({}))
    try:
        info = cli.request({"cmd": "info"})
    finally:
        cli.close()
    assert (info["expansions"], info["rules"]) == expansion_masks({})


def test_ui_parser_takes_the_river() -> None:
    assert parse_ui_config([]).expansions == {}
    assert parse_ui_config(["--river=on"]).expansions == {"river": "on"}
    with pytest.raises(SystemExit):
        parse_ui_config(["--river=tiles"])


def test_view_shift_pan_moves_the_content_back_by_the_shift() -> None:
    assert view_shift_pan((10, 10), (11, 9)) == (CELL_SIZE, -CELL_SIZE)
    assert view_shift_pan((10, 10), (10, 10)) == (0, 0)


def test_river_game_view_follows_the_tiles(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(adapter_module, "BotCliClient", FakeBotCli)
    adapter = CppCarcassonneAdapter(
        seed=11,
        player_specs=(PlayerSpec(type="human"), PlayerSpec(type="random")),
        expansions={"river": "on"},
    )
    spring = START_POS
    shifts = 0
    while len(adapter.move_records) < 16 and not adapter.state.game_over:
        before = adapter.view_origin
        spring_before = (spring[0] - before[0], spring[1] - before[1])
        adapter.confirm_tile(adapter.get_valid_moves()[0])
        adapter.apply_meeple(-1)  # the random bot replies on its own
        after = adapter.view_origin
        # The UI's view is the engine's, and every tile is inside it.
        assert after == tuple(adapter._engine.view_origin)
        for x, y in adapter.state.board:
            assert 0 <= x - after[0] < BOARD_SIZE and 0 <= y - after[1] < BOARD_SIZE
        if after != before:
            shifts += 1
            # The same tile's grid cell moves by minus the shift; view_shift_pan undoes it.
            dx, dy = after[0] - before[0], after[1] - before[1]
            assert (spring[0] - after[0], spring[1] - after[1]) == (spring_before[0] - dx, spring_before[1] - dy)
            assert view_shift_pan(before, after) == (dx * CELL_SIZE, dy * CELL_SIZE)
    assert shifts > 0  # the river carries the tiles away from the spring
    adapter.close()


ALL_EXPANSIONS = {"inns_cathedrals": "on", "traders_builders": "on", "river": "on", "princess_dragon": "tiles"}
PIECES_PER_PLAYER = {"meeple": 7, "big": 1, "builder": 1, "pig": 1}


def _native_game(modes: dict) -> _carcassonne_cpp.Carcassonne:
    masks = expansion_masks(modes)
    return _carcassonne_cpp.Carcassonne(expansions=masks[0], rules=masks[1])


def test_native_binding_reports_the_expansion_pieces() -> None:
    base = _carcassonne_cpp.Carcassonne()
    assert (base.holding_big_meeples, base.holding_builders, base.holding_pigs) == ([0, 0], [0, 0], [0, 0])
    assert not (base.big_meeple_rules or base.builder_rules or base.pig_rules or base.goods_rules)

    inns = _native_game({"inns_cathedrals": "on"})
    assert inns.big_meeple_rules and inns.holding_big_meeples == [1, 1]
    assert _native_game({"inns_cathedrals": "tiles"}).holding_big_meeples == [0, 0]

    traders = _native_game({"traders_builders": "on"})
    assert traders.builder_rules and traders.pig_rules and traders.goods_rules
    assert (traders.holding_builders, traders.holding_pigs) == ([1, 1], [1, 1])
    assert traders.goods_tokens == [[0, 0, 0], [0, 0, 0]]
    assert not traders.builder_second_tile
    assert not _native_game({"traders_builders": "tiles"}).builder_rules

    moves = (-1, 3, MEEPLE_POS_BIG + 4, MEEPLE_POS_BIG + MEEPLE_POS_INNER_FIELD, MEEPLE_POS_BUILDER + 2, MEEPLE_POS_PIG + 5)
    assert [meeple_spot(pos) for pos in moves] == [-1, 3, 4, MEEPLE_POS_INNER_FIELD, 2, MEEPLE_POS_FIELD + 5]


def test_adapter_draws_expansion_pieces_while_they_are_on_the_board() -> None:
    sx, sy = START_POS
    tiles = [(sx + dx, sy) for dx in range(4)]
    big_on_city = MEEPLE_POS_BIG + 1
    builder = MEEPLE_POS_BUILDER + 3
    pig = MEEPLE_POS_PIG + 2
    big_farmer = MEEPLE_POS_BIG + MEEPLE_POS_FIELD + 6
    adapter = CppCarcassonneAdapter(seed=42)
    adapter.move_records = [
        _meeple_record(1, sx, sy, big_on_city),
        _meeple_record(1, sx + 1, sy, builder),
        _meeple_record(2, sx + 2, sy, pig),
        _meeple_record(2, sx + 3, sy, big_farmer),
    ]

    # The big meeple's city and the builder's road were completed: both went home.
    adapter._engine = FakeMeepleEngine(tiles, [])
    board = adapter._build_board()
    assert {pos: tile.meeple_markers for pos, tile in board.items() if tile.meeple_markers} == {
        (sx + 2, sy): [(2, pig)],  # pigs and farmers stay till the end
        (sx + 3, sy): [(2, big_farmer)],
    }

    # Still open: the big meeple's side and the builder's side hold their owner's tokens.
    adapter._engine = FakeMeepleEngine(tiles, [(0, sx, sy, 1), (0, sx + 1, sy, 3)])
    board = adapter._build_board()
    assert board[(sx, sy)].meeple_markers == [(1, big_on_city)]
    assert board[(sx + 1, sy)].meeple_markers == [(1, builder)]


def _markers_by_piece(state) -> dict:
    counts = {(player, kind): 0 for player in (1, 2) for kind in PIECES_PER_PLAYER}
    for tile in state.board.values():
        for owner, pos in tile.meeple_markers:
            counts[(owner, meeple_piece(pos))] += 1
    return counts


def test_expansion_games_show_every_piece_until_it_comes_back() -> None:
    import random

    placed = set()
    double_turns = 0
    for seed in range(5):
        rng = random.Random(seed)
        adapter = CppCarcassonneAdapter(seed=seed, expansions=ALL_EXPANSIONS)
        assert adapter.state.piece_kinds == ("big", "builder", "pig")
        while not adapter.state.game_over:
            player = adapter.state.current_player
            options = adapter.confirm_tile(rng.choice(adapter.get_valid_moves()))
            # Half the time play an expansion's piece when one fits, so every game uses them.
            special = [pos for pos in options if pos != -1 and meeple_piece(pos) != "meeple"]
            pos = rng.choice(special) if special and rng.random() < 0.5 else rng.choice(options)
            adapter.apply_meeple(pos)
            if pos != -1:
                placed.add(meeple_piece(pos))
            state = adapter.state
            if state.game_over:
                break
            # Only the builder gives a player a second tile.
            assert state.builder_extra_tile == (state.current_player == player)
            double_turns += state.builder_extra_tile
            # Every piece out of hand is drawn on the board, and only those.
            markers = _markers_by_piece(state)
            for owner in (1, 2):
                assert markers[(owner, "meeple")] == PIECES_PER_PLAYER["meeple"] - state.meeples_remaining[owner]
                for kind in ("big", "builder", "pig"):
                    assert markers[(owner, kind)] == PIECES_PER_PLAYER[kind] - state.pieces_remaining[owner][kind]
            assert set(state.goods) == {1, 2}
        adapter.close()
    assert placed == {"meeple", "big", "builder", "pig"}
    assert double_turns > 0


def test_tiles_only_expansions_have_no_extra_pieces() -> None:
    adapter = CppCarcassonneAdapter(
        seed=3, expansions={"inns_cathedrals": "tiles", "traders_builders": "tiles", "princess_dragon": "tiles"}
    )
    assert adapter.state.piece_kinds == ()
    assert adapter.state.goods == {}
    options = adapter.confirm_tile(adapter.get_valid_moves()[0])
    assert all(meeple_piece(pos) == "meeple" for pos in options)
    adapter.close()


def test_ui_parser_takes_every_expansion() -> None:
    argv = ["--inns_cathedrals=on", "--traders_builders=tiles", "--river=on", "--princess_dragon=tiles"]
    assert parse_ui_config(argv).expansions == {
        "inns_cathedrals": "on",
        "traders_builders": "tiles",
        "river": "on",
        "princess_dragon": "tiles",
    }
    with pytest.raises(SystemExit):
        parse_ui_config(["--princess_dragon=on"])  # the UI does not play the dragon yet


def test_bot_cli_plays_a_game_with_every_expansion() -> None:
    adapter = CppCarcassonneAdapter(
        seed=4,
        player_specs=(PlayerSpec(type="random"), PlayerSpec(type="random")),
        expansions=ALL_EXPANSIONS,
    )
    try:
        assert set(adapter._bot_clis) == {0, 1}  # both bots started, with matching expansions
        assert adapter.state.game_over, adapter.ai_status
        pieces = {meeple_piece(record.meeple_pos) for record in adapter.move_records if record.meeple_pos != -1}
        assert pieces == {"meeple", "big", "builder", "pig"}
    finally:
        adapter.close()
