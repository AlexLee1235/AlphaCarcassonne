from __future__ import annotations

import json
import random
import secrets
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
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
OPPONENT_MODES = {"player", "random", "mcts", "alphazero", "az"}
PLAYER_TYPES = {"human", "random", "mcts", "alphazero", "az"}
BOT_TYPES = {"random", "mcts", "alphazero"}
DEFAULT_MAX_SIMULATIONS = 200
DEFAULT_BOT_CLI = Path(__file__).resolve().parents[1] / "bin" / "carcassonne_bot_cli"


def _clamp(value: int, lower: int, upper: int) -> int:
    return max(lower, min(upper, value))


def _physical_to_art_id(physical_id: int) -> int:
    if physical_id <= 0:
        return 0
    return PHYSICAL_TO_CANONICAL_TYPE[physical_id]


def _normalize_player_type(player_type: str) -> str:
    mode = player_type.strip().lower()
    if mode == "player":
        mode = "human"
    if mode == "az":
        mode = "alphazero"
    if mode not in {"human", "random", "mcts", "alphazero"}:
        raise ValueError(f"Unknown player type: {player_type}")
    return mode


@dataclass(frozen=True)
class PlayerSpec:
    type: str = "human"
    az_path: str = ""
    az_checkpoint: Optional[int] = None
    az_graph_def: str = "vpnet.pb"
    max_simulations: Optional[int] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "type", _normalize_player_type(self.type))
        if self.max_simulations is not None and self.max_simulations < 1:
            raise ValueError("max_simulations must be positive.")

    @property
    def is_human(self) -> bool:
        return self.type == "human"

    @property
    def is_bot(self) -> bool:
        return self.type in BOT_TYPES

    @property
    def label(self) -> str:
        return "az" if self.type == "alphazero" else self.type

    def bot_env(self) -> Dict[str, str]:
        env: Dict[str, str] = {}
        if self.az_path:
            env["CARCASSONNE_AZ_PATH"] = self.az_path
        if self.az_checkpoint is not None:
            env["CARCASSONNE_AZ_CHECKPOINT"] = str(self.az_checkpoint)
        if self.az_graph_def:
            env["CARCASSONNE_AZ_GRAPH_DEF"] = self.az_graph_def
        if self.max_simulations is not None:
            if self.type == "alphazero":
                env["CARCASSONNE_AZ_SIMULATIONS"] = str(self.max_simulations)
            elif self.type == "mcts":
                env["CARCASSONNE_MCTS_SIMULATIONS"] = str(self.max_simulations)
        return env


class BotCliClient:
    def __init__(self, path: Optional[str] = None, env: Optional[Dict[str, str]] = None):
        cli_path = Path(path or os.getenv("CARCASSONNE_BOT_CLI", str(DEFAULT_BOT_CLI)))
        if not cli_path.exists():
            raise RuntimeError(
                f"Carcassonne bot CLI not found at {cli_path}. Build it with `python play/setup.py build_ext --inplace`."
            )
        process_env = os.environ.copy()
        if env:
            process_env.update(env)
        self._proc = subprocess.Popen(
            [str(cli_path)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            env=process_env,
        )
        self.request({"cmd": "reset"})

    def close(self) -> None:
        proc = self._proc
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=1.0)

    def request(self, payload: dict) -> dict:
        if self._proc.poll() is not None:
            stderr = self._proc.stderr.read() if self._proc.stderr is not None else ""
            raise RuntimeError(f"Carcassonne bot CLI exited unexpectedly. {stderr}".strip())
        if self._proc.stdin is None or self._proc.stdout is None:
            raise RuntimeError("Carcassonne bot CLI pipes are unavailable.")
        self._proc.stdin.write(json.dumps(payload, separators=(",", ":")) + "\n")
        self._proc.stdin.flush()
        line = self._proc.stdout.readline()
        if not line:
            stderr = self._proc.stderr.read() if self._proc.stderr is not None else ""
            raise RuntimeError(f"Carcassonne bot CLI returned no response. {stderr}".strip())
        response = json.loads(line)
        if not response.get("ok", False):
            raise RuntimeError(str(response.get("error", "Unknown Carcassonne bot CLI error.")))
        return response


class CppCarcassonneAdapter:
    def __init__(
        self,
        seed: Optional[int] = None,
        opponent_mode: str = "player",
        player_specs: Optional[Tuple[PlayerSpec, PlayerSpec]] = None,
    ):
        if seed is None:
            seed = secrets.randbits(32)
        self._rng = random.Random(seed)
        self.player_specs = self._resolve_player_specs(opponent_mode, player_specs)
        self.opponent_mode = "player" if self.player_specs[1].is_human else self.player_specs[1].type
        self.ai_status = ""
        self._engine = _carcassonne_cpp.Carcassonne()
        self._bot_clis: Dict[int, BotCliClient] = {}
        self._latest_tile_marker: Optional[Tuple[Tuple[int, int], int]] = None
        self._turn = 1
        self._pending_meeple_options: List[int] = []
        self._viewport_origin = self._default_viewport_origin()
        self._start_bot_clis()
        self._resolve_chance_phase()
        self.state = self._build_state()
        self.run_ai_turns()

    def close(self) -> None:
        for client in self._bot_clis.values():
            client.close()
        self._bot_clis = {}

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def _resolve_player_specs(
        self,
        opponent_mode: str,
        player_specs: Optional[Tuple[PlayerSpec, PlayerSpec]],
    ) -> Tuple[PlayerSpec, PlayerSpec]:
        if player_specs is not None:
            if len(player_specs) != 2:
                raise ValueError("player_specs must contain exactly two players.")
            return (
                PlayerSpec(**player_specs[0].__dict__),
                PlayerSpec(**player_specs[1].__dict__),
            )

        mode = self._normalize_opponent_mode(opponent_mode)
        p2_type = "human" if mode == "player" else mode
        return PlayerSpec("human"), PlayerSpec(p2_type)

    def _normalize_opponent_mode(self, opponent_mode: str) -> str:
        mode = opponent_mode.strip().lower()
        if mode == "az":
            mode = "alphazero"
        if mode not in OPPONENT_MODES:
            raise ValueError(f"Unknown opponent mode: {opponent_mode}")
        return mode

    def _next_seed(self) -> int:
        return self._rng.randrange(1, 2**31)

    def _mcts_simulations(self) -> int:
        value = int(os.getenv("CARCASSONNE_MCTS_SIMULATIONS", "200"))
        return max(1, value)

    def _az_simulations(self) -> int:
        value = os.getenv("CARCASSONNE_AZ_SIMULATIONS", os.getenv("CARCASSONNE_MCTS_SIMULATIONS", "200"))
        return max(1, int(value))

    def _simulations_for(self, spec: PlayerSpec) -> int:
        if spec.max_simulations is not None:
            return spec.max_simulations
        if spec.type == "alphazero":
            return self._az_simulations()
        if spec.type == "mcts":
            return self._mcts_simulations()
        return DEFAULT_MAX_SIMULATIONS

    def _start_bot_clis(self) -> None:
        for player in range(2):
            if self.player_specs[player].is_bot:
                self._start_bot_cli(player)

    def _start_bot_cli(self, player: int) -> None:
        if player in self._bot_clis:
            return
        spec = self.player_specs[player]
        if not spec.is_bot:
            return
        try:
            self._bot_clis[player] = BotCliClient(env=spec.bot_env())
        except Exception as exc:
            self.ai_status = f"P{player + 1} {spec.label}: {exc}"

    def _bot_request(self, player: int, payload: dict) -> dict:
        if player not in self._bot_clis:
            self._start_bot_cli(player)
        client = self._bot_clis.get(player)
        if client is None:
            raise RuntimeError(self.ai_status or "Carcassonne bot CLI is unavailable.")
        return client.request(payload)

    def _sync_bots(self, payload: dict) -> None:
        for player, spec in enumerate(self.player_specs):
            if not spec.is_bot:
                continue
            try:
                self._bot_request(player, payload)
            except Exception as exc:
                self.ai_status = f"P{player + 1} {spec.label}: {exc}"

    def has_bot_players(self) -> bool:
        return any(spec.is_bot for spec in self.player_specs)

    def controller_label(self, player: int) -> str:
        return self.player_specs[player - 1].label

    def mode_label(self) -> str:
        return f"P1 {self.controller_label(1)} vs P2 {self.controller_label(2)}"

    def _current_player_index(self) -> int:
        return int(self._engine.current_player)

    def _current_player_spec(self) -> PlayerSpec:
        return self.player_specs[self._current_player_index()]

    def is_ai_turn(self) -> bool:
        if self._engine.is_game_over or self._pending_meeple_options:
            return False
        return self._current_player_spec().is_bot

    def _choose_payload(self, spec: PlayerSpec) -> dict:
        payload = {"cmd": "choose", "bot": spec.type, "seed": self._next_seed()}
        if spec.type in {"mcts", "alphazero"}:
            payload["simulations"] = self._simulations_for(spec)
        return payload

    def _choose_bot_tile_move(self, player: int) -> Tuple[int, int, int]:
        spec = self.player_specs[player]
        if not spec.is_bot:
            raise ValueError(f"P{player + 1} is not a bot.")
        response = self._bot_request(player, self._choose_payload(spec))
        if response.get("kind") != "tile":
            raise RuntimeError(f"Expected tile action from bot CLI, got {response.get('kind')}.")
        return int(response["x"]), int(response["y"]), int(response["rot"])

    def _choose_bot_meeple_move(self, player: int) -> int:
        spec = self.player_specs[player]
        if not spec.is_bot:
            raise ValueError(f"P{player + 1} is not a bot.")
        response = self._bot_request(player, self._choose_payload(spec))
        if response.get("kind") != "meeple":
            raise RuntimeError(f"Expected meeple action from bot CLI, got {response.get('kind')}.")
        return int(response["pos"])

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
            draw_type = self._sample_draw_type(draws)
            self._engine.draw_tile(draw_type)
            self._sync_bots({"cmd": "apply_draw", "type": draw_type})

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
        if self.is_ai_turn() or self._pending_meeple_options or self._engine.current_phase != PHASE_TILE:
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
        if self.is_ai_turn():
            raise ValueError("It is the AI player's turn.")

        engine_x, engine_y = self.to_engine_coords(move.x, move.y)
        legal = {(x, y, r) for (x, y, r) in self._engine.get_legal_tile_moves()}
        if (engine_x, engine_y, move.rotation) not in legal:
            raise ValueError(f"Invalid move: ({move.x}, {move.y}, r={move.rotation})")

        self._engine.place_tile(engine_x, engine_y, move.rotation)
        self._latest_tile_marker = ((engine_x, engine_y), self._engine.current_player + 1)
        self._sync_bots({"cmd": "apply_tile", "x": engine_x, "y": engine_y, "rot": move.rotation})
        self._pending_meeple_options = list(self._engine.get_legal_meeple_moves())
        self.state = self._build_state()
        return list(self._pending_meeple_options)

    def apply_meeple(self, meeple_pos: int) -> None:
        if meeple_pos not in self._pending_meeple_options:
            raise ValueError(f"Invalid meeple position: {meeple_pos}")

        self._engine.place_meeple(meeple_pos)
        self._sync_bots({"cmd": "apply_meeple", "pos": meeple_pos})
        self._pending_meeple_options = []
        self._turn += 1
        self._resolve_chance_phase()
        self.run_ai_turns()
        self.state = self._build_state()

    def run_ai_turns(self) -> None:
        self.ai_status = ""
        if not self.has_bot_players():
            return

        ai_turns = 0
        last_label = ""
        try:
            while not self._engine.is_game_over and self._current_player_spec().is_bot:
                player = self._current_player_index()
                spec = self.player_specs[player]
                last_label = f"P{player + 1} {spec.label}"
                if self._engine.current_phase == PHASE_CHANCE:
                    self._resolve_chance_phase()
                    continue
                if self._engine.current_phase == PHASE_TILE:
                    x, y, rotation = self._choose_bot_tile_move(player)
                    self._engine.place_tile(x, y, rotation)
                    self._latest_tile_marker = ((x, y), self._engine.current_player + 1)
                    self._sync_bots({"cmd": "apply_tile", "x": x, "y": y, "rot": rotation})
                    continue
                if self._engine.current_phase == PHASE_MEEPLE:
                    meeple_pos = self._choose_bot_meeple_move(player)
                    self._engine.place_meeple(meeple_pos)
                    self._sync_bots({"cmd": "apply_meeple", "pos": meeple_pos})
                    ai_turns += 1
                    self._turn += 1
                    self._resolve_chance_phase()
                    continue
                break
        except Exception as exc:
            self.ai_status = str(exc)
            self.state = self._build_state()
            return

        if ai_turns:
            self.ai_status = f"{last_label} played {ai_turns} bot turn(s)."
        self.state = self._build_state()

    def _build_board(self) -> Dict[Tuple[int, int], PlacedTile]:
        board: Dict[Tuple[int, int], PlacedTile] = {}
        latest_pos = self._latest_tile_marker[0] if self._latest_tile_marker is not None else None
        latest_owner = self._latest_tile_marker[1] if self._latest_tile_marker is not None else None
        for x, y, physical_id, rotation in self._engine.get_placed_tiles():
            board[(x, y)] = PlacedTile(
                tile_id=_physical_to_art_id(physical_id),
                rotation=rotation,
                tile_owner=latest_owner if (x, y) == latest_pos else None,
            )
        for player, x, y, pos in self._engine.get_meeple_tokens():
            tile = board.get((x, y))
            if tile is None:
                raise RuntimeError(f"Native meeple token ({player}, {x}, {y}, {pos}) has no matching tile snapshot")
            owner = 0 if player < 0 else player + 1
            tile.meeple_markers.append((owner, pos))
        for tile in board.values():
            self._merge_contested_meeple_markers(tile)
        return board

    def _merge_contested_meeple_markers(self, tile: PlacedTile) -> None:
        if not tile.meeple_markers:
            return

        owners_by_pos: Dict[int, set[int]] = {}
        for owner, pos in tile.meeple_markers:
            owners_by_pos.setdefault(pos, set()).add(owner)

        merged: List[Tuple[int, int]] = []
        for pos in sorted(owners_by_pos):
            owners = owners_by_pos[pos]
            if 0 in owners or (1 in owners and 2 in owners):
                merged.append((0, pos))
            else:
                merged.extend((owner, pos) for owner in sorted(owners))

        tile.meeple_markers = merged
        tile.meeple_owner, tile.meeple_pos = merged[0]

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
