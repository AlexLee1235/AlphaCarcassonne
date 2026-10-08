from __future__ import annotations

import json
import random
import secrets
import os
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    from domain import BotValue, GameState, Move, MoveRecord, PlacedTile
except ImportError:  # pragma: no cover - package import fallback
    from ..domain import BotValue, GameState, Move, MoveRecord, PlacedTile

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


ENGINE_BOARD_SIZE = int(_carcassonne_cpp.BOARD_SIZE)
# The UI shows the engine's view, which follows the tiles and holds every tile
# and every legal move: UI coordinates are engine coordinates less its origin.
BOARD_SIZE = int(_carcassonne_cpp.VIEW_SIZE)
START_POS = (ENGINE_BOARD_SIZE // 2, ENGINE_BOARD_SIZE // 2)
PHASE_CHANCE = int(_carcassonne_cpp.PHASE_CHANCE)
PHASE_TILE = int(_carcassonne_cpp.PHASE_TILE)
PHASE_MEEPLE = int(_carcassonne_cpp.PHASE_MEEPLE)
PHASE_TERMINAL = int(_carcassonne_cpp.PHASE_TERMINAL)
# The Princess & the Dragon: the dragon's steps, and the second step of a choice by cell.
PHASE_DRAGON = int(_carcassonne_cpp.PHASE_DRAGON)
PHASE_SPOT = int(_carcassonne_cpp.PHASE_SPOT)
# The phases that are part of a player's turn after its tile.
PIECE_PHASES = (PHASE_MEEPLE, PHASE_SPOT, PHASE_DRAGON)
# The choices by cell, in the engine's SpotChoice order, as the bot CLI names them.
SPOT_CHOICES = ("portal", "princess", "fairy")
assert [int(_carcassonne_cpp.SPOT_PORTAL), int(_carcassonne_cpp.SPOT_PRINCESS), int(_carcassonne_cpp.SPOT_FAIRY)] == [0, 1, 2]
DRAGON_STEPS = int(_carcassonne_cpp.DRAGON_STEPS)
# The dragon's steps by side: 0 N, 1 E, 2 S, 3 W.
SIDE_LETTERS = "NESW"
SIDE_STEPS = ((0, -1), (1, 0), (0, 1), (-1, 0))
PHYSICAL_TO_CANONICAL_TYPE = list(getattr(_carcassonne_cpp, "PHYSICAL_TO_CANONICAL_TYPE", []))
# Meeple positions: 0..3 the feature on that side, 4 the monastery, then farmers:
# MEEPLE_POS_FIELD + half-edge (0..7, clockwise from north-west) and the inner field.
MEEPLE_POS_SKIP = int(_carcassonne_cpp.MEEPLE_POS_SKIP)
MEEPLE_POS_MONASTERY = int(_carcassonne_cpp.MEEPLE_POS_MONASTERY)
MEEPLE_POS_FIELD = int(_carcassonne_cpp.MEEPLE_POS_FIELD)
MEEPLE_POS_INNER_FIELD = int(_carcassonne_cpp.MEEPLE_POS_INNER_FIELD)
# The big meeple, the builder and the pig take the same spots, offset by these (game.hpp).
MEEPLE_POS_BIG = int(_carcassonne_cpp.MEEPLE_POS_BIG)
MEEPLE_POS_BUILDER = int(_carcassonne_cpp.MEEPLE_POS_BUILDER)
MEEPLE_POS_PIG = int(_carcassonne_cpp.MEEPLE_POS_PIG)
HALF_EDGE_COUNT = int(_carcassonne_cpp.HALF_EDGE_COUNT)
# Traders & Builders goods, in the engine's order.
GOODS_NAMES = ("wine", "wheat", "cloth")
assert len(GOODS_NAMES) == int(_carcassonne_cpp.GOODS_KINDS)
# Expansions, named as the game parameters name them; EXPANSION_NAMES[0] is the base.
EXPANSION_NAMES = list(_carcassonne_cpp.EXPANSION_NAMES)
BASE_ONLY = int(_carcassonne_cpp.BASE_ONLY)
RULED_EXPANSIONS = int(_carcassonne_cpp.RULED_EXPANSIONS)
RULES_REQUIRED_EXPANSIONS = int(_carcassonne_cpp.RULES_REQUIRED_EXPANSIONS)
EXPANSION_LABELS = {
    "inns_cathedrals": "Inns & Cathedrals",
    "traders_builders": "Traders & Builders",
    "river": "River",
    "princess_dragon": "Princess & Dragon",
}
OPPONENT_MODES = {"player", "random", "mcts", "alphazero", "az"}
PLAYER_TYPES = {"human", "random", "mcts", "alphazero", "az"}
BOT_TYPES = {"random", "mcts", "alphazero"}
DEFAULT_MAX_SIMULATIONS = 200
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BOT_CLI = REPO_ROOT / "play" / "bin" / "carcassonne_bot_cli"
REBUILD_HINT = "Rebuild it with `python play/setup.py build_ext --inplace` after the OpenSpiel CMake build."
# Finished games go here (CARCASSONNE_GAME_LOG_DIR overrides it): next to the
# training run directories, outside the repo, so switching branches leaves them alone.
DEFAULT_GAME_LOG_DIR = REPO_ROOT.parent / "games"
# One line per game in the actor log format, so tools/actor_log.hpp replays them.
GAME_LOG_FILE = "log-actor-gui.txt"


def game_log_file(expansions: Dict[str, str]) -> str:
    """The actor log for games with these expansions. tools/actor_log.hpp replays base
    games only, so the others go to their own file: log-actor-gui-river.txt."""
    if not expansions:
        return GAME_LOG_FILE
    parts = [name if mode == "on" else f"{name}-{mode}" for name, mode in sorted(expansions.items())]
    return f"log-actor-gui-{'-'.join(parts)}.txt"


def expansion_modes(name: str) -> Tuple[str, ...]:
    """The modes an expansion takes, as CarcassonneGame accepts them: "tiles" deals its
    tiles alone (not for one whose tiles need its rules), "on" adds its rules."""
    bit = int(_carcassonne_cpp.expansion_bit(EXPANSION_NAMES.index(name)))
    modes = ["off"]
    if not RULES_REQUIRED_EXPANSIONS & bit:
        modes.append("tiles")
    if RULED_EXPANSIONS & bit:
        modes.append("on")
    return tuple(modes)


def expansion_masks(modes: Dict[str, str]) -> Tuple[int, int]:
    """The engine's (expansions, rules) bit masks for {expansion name: mode}."""
    expansions, rules = BASE_ONLY, 0
    for name, mode in modes.items():
        if name not in EXPANSION_NAMES[1:]:
            raise ValueError(f"Unknown expansion: {name}")
        if mode not in expansion_modes(name):
            raise ValueError(f"{name}={mode}; expected one of {', '.join(expansion_modes(name))}")
        bit = int(_carcassonne_cpp.expansion_bit(EXPANSION_NAMES.index(name)))
        if mode != "off":
            expansions |= bit
        if mode == "on":
            rules |= bit
    return expansions, rules


def expansion_parameters(modes: Dict[str, str]) -> str:
    """The expansions as the bot CLI reads them from CARCASSONNE_EXPANSIONS: "river=on"."""
    return ",".join(f"{name}={mode}" for name, mode in modes.items())


def _clamp(value: int, lower: int, upper: int) -> int:
    return max(lower, min(upper, value))


def _physical_to_art_id(physical_id: int) -> int:
    if physical_id <= 0:
        return 0
    return PHYSICAL_TO_CANONICAL_TYPE[physical_id]


def meeple_spot(meeple_pos: int) -> int:
    """Where a meeple move puts its piece, whichever piece: -1 .. MEEPLE_POS_INNER_FIELD
    (a pig's is the farmer spot of its field)."""
    return int(_carcassonne_cpp.meeple_spot(meeple_pos))


def meeple_piece(meeple_pos: int) -> str:
    """The piece a meeple move places: "meeple", or an expansion's "big", "builder" or "pig"."""
    if meeple_pos >= MEEPLE_POS_PIG:
        return "pig"
    if meeple_pos >= MEEPLE_POS_BUILDER:
        return "builder"
    if meeple_pos >= MEEPLE_POS_BIG:
        return "big"
    return "meeple"


def piece_meeple_pos(kind: int, spot: int) -> int:
    """The meeple move that puts a piece of this kind (PieceKind) on this spot, the
    inverse of meeple_piece() and meeple_spot()."""
    if kind == _carcassonne_cpp.PIECE_BIG_MEEPLE:
        return MEEPLE_POS_BIG + spot
    if kind == _carcassonne_cpp.PIECE_BUILDER:
        return MEEPLE_POS_BUILDER + spot
    if kind == _carcassonne_cpp.PIECE_PIG:
        return MEEPLE_POS_PIG + spot - MEEPLE_POS_FIELD
    return spot


def _spot_argument(spot: int) -> str:
    if spot == MEEPLE_POS_MONASTERY:
        return "monastery"
    if spot == MEEPLE_POS_INNER_FIELD:
        return "inner_field"
    if spot >= MEEPLE_POS_FIELD:
        return f"field={spot - MEEPLE_POS_FIELD}"
    return f"edge={spot}"


# The moves below are written as CarcassonneState::ActionToString writes them in the actor logs.


def _meeple_action_string(meeple_pos: int) -> str:
    if meeple_pos == MEEPLE_POS_SKIP:
        return "place_meeple(skip)"
    verb = {"pig": "place_pig", "builder": "place_builder", "big": "place_big_meeple"}.get(
        meeple_piece(meeple_pos), "place_meeple"
    )
    return f"{verb}({_spot_argument(meeple_spot(meeple_pos))})"


def _cell_action_string(choice: str, x: int, y: int) -> str:
    verb = {"portal": "portal", "princess": "princess", "fairy": "move_fairy"}[choice]
    return f"{verb}(x={x}, y={y})"


def _spot_action_string(choice: str, pos: int) -> str:
    """The second step of a choice by cell: a portal's meeple move, or the princess's
    knight or the fairy's meeple by its spot."""
    if choice == "princess":
        return f"remove_knight({_spot_argument(pos)})"
    if choice == "fairy":
        return f"fairy_meeple({_spot_argument(pos)})"
    return _meeple_action_string(pos)


def _dragon_action_string(side: int) -> str:
    return f"move_dragon(dir={SIDE_LETTERS[side]})"


def _git_revision() -> Optional[str]:
    # Not --dirty: checking the working tree takes seconds on /mnt/c, and this runs as the game ends.
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None


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
    # None lets the bot CLI read it from <az_path>/config.json.
    value_is_current_player: Optional[bool] = None

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
        if self.value_is_current_player is not None:
            env["CARCASSONNE_AZ_VALUE_IS_CURRENT_PLAYER"] = "true" if self.value_is_current_player else "false"
        return env


class BotCliClient:
    def __init__(
        self,
        path: Optional[str] = None,
        env: Optional[Dict[str, str]] = None,
        expansion_masks: Optional[Tuple[int, int]] = None,
    ):
        cli_path = Path(path or os.getenv("CARCASSONNE_BOT_CLI", str(DEFAULT_BOT_CLI)))
        if not cli_path.exists():
            raise RuntimeError(
                f"Carcassonne bot CLI not found at {cli_path}. {REBUILD_HINT}"
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
        info = self.request({"cmd": "info"})
        self._check_board_size(info)
        if expansion_masks is not None:
            self._check_expansions(info, expansion_masks)

    def _check_board_size(self, info: dict) -> None:
        shape = info.get("observation_shape", [])
        if len(shape) != 3 or shape[1] != BOARD_SIZE:
            self.close()
            raise RuntimeError(
                f"The bot CLI was built for a different board (observation shape {shape}, "
                f"expected {BOARD_SIZE}x{BOARD_SIZE}). {REBUILD_HINT}"
            )

    def _check_expansions(self, info: dict, expected: Tuple[int, int]) -> None:
        # A bot CLI built before it read CARCASSONNE_EXPANSIONS reports no masks and
        # would choose its moves for a base game.
        reported = (info.get("expansions"), info.get("rules"))
        if reported == (None, None) and expected == (BASE_ONLY, 0):
            return
        if reported != expected:
            self.close()
            raise RuntimeError(
                f"The bot CLI did not pick up the expansions (it reports {reported}, "
                f"expected {expected}). {REBUILD_HINT}"
            )

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
        auto_run_bots: bool = True,
        expansions: Optional[Dict[str, str]] = None,
    ):
        # {expansion name: mode} for the expansions in play; "off" ones are dropped.
        self.expansions = {name: mode for name, mode in (expansions or {}).items() if mode != "off"}
        self.expansion_masks = expansion_masks(self.expansions)
        if seed is None:
            seed = secrets.randbits(32)
        self.seed = seed
        self._rng = random.Random(seed)
        self.player_specs = self._resolve_player_specs(opponent_mode, player_specs)
        self.opponent_mode = "player" if self.player_specs[1].is_human else self.player_specs[1].type
        self.ai_status = ""
        self.auto_run_bots = auto_run_bots
        if self.expansions:
            expansion_bits, rule_bits = self.expansion_masks
            self._engine = _carcassonne_cpp.Carcassonne(expansions=expansion_bits, rules=rule_bits)
        else:
            self._engine = _carcassonne_cpp.Carcassonne()
        self._bot_clis: Dict[int, BotCliClient] = {}
        self._latest_tile_marker: Optional[Tuple[Tuple[int, int], int]] = None
        self.move_records: List[MoveRecord] = []
        self._pending_tile_move: Optional[Tuple[int, int, int, int, int]] = None
        # The value an AlphaZero player reported for its last tile placement.
        self._pending_bot_value: Optional[BotValue] = None
        self.last_bot_value: Optional[BotValue] = None
        self._turn = 1
        # The turn under way: the moves after its tile, the meeple move among them
        # (a portal's too) and the scores before its tile.
        self._turn_actions: List[str] = []
        self._turn_meeple_pos = MEEPLE_POS_SKIP
        self._scores_before_turn: Dict[int, int] = {}
        # The game as it is saved when it ends: every action in the actor log
        # format, and one entry per completed turn.
        self._started_at = datetime.now()
        self._actions: List[str] = []
        self._turns: List[dict] = []
        self.saved_game_path: Optional[Path] = None
        self.save_error = ""
        self._viewport_origin = self._engine_view_origin()
        self._start_bot_clis()
        self._resolve_chance_phase()
        self.state = self._build_state()
        if auto_run_bots:
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
        env = spec.bot_env()
        if self.expansions:
            env["CARCASSONNE_EXPANSIONS"] = expansion_parameters(self.expansions)
        try:
            self._bot_clis[player] = BotCliClient(env=env, expansion_masks=self.expansion_masks)
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
        label = f"P1 {self.controller_label(1)} vs P2 {self.controller_label(2)}"
        for name, mode in self.expansions.items():
            label += f" · {EXPANSION_LABELS.get(name, name)}: {mode}"
        return label

    def _current_player_index(self) -> int:
        return int(self._engine.current_player)

    def _current_player_spec(self) -> PlayerSpec:
        return self.player_specs[self._current_player_index()]

    def is_ai_turn(self) -> bool:
        """Whether a bot is to decide. In the dragon's move the players take turns, so
        that can change in the middle of a turn."""
        if self._engine.is_game_over:
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
        # Only AlphaZero reports a value. Its meeple search often has a single legal
        # move and stops after one simulation, so the tile search's value is the one kept.
        self._pending_bot_value = None
        if "value" in response:
            self._pending_bot_value = BotValue(
                player=player + 1,
                value=float(response["value"]),
                raw_value=float(response["raw_value"]),
                simulations=int(response["simulations"]),
            )
            self.last_bot_value = self._pending_bot_value
        return int(response["x"]), int(response["y"]), int(response["rot"])

    def _is_legal_bot_move(self, response: dict) -> bool:
        """Whether a bot's move after its tile is legal here. The bot CLI decides on its
        own copy of the game, so a move that is not means the two have drifted apart."""
        engine, kind = self._engine, response["kind"]
        if kind == "meeple":
            return int(response["pos"]) in engine.get_legal_meeple_moves()
        if kind == "spot":
            return int(response["pos"]) in engine.get_legal_spot_moves()
        if kind == "dragon":
            return int(response["side"]) in engine.get_legal_dragon_moves()
        cells = {
            "portal": engine.get_legal_portal_cells,
            "princess": engine.get_legal_princess_cells,
            "fairy": engine.get_legal_fairy_cells,
        }.get(str(response["choice"]))
        return cells is not None and (int(response["x"]), int(response["y"])) in cells()

    def _choose_bot_piece_move(self, player: int) -> dict:
        """The bot's move after its tile: a meeple move or a choice by cell in the meeple
        phase, a spot in PHASE_SPOT, a dragon step in PHASE_DRAGON."""
        spec = self.player_specs[player]
        if not spec.is_bot:
            raise ValueError(f"P{player + 1} is not a bot.")
        response = self._bot_request(player, self._choose_payload(spec))
        expected = {PHASE_MEEPLE: ("meeple", "cell"), PHASE_SPOT: ("spot",), PHASE_DRAGON: ("dragon",)}
        if response.get("kind") not in expected[self._engine.current_phase]:
            raise RuntimeError(f"Unexpected {response.get('kind')} move from the bot CLI.")
        return response

    def _score_snapshot(self) -> Dict[int, int]:
        scores = self._engine.player_scores
        return {1: int(scores[0]), 2: int(scores[1])}

    def _score_deltas(self, before: Dict[int, int]) -> Dict[int, int]:
        after = self._score_snapshot()
        return {player: after[player] - before.get(player, 0) for player in (1, 2)}

    def _current_tile_art_id(self) -> int:
        return _physical_to_art_id(self._engine.current_tile_in_hand)

    def _remember_tile_move(self, player: int, tile_id: int, x: int, y: int, rotation: int) -> None:
        self._pending_tile_move = (player, tile_id, x, y, rotation)

    def _record_completed_turn(self, score_deltas: Dict[int, int]) -> None:
        if self._pending_tile_move is None:
            return
        player, tile_id, x, y, rotation = self._pending_tile_move
        meeple_pos, actions = self._turn_meeple_pos, tuple(self._turn_actions)
        bot_value, self._pending_bot_value = self._pending_bot_value, None
        if bot_value is not None and bot_value.player != player:
            bot_value = None
        self.move_records.insert(
            0,
            MoveRecord(
                player=player,
                tile_id=tile_id,
                x=x,
                y=y,
                rotation=rotation,
                meeple_pos=meeple_pos,
                score_deltas=score_deltas,
                value=bot_value.value if bot_value else None,
                raw_value=bot_value.raw_value if bot_value else None,
                actions=actions,
            ),
        )
        # Scores and meeples after the turn; the last turn's include the end-game scoring.
        self._turns.append(
            {
                "turn": self._turn,
                "player": player,
                "tile_id": tile_id,
                "x": x,
                "y": y,
                "rotation": rotation,
                "meeple_pos": meeple_pos,
                "actions": list(actions),
                "score_deltas": [score_deltas[1], score_deltas[2]],
                "scores": [int(score) for score in self._engine.player_scores],
                "meeples": [int(count) for count in self._engine.holding_meeples],
                "big_meeples": [int(count) for count in self._engine.holding_big_meeples],
                "builders": [int(count) for count in self._engine.holding_builders],
                "pigs": [int(count) for count in self._engine.holding_pigs],
                "goods": [[int(count) for count in tokens] for tokens in self._engine.goods_tokens],
                "value": bot_value.value if bot_value else None,
                "raw_value": bot_value.raw_value if bot_value else None,
            }
        )
        self._pending_tile_move = None

    def _save_game(self) -> None:
        """Writes the finished game under CARCASSONNE_GAME_LOG_DIR: a JSON record of it, and a
        line in log-actor-gui.txt that tools/actor_log.hpp replays like a self-play game."""
        if self.saved_game_path is not None:
            return
        try:
            directory = Path(os.getenv("CARCASSONNE_GAME_LOG_DIR") or DEFAULT_GAME_LOG_DIR)
            directory.mkdir(parents=True, exist_ok=True)
            finished_at = datetime.now()
            scores = [int(score) for score in self._engine.player_scores]
            returns = [(scores[p] > scores[1 - p]) - (scores[p] < scores[1 - p]) for p in range(2)]

            log_file = game_log_file(self.expansions)
            log_path = directory / log_file
            game_number = 1
            if log_path.exists():
                with log_path.open(encoding="utf-8") as log:
                    game_number += sum(1 for line in log if line.strip())

            labels = "_vs_".join(spec.label for spec in self.player_specs)
            path = directory / f"{finished_at:%Y%m%d-%H%M%S}_{labels}.json"
            suffix = 2
            while path.exists():
                path = directory / f"{finished_at:%Y%m%d-%H%M%S}_{labels}_{suffix}.json"
                suffix += 1
            record = {
                "started_at": self._started_at.isoformat(timespec="seconds"),
                "finished_at": finished_at.isoformat(timespec="seconds"),
                "seed": self.seed,
                "players": [asdict(spec) for spec in self.player_specs],
                "bot_cli": os.getenv("CARCASSONNE_BOT_CLI", str(DEFAULT_BOT_CLI)),
                "git": _git_revision(),
                "expansions": self.expansions,
                "log_file": log_file,
                "log_game": game_number,
                "scores": scores,
                "returns": returns,
                "turns": self._turns,
                "actions": self._actions,
            }
            path.write_text(json.dumps(record, indent=1) + "\n", encoding="utf-8")

            stamp = f"{finished_at:%Y-%m-%d %H:%M:%S}.{finished_at.microsecond // 1000:03d}"
            with log_path.open("a", encoding="utf-8") as log:
                log.write(
                    f"[{stamp}] Game {game_number}: Returns: {returns[0]} {returns[1]}; "
                    f"Actions: {' '.join(self._actions)}\n"
                )
            self.saved_game_path = path
        except Exception as exc:
            self.save_error = f"Could not save the game: {exc}"

    @property
    def view_origin(self) -> Tuple[int, int]:
        return self._viewport_origin

    def to_engine_coords(self, x: int, y: int) -> Tuple[int, int]:
        origin_x, origin_y = self._viewport_origin
        return origin_x + x, origin_y + y

    def _engine_view_origin(self) -> Tuple[int, int]:
        """The origin of the engine's view, which moves as tiles are placed."""
        origin_x, origin_y = self._engine.view_origin
        return int(origin_x), int(origin_y)

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
            self._actions.append(f"draw_type({draw_type})")
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

    def decision(self) -> Optional[str]:
        """What a human player to move decides: "tile", "piece" (the meeple phase:
        a meeple move or a choice by cell), "spot" or "dragon"; None while a bot is
        to move and once the game is over."""
        if self._engine.is_game_over or self._current_player_spec().is_bot:
            return None
        phase = self._engine.current_phase
        return {PHASE_TILE: "tile", PHASE_MEEPLE: "piece", PHASE_SPOT: "spot", PHASE_DRAGON: "dragon"}.get(phase)

    def meeple_options(self) -> List[int]:
        """The meeple moves of a human's meeple phase, skip included."""
        if self.decision() != "piece":
            return []
        return [int(pos) for pos in self._engine.get_legal_meeple_moves()]

    def cell_options(self) -> Dict[str, List[Tuple[int, int]]]:
        """A human's choices by cell in the meeple phase, {choice: UI cells}, for the
        choices that have one."""
        if self.decision() != "piece":
            return {}
        found = {
            "portal": self._engine.get_legal_portal_cells(),
            "princess": self._engine.get_legal_princess_cells(),
            "fairy": self._engine.get_legal_fairy_cells(),
        }
        options: Dict[str, List[Tuple[int, int]]] = {}
        for choice, cells in found.items():
            ui_cells = [cell for cell in (self._to_ui_coords(x, y) for x, y in cells) if cell is not None]
            if ui_cells:
                options[choice] = ui_cells
        return options

    def spot_options(self) -> Tuple[str, List[int]]:
        """In PHASE_SPOT, the choice by cell under way and its moves on the cell."""
        if self.decision() != "spot":
            return "", []
        return SPOT_CHOICES[self._engine.spot_choice], [int(pos) for pos in self._engine.get_legal_spot_moves()]

    def spot_cell(self) -> Optional[Tuple[int, int]]:
        """In PHASE_SPOT, the UI cell the choice by cell took."""
        if self.decision() != "spot":
            return None
        return self._to_ui_coords(*self._engine.spot_cell)

    def dragon_options(self) -> List[Tuple[int, Tuple[int, int]]]:
        """In a human's dragon step, (side, UI cell) for each cell the dragon can enter."""
        if self.decision() != "dragon":
            return []
        dragon_x, dragon_y = self._engine.dragon_cell
        options = []
        for side in self._engine.get_legal_dragon_moves():
            step_x, step_y = SIDE_STEPS[side]
            cell = self._to_ui_coords(dragon_x + step_x, dragon_y + step_y)
            if cell is not None:
                options.append((int(side), cell))
        return options

    def get_valid_moves(self) -> List[Move]:
        if self.decision() != "tile":
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

        self._place_tile(engine_x, engine_y, move.rotation)
        self.state = self._build_state()
        return self.meeple_options()

    def apply_meeple(self, meeple_pos: int) -> None:
        if meeple_pos not in self.meeple_options():
            raise ValueError(f"Invalid meeple position: {meeple_pos}")
        self._place_meeple(meeple_pos)
        self._after_human_move()

    def apply_cell(self, choice: str, x: int, y: int) -> None:
        """A choice by cell for the meeple phase, on UI cell (x, y)."""
        if (x, y) not in self.cell_options().get(choice, []):
            raise ValueError(f"Invalid {choice} cell: ({x}, {y})")
        self._choose_cell(choice, *self.to_engine_coords(x, y))
        self._after_human_move()

    def apply_spot(self, pos: int) -> None:
        if pos not in self.spot_options()[1]:
            raise ValueError(f"Invalid spot: {pos}")
        self._choose_spot(pos)
        self._after_human_move()

    def apply_dragon(self, side: int) -> None:
        if side not in [option for option, _ in self.dragon_options()]:
            raise ValueError(f"The dragon cannot go {SIDE_LETTERS[side]}.")
        self._move_dragon(side)
        self._after_human_move()

    def _after_human_move(self) -> None:
        if self.auto_run_bots:
            self.run_ai_turns()
        self.state = self._build_state()

    def _place_tile(self, x: int, y: int, rotation: int) -> None:
        """Places the tile in hand for the player to move and passes it on to the bots."""
        player = self._engine.current_player + 1
        tile_id = self._current_tile_art_id()
        self._scores_before_turn = self._score_snapshot()
        self._turn_actions, self._turn_meeple_pos = [], MEEPLE_POS_SKIP
        self._engine.place_tile(x, y, rotation)
        # Follow the view, out of which no tile can go.
        self._viewport_origin = self._engine_view_origin()
        self._actions.append(f"place_tile(x={x}, y={y}, rot={rotation})")
        self._latest_tile_marker = ((x, y), player)
        self._remember_tile_move(player, tile_id, x, y, rotation)
        self._sync_bots({"cmd": "apply_tile", "x": x, "y": y, "rot": rotation})

    # The moves after the tile, for whoever makes them. Each passes the move on to the
    # bots and ends the turn if it was the last one.

    def _place_meeple(self, meeple_pos: int) -> None:
        self._engine.place_meeple(meeple_pos)
        self._turn_meeple_pos = meeple_pos
        self._after_piece_move(_meeple_action_string(meeple_pos), {"cmd": "apply_meeple", "pos": meeple_pos})

    def _choose_cell(self, choice: str, x: int, y: int) -> None:
        self._engine.choose_cell(SPOT_CHOICES.index(choice), x, y)
        self._after_piece_move(_cell_action_string(choice, x, y), {"cmd": "apply_cell", "choice": choice, "x": x, "y": y})

    def _choose_spot(self, pos: int) -> None:
        choice = SPOT_CHOICES[self._engine.spot_choice]
        self._engine.choose_spot(pos)
        if choice == "portal":
            self._turn_meeple_pos = pos
        self._after_piece_move(_spot_action_string(choice, pos), {"cmd": "apply_spot", "pos": pos})

    def _move_dragon(self, side: int) -> None:
        self._engine.move_dragon(side)
        self._after_piece_move(_dragon_action_string(side), {"cmd": "apply_dragon", "side": side})

    def _after_piece_move(self, action: str, bot_payload: dict) -> None:
        """Logs the move and passes it on; once the turn is over, records it, draws the
        next tile, and saves the game if it is over."""
        self._actions.append(action)
        self._turn_actions.append(action)
        self._sync_bots(bot_payload)
        if self._engine.current_phase in PIECE_PHASES:
            return
        self._record_completed_turn(self._score_deltas(self._scores_before_turn))
        self._turn += 1
        self._resolve_chance_phase()
        if self._engine.is_game_over:
            self._save_game()

    def run_ai_turns(self, max_turns: Optional[int] = None) -> int:
        self.ai_status = ""
        if not self.has_bot_players():
            return 0

        ai_turns = 0
        moves = 0
        last_label = ""
        try:
            # Until a human is to decide: the next turn, or a step of the dragon's move.
            while not self._engine.is_game_over and self._current_player_spec().is_bot:
                player = self._current_player_index()
                spec = self.player_specs[player]
                last_label = f"P{player + 1} {spec.label}"
                phase = self._engine.current_phase
                if phase == PHASE_CHANCE:
                    self._resolve_chance_phase()
                    continue
                if phase == PHASE_TILE:
                    x, y, rotation = self._choose_bot_tile_move(player)
                    if (x, y, rotation) not in set(self._engine.get_legal_tile_moves()):
                        raise RuntimeError(f"{last_label} chose an illegal tile placement: the bot CLI is out of sync.")
                    self._place_tile(x, y, rotation)
                    moves += 1
                    continue
                if phase not in PIECE_PHASES:
                    break
                turn = self._turn
                response = self._choose_bot_piece_move(player)
                if not self._is_legal_bot_move(response):
                    raise RuntimeError(f"{last_label} chose an illegal move {response}: the bot CLI is out of sync.")
                kind = response["kind"]
                if kind == "meeple":
                    self._place_meeple(int(response["pos"]))
                elif kind == "cell":
                    self._choose_cell(str(response["choice"]), int(response["x"]), int(response["y"]))
                elif kind == "spot":
                    self._choose_spot(int(response["pos"]))
                else:
                    self._move_dragon(int(response["side"]))
                moves += 1
                if self._turn != turn:
                    ai_turns += 1
                    if max_turns is not None and ai_turns >= max_turns:
                        break
        except Exception as exc:
            self.ai_status = str(exc)
            self.state = self._build_state()
            return ai_turns

        if ai_turns:
            self.ai_status = f"{last_label} played {ai_turns} bot turn(s)."
        elif moves:
            self.ai_status = f"{last_label} moved the dragon."
        self.state = self._build_state()
        return ai_turns

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
        # Every piece on the board, where it stands (Carcassonne::pieces): pieces leave it
        # when their road, city or monastery is scored, when the dragon eats them and when
        # the princess sends a knight home, and a portal's meeple goes on an earlier tile.
        for x, y, owner, kind, spot in self._engine.get_pieces():
            tile = board.get((x, y))
            if tile is None:
                raise RuntimeError(f"A piece at ({x}, {y}) has no tile under it")
            tile.meeple_markers.append((owner + 1, piece_meeple_pos(kind, spot)))
        for tile in board.values():
            # In a fixed order: the engine reorders its pieces as they leave.
            tile.meeple_markers.sort()
            if tile.meeple_markers:
                tile.meeple_owner, tile.meeple_pos = tile.meeple_markers[-1]
        return board

    def _build_state(self) -> GameState:
        game_over = bool(self._engine.is_game_over)
        holding_tile_id = None
        if self._engine.current_phase == PHASE_TILE:
            holding_tile_id = _physical_to_art_id(self._engine.current_tile_in_hand)

        scores = self._engine.player_scores
        meeples = self._engine.holding_meeples
        current_player_ui = self._engine.current_player + 1
        holdings = {
            "big": (self._engine.big_meeple_rules, self._engine.holding_big_meeples),
            "builder": (self._engine.builder_rules, self._engine.holding_builders),
            "pig": (self._engine.pig_rules, self._engine.holding_pigs),
        }
        piece_kinds = tuple(kind for kind, (rules, _) in holdings.items() if rules)
        goods = {}
        if self._engine.goods_rules:
            goods = {player + 1: tuple(int(n) for n in tokens) for player, tokens in enumerate(self._engine.goods_tokens)}
        engine = self._engine
        choices = (("portal", engine.portal_rules), ("princess", engine.princess_rules), ("fairy", engine.fairy_rules))
        dragon_x, dragon_y = engine.dragon_cell
        fairy_x, fairy_y = engine.fairy_cell
        in_dragon_move = engine.current_phase == PHASE_DRAGON

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
            piece_kinds=piece_kinds,
            pieces_remaining={
                player: {kind: int(holdings[kind][1][player - 1]) for kind in piece_kinds} for player in (1, 2)
            },
            goods=goods,
            builder_extra_tile=bool(self._engine.builder_second_tile) and not game_over,
            phase=int(engine.current_phase),
            choice_kinds=tuple(name for name, rules in choices if rules),
            dragon=(int(dragon_x), int(dragon_y)) if dragon_x >= 0 else None,
            dragon_visited=[(int(x), int(y)) for x, y in engine.dragon_visited] if in_dragon_move else [],
            dragon_steps=int(engine.dragon_steps) if in_dragon_move else 0,
            fairy=(int(fairy_x), int(fairy_y), int(engine.fairy_spot)) if fairy_x >= 0 else None,
            fairy_owner=int(engine.fairy_owner) + 1 if engine.fairy_owner >= 0 else None,
        )
