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


# Expansions the engine plays "on" but the UI cannot yet: The Princess & the Dragon's
# dragon moves in a phase of its own, which the UI and the bot CLI do not drive.
UI_UNPLAYED_RULES = {"princess_dragon"}


def expansion_modes(name: str) -> Tuple[str, ...]:
    """The modes an expansion takes in the UI, as CarcassonneGame accepts them: "tiles"
    deals its tiles alone (not for one whose tiles need its rules), "on" adds its rules."""
    bit = int(_carcassonne_cpp.expansion_bit(EXPANSION_NAMES.index(name)))
    modes = ["off"]
    if not RULES_REQUIRED_EXPANSIONS & bit:
        modes.append("tiles")
    if RULED_EXPANSIONS & bit and name not in UI_UNPLAYED_RULES:
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


def _meeple_action_string(meeple_pos: int) -> str:
    """The meeple move as CarcassonneState::ActionToString writes it in the actor logs."""
    if meeple_pos == MEEPLE_POS_SKIP:
        return "place_meeple(skip)"
    if meeple_pos >= MEEPLE_POS_PIG:
        verb, spot = "place_pig", MEEPLE_POS_FIELD + meeple_pos - MEEPLE_POS_PIG
    elif meeple_pos >= MEEPLE_POS_BUILDER:
        verb, spot = "place_builder", meeple_pos - MEEPLE_POS_BUILDER
    elif meeple_pos >= MEEPLE_POS_BIG:
        verb, spot = "place_big_meeple", meeple_pos - MEEPLE_POS_BIG
    else:
        verb, spot = "place_meeple", meeple_pos
    if spot == MEEPLE_POS_MONASTERY:
        arg = "monastery"
    elif spot == MEEPLE_POS_INNER_FIELD:
        arg = "inner_field"
    elif spot >= MEEPLE_POS_FIELD:
        arg = f"field={spot - MEEPLE_POS_FIELD}"
    else:
        arg = f"edge={spot}"
    return f"{verb}({arg})"


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
        self._pending_meeple_options: List[int] = []
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

    def _choose_bot_meeple_move(self, player: int) -> int:
        spec = self.player_specs[player]
        if not spec.is_bot:
            raise ValueError(f"P{player + 1} is not a bot.")
        response = self._bot_request(player, self._choose_payload(spec))
        if response.get("kind") != "meeple":
            raise RuntimeError(f"Expected meeple action from bot CLI, got {response.get('kind')}.")
        return int(response["pos"])

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

    def _record_completed_turn(self, meeple_pos: int, score_deltas: Dict[int, int]) -> None:
        if self._pending_tile_move is None:
            return
        player, tile_id, x, y, rotation = self._pending_tile_move
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

        self._place_tile(engine_x, engine_y, move.rotation)
        self._pending_meeple_options = list(self._engine.get_legal_meeple_moves())
        self.state = self._build_state()
        return list(self._pending_meeple_options)

    def apply_meeple(self, meeple_pos: int) -> None:
        if meeple_pos not in self._pending_meeple_options:
            raise ValueError(f"Invalid meeple position: {meeple_pos}")

        self._pending_meeple_options = []
        self._place_meeple(meeple_pos)
        if self.auto_run_bots:
            self.run_ai_turns()
        self.state = self._build_state()

    def _place_tile(self, x: int, y: int, rotation: int) -> None:
        """Places the tile in hand for the player to move and passes it on to the bots."""
        player = self._engine.current_player + 1
        tile_id = self._current_tile_art_id()
        self._engine.place_tile(x, y, rotation)
        # Follow the view, out of which no tile can go.
        self._viewport_origin = self._engine_view_origin()
        self._actions.append(f"place_tile(x={x}, y={y}, rot={rotation})")
        self._latest_tile_marker = ((x, y), player)
        self._remember_tile_move(player, tile_id, x, y, rotation)
        self._sync_bots({"cmd": "apply_tile", "x": x, "y": y, "rot": rotation})

    def _place_meeple(self, meeple_pos: int) -> None:
        """Ends the turn with the meeple move, draws the next tile, and saves the game if it is over."""
        score_before = self._score_snapshot()
        self._engine.place_meeple(meeple_pos)
        self._actions.append(_meeple_action_string(meeple_pos))
        self._sync_bots({"cmd": "apply_meeple", "pos": meeple_pos})
        self._record_completed_turn(meeple_pos, self._score_deltas(score_before))
        self._turn += 1
        self._resolve_chance_phase()
        if self._engine.is_game_over:
            self._save_game()

    def run_ai_turns(self, max_turns: Optional[int] = None) -> int:
        self.ai_status = ""
        if not self.has_bot_players():
            return 0

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
                    self._place_tile(x, y, rotation)
                    continue
                if self._engine.current_phase == PHASE_MEEPLE:
                    self._place_meeple(self._choose_bot_meeple_move(player))
                    ai_turns += 1
                    if max_turns is not None and ai_turns >= max_turns:
                        break
                    continue
                break
        except Exception as exc:
            self.ai_status = str(exc)
            self.state = self._build_state()
            return ai_turns

        if ai_turns:
            self.ai_status = f"{last_label} played {ai_turns} bot turn(s)."
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
        # The engine only knows meeple counts per feature, so get_meeple_tokens() marks every
        # edge of a claimed road/city. Draw each piece where it was actually placed (from
        # move_records) and use the tokens only to tell whether it is still on the board:
        # completing a feature or monastery clears its token and returns the meeple.
        # - A big meeple counts in the same tokens as a meeple.
        # - A builder goes on a road or city that holds one of its owner's meeples, which
        #   stays there till the feature is completed and both go home: it has a token as
        #   long as the builder is there.
        # - Farmers and pigs have no token and are never returned, so they always stay.
        live_tokens = set(self._engine.get_meeple_tokens())
        for record in self.move_records:
            if record.meeple_pos == MEEPLE_POS_SKIP:
                continue
            spot = meeple_spot(record.meeple_pos)
            if spot < MEEPLE_POS_FIELD and (record.player - 1, record.x, record.y, spot) not in live_tokens:
                continue
            tile = board.get((record.x, record.y))
            if tile is None:
                raise RuntimeError(f"Meeple record {record} has no matching tile snapshot")
            tile.meeple_markers.append((record.player, record.meeple_pos))
            tile.meeple_owner, tile.meeple_pos = record.player, record.meeple_pos
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
        )
