from __future__ import annotations

import argparse
import asyncio
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import flet as ft

try:
    from domain import BotValue, Move, MoveRecord
    from engine import (
        BOARD_SIZE,
        DRAGON_STEPS,
        EXPANSION_LABELS,
        EXPANSION_NAMES,
        GOODS_NAMES,
        HALF_EDGE_COUNT,
        MEEPLE_POS_FIELD,
        MEEPLE_POS_INNER_FIELD,
        MEEPLE_POS_PIG,
        PHASE_DRAGON,
        CppCarcassonneAdapter,
        PlayerSpec,
        expansion_modes,
        meeple_piece,
        meeple_spot,
    )
except ImportError:  # pragma: no cover - package import fallback
    from ..domain import BotValue, Move, MoveRecord
    from ..engine import (
        BOARD_SIZE,
        DRAGON_STEPS,
        EXPANSION_LABELS,
        EXPANSION_NAMES,
        GOODS_NAMES,
        HALF_EDGE_COUNT,
        MEEPLE_POS_FIELD,
        MEEPLE_POS_INNER_FIELD,
        MEEPLE_POS_PIG,
        PHASE_DRAGON,
        CppCarcassonneAdapter,
        PlayerSpec,
        expansion_modes,
        meeple_piece,
        meeple_spot,
    )


IMAGE_FIT = getattr(ft, "ImageFit", ft.BoxFit)
# Flet 1.0 dropped ElevatedButton; older releases have both.
BUTTON = getattr(ft, "Button", None) or ft.ElevatedButton
ALIGN_CENTER = ft.alignment.Alignment(0, 0)

CELL_SIZE = 40
MEEPLE_SIZE = 15
BIG_MEEPLE_SIZE = 20
# The pieces a meeple move can place (adapter.meeple_piece), each with its own tab of
# buttons and its symbol in the pieces left in hand.
PIECE_KINDS = ("meeple", "big", "builder", "pig")
PIECE_TAB_LABELS = {"meeple": "Meeple", "big": "Big meeple", "builder": "Builder", "pig": "Pig"}
PIECE_GLYPHS = {"meeple": "■", "big": "◆", "builder": "▲", "pig": "●"}
# The Princess & the Dragon's choices by cell (adapter.SPOT_CHOICES), each with a tab
# after the pieces': it colours the tiles the choice can take, and a click takes one.
CHOICE_KINDS = ("portal", "princess", "fairy")
CHOICE_TAB_LABELS = {"portal": "Portal", "princess": "Princess", "fairy": "Fairy"}
CHOICE_HINTS = {
    "portal": "Click a purple tile to put a meeple on it.",
    "princess": "Click a pink tile to send a knight on it home.",
    "fairy": "Click a blue tile to move the fairy next to your meeple there.",
}
# Cell highlights: the empty cells the tile in hand can go on, the tiles a choice by
# cell can take, and the tiles the dragon can step on.
TILE_MOVE_BG, TILE_MOVE_BORDER = "#ecfdf3", "#63b36f"
HIGHLIGHT_COLORS = {"portal": "#9333ea", "princess": "#db2777", "fairy": "#0284c7", "dragon": "#ea580c"}
DRAGON_VISITED_COLOR = "#ea580c"
DRAGON_WIDTH, DRAGON_HEIGHT = 34, 17
FAIRY_SIZE = 16
MIN_BOARD_SCALE = 0.3
MAX_BOARD_SCALE = 4.0
ZOOM_STEP = 1.25
QUARTER_TURN = math.pi / 2
OPPONENT_TYPES = ("az", "mcts", "random", "human")
DEFAULT_SIMULATIONS = 800


def _padding(left: int = 0, top: int = 0, right: int = 0, bottom: int = 0) -> ft.Padding:
    # ft.padding.only() is gone in Flet 1.0; the Padding constructor works everywhere.
    return ft.Padding(left=left, top=top, right=right, bottom=bottom)


def _border(width: int, color: str) -> ft.Border:
    side = ft.BorderSide(width=width, color=color)
    return ft.Border(left=side, top=side, right=side, bottom=side)


def _margin(value: float) -> ft.Margin:
    return ft.Margin(left=value, top=value, right=value, bottom=value)


SIDE_NAMES = ("Up", "Right", "Down", "Left")
# Half-edge e = 2 * side + h runs clockwise round the tile from north-west (tile.hpp).
HALF_EDGE_NAMES = (
    "Top-left",
    "Top-right",
    "Right-top",
    "Right-bottom",
    "Bottom-right",
    "Bottom-left",
    "Left-bottom",
    "Left-top",
)
assert len(HALF_EDGE_NAMES) == HALF_EDGE_COUNT
_SIDE_DIRECTIONS = ((0, -1), (1, 0), (0, 1), (-1, 0))
_SIDE_TANGENTS = ((1, 0), (0, 1), (-1, 0), (0, -1))  # clockwise along each side


def is_farmer(meeple_pos: int) -> bool:
    """A meeple or the big meeple lying in a field (a pig there is no farmer)."""
    return meeple_piece(meeple_pos) in ("meeple", "big") and meeple_spot(meeple_pos) >= MEEPLE_POS_FIELD


def _spot_label(spot: int) -> str:
    if spot < len(SIDE_NAMES):
        return SIDE_NAMES[spot]
    if spot == MEEPLE_POS_INNER_FIELD:
        return "Farmer: Inner"
    if spot >= MEEPLE_POS_FIELD:
        return f"Farmer: {HALF_EDGE_NAMES[spot - MEEPLE_POS_FIELD]}"
    return "Center"


def meeple_button_labels() -> Dict[int, str]:
    """A label for every meeple position the engine can offer, skip aside. The tab a
    button sits in names the piece, so the label only says where it goes."""
    labels: Dict[int, str] = {}
    for pos in range(MEEPLE_POS_PIG + HALF_EDGE_COUNT):
        spot = meeple_spot(pos)
        if meeple_piece(pos) == "pig":
            labels[pos] = f"Field: {HALF_EDGE_NAMES[spot - MEEPLE_POS_FIELD]}"
        else:
            labels[pos] = _spot_label(spot)
    return labels


def meeple_marker(meeple_pos: int) -> Tuple[str, int]:
    """The image under meeples/ (less _p<owner>.png) and the size a piece is drawn at."""
    piece = meeple_piece(meeple_pos)
    if piece in ("builder", "pig"):
        return piece, MEEPLE_SIZE
    image = "farmer" if is_farmer(meeple_pos) else "standing"
    return image, BIG_MEEPLE_SIZE if piece == "big" else MEEPLE_SIZE


def pieces_in_hand(meeples: int, pieces: Dict[str, int]) -> str:
    """The pieces a player holds as symbols: a square per meeple, then the expansions' pieces."""
    special = "".join(PIECE_GLYPHS[kind] * pieces[kind] for kind in PIECE_KINDS[1:] if pieces.get(kind))
    text = PIECE_GLYPHS["meeple"] * meeples + (" " + special if meeples and special else special)
    return text or "-"


def offered_pieces(meeple_options: Sequence[int]) -> List[str]:
    """The pieces the meeple moves can place, in tab order; skip places none."""
    kinds = {meeple_piece(pos) for pos in meeple_options if pos != -1}
    return [kind for kind in PIECE_KINDS if kind in kinds]


def offered_tabs(meeple_options: Sequence[int], cell_options: Dict[str, List[Tuple[int, int]]]) -> List[str]:
    """The tabs of the meeple phase that have a move: pieces, then choices by cell."""
    return offered_pieces(meeple_options) + [choice for choice in CHOICE_KINDS if cell_options.get(choice)]


def spot_button_label(choice: str, pos: int) -> str:
    """A button for the second step of a choice by cell: a portal's meeple move, or
    the princess's knight or the fairy's meeple on the cell, by its spot."""
    if choice == "portal":
        label = meeple_button_labels()[pos]
        return f"Big: {label}" if meeple_piece(pos) == "big" else label
    return f"{'Knight' if choice == 'princess' else 'Next to'}: {_spot_label(pos)}"


def turn_extras(actions: Sequence[str]) -> str:
    """A turn's choices by cell and its dragon steps, for the record: "portal(12,9) dragon NES"."""
    parts: List[str] = []
    steps = ""
    for action in actions:
        name, _, args = action.partition("(")
        if name in ("portal", "princess", "move_fairy"):
            x, y = (int(part.split("=")[1]) for part in args.rstrip(")").split(", "))
            parts.append(f"{'fairy' if name == 'move_fairy' else name}({x},{y})")
        elif name == "move_dragon":
            steps += args[len("dir=") : -1]
    if steps:
        parts.append(f"dragon {steps}")
    return " ".join(parts)


def dragon_status(state) -> str:
    if state.dragon is None:
        return "Dragon: not in play yet (a volcano brings it)"
    if state.phase == PHASE_DRAGON:
        return f"Dragon: moving, step {state.dragon_steps + 1}/{DRAGON_STEPS}, P{state.current_player} to move"
    return "Dragon: on the board"


def fairy_status(state) -> str:
    if state.fairy is None:
        return "Fairy: in the supply"
    if state.fairy_owner is None:
        return "Fairy: on the board, next to no one"
    return f"Fairy: next to P{state.fairy_owner}'s meeple"


def fairy_alignment(spot: int) -> Tuple[float, float]:
    """Where the fairy stands in its cell: beside the spot of its meeple, or in the
    middle when it is next to no one."""
    if spot < 0:
        return 0.0, 0.0
    align_x, align_y = meeple_alignment(spot)
    return (align_x - 0.5 if align_x > 0.2 else align_x + 0.5), align_y


def format_goods(goods: Tuple[int, int, int]) -> str:
    return " · ".join(f"{name} {count}" for name, count in zip(GOODS_NAMES, goods))


def meeple_alignment(meeple_pos: int) -> Tuple[float, float]:
    """Where a meeple sits in its cell, as an Alignment (-1..1 on each axis)."""
    if 0 <= meeple_pos < 4:
        dx, dy = _SIDE_DIRECTIONS[meeple_pos]
        return 0.75 * dx, 0.75 * dy
    if MEEPLE_POS_FIELD <= meeple_pos < MEEPLE_POS_FIELD + HALF_EDGE_COUNT:
        # A farmer lies in its field, just inside that half of the side.
        side, half = divmod(meeple_pos - MEEPLE_POS_FIELD, 2)
        (dx, dy), (tx, ty) = _SIDE_DIRECTIONS[side], _SIDE_TANGENTS[side]
        along = 0.5 if half else -0.5
        return 0.7 * dx + along * tx, 0.7 * dy + along * ty
    return 0.0, 0.0  # the monastery and the inner field


def view_shift_pan(old_origin: Tuple[int, int], new_origin: Tuple[int, int]) -> Tuple[float, float]:
    """The viewer pan that keeps the tiles still on screen when the engine's view moves
    from old_origin to new_origin. A tile's grid cell moves by -delta; pan() adds d in
    content units (the screen shows s * (p + d) + t), so d = +delta whatever the zoom."""
    return (new_origin[0] - old_origin[0]) * CELL_SIZE, (new_origin[1] - old_origin[1]) * CELL_SIZE


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


@dataclass(frozen=True)
class PlayUiConfig:
    game: str = "carcassonne"
    p1_spec: PlayerSpec = field(default_factory=PlayerSpec)
    p2_spec: PlayerSpec = field(default_factory=PlayerSpec)
    seed: Optional[int] = None
    # {expansion name: mode} for the expansions in play.
    expansions: Dict[str, str] = field(default_factory=dict)

    @property
    def player_specs(self) -> Tuple[PlayerSpec, PlayerSpec]:
        return self.p1_spec, self.p2_spec


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Carcassonne Flet UI")
    player_types = ("human", "random", "mcts", "az", "alphazero")
    parser.add_argument("--game", default="carcassonne")
    parser.add_argument("--p1_type", choices=player_types, default="human")
    parser.add_argument("--p2_type", choices=player_types, default="human")
    parser.add_argument("--p1_az_path", default="")
    parser.add_argument("--p2_az_path", default="")
    parser.add_argument("--p1_az_checkpoint", type=int, default=None)
    parser.add_argument("--p2_az_checkpoint", type=int, default=None)
    parser.add_argument("--p1_az_graph_def", default="vpnet.pb")
    parser.add_argument("--p2_az_graph_def", default="vpnet.pb")
    parser.add_argument("--p1_max_simulations", type=_positive_int, default=None)
    parser.add_argument("--p2_max_simulations", type=_positive_int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    for name in EXPANSION_NAMES[1:]:
        parser.add_argument(f"--{name}", choices=expansion_modes(name), default="off")
    return parser


def parse_ui_config(argv: Optional[Sequence[str]] = None) -> PlayUiConfig:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if args.game != "carcassonne":
        parser.error("play UI currently supports --game=carcassonne only.")
    return PlayUiConfig(
        game=args.game,
        p1_spec=PlayerSpec(
            type=args.p1_type,
            az_path=args.p1_az_path,
            az_checkpoint=args.p1_az_checkpoint,
            az_graph_def=args.p1_az_graph_def,
            max_simulations=args.p1_max_simulations,
        ),
        p2_spec=PlayerSpec(
            type=args.p2_type,
            az_path=args.p2_az_path,
            az_checkpoint=args.p2_az_checkpoint,
            az_graph_def=args.p2_az_graph_def,
            max_simulations=args.p2_max_simulations,
        ),
        seed=None if args.seed == 0 else args.seed,
        expansions={name: getattr(args, name) for name in EXPANSION_NAMES[1:] if getattr(args, name) != "off"},
    )


def format_move_record(record: MoveRecord) -> str:
    text = _format_move_and_score(record)
    extras = turn_extras(record.actions)
    if extras:
        text += f" {extras}"
    if record.value is not None:
        text += f" v{record.value:+.2f}"
    return text


def _format_move_and_score(record: MoveRecord) -> str:
    base = f"P{record.player}({record.tile_id},{record.x},{record.y},{record.rotation},{record.meeple_pos})"
    nonzero_deltas = {player: delta for player, delta in sorted(record.score_deltas.items()) if delta}
    if not nonzero_deltas:
        return f"{base} +0(得分)"
    if set(nonzero_deltas) == {record.player}:
        points = nonzero_deltas[record.player]
        return f"{base} {points:+d}(得分)"

    deltas = "/".join(f"P{player}{delta:+d}" for player, delta in nonzero_deltas.items())
    return f"{base} {deltas}(得分)"


def format_bot_value(bot_value: BotValue) -> str:
    """AlphaZero's view of the game when it placed its last tile, from its own side."""
    return (
        f"AZ (P{bot_value.player}) value {bot_value.value:+.2f}"
        f" · net {bot_value.raw_value:+.2f} · {bot_value.simulations} sims"
    )


def summarize_ai_status(status: str, limit: int = 240) -> str:
    """First line of a bot message. libtorch errors carry a stack trace long enough
    to push the rest of the side panel (Record included) out of view."""
    first_line = next((line.strip() for line in status.splitlines() if line.strip()), "")
    return first_line if len(first_line) <= limit else first_line[: limit - 3] + "..."


def is_bot_vs_bot(player_specs: Tuple[PlayerSpec, PlayerSpec]) -> bool:
    return all(spec.is_bot for spec in player_specs)


def should_show_start_game(player_specs: Tuple[PlayerSpec, PlayerSpec], started: bool) -> bool:
    return is_bot_vs_bot(player_specs) and not started


def has_bot_player(player_specs: Tuple[PlayerSpec, PlayerSpec]) -> bool:
    return any(spec.is_bot for spec in player_specs)


def human_seat(player_specs: Tuple[PlayerSpec, PlayerSpec]) -> Optional[int]:
    """The 1-based seat of the only human, or None when both or neither are human."""
    humans = [index + 1 for index, spec in enumerate(player_specs) if spec.is_human]
    return humans[0] if len(humans) == 1 else None


def build_player_specs(
    human_player: int,
    opponent: str,
    az_path: str = "",
    az_checkpoint: Optional[int] = None,
    max_simulations: Optional[int] = None,
) -> Tuple[PlayerSpec, PlayerSpec]:
    """Seat a human against one bot, as chosen in the setup panel."""
    if human_player not in (1, 2):
        raise ValueError("human_player must be 1 or 2.")
    bot = PlayerSpec(
        type=opponent,
        az_path=az_path.strip() if opponent in ("az", "alphazero") else "",
        az_checkpoint=az_checkpoint if opponent in ("az", "alphazero") else None,
        max_simulations=max_simulations if opponent in ("az", "alphazero", "mcts") else None,
    )
    if bot.type == "alphazero" and not bot.az_path:
        raise ValueError("Enter the model directory (the training run's --path).")
    human = PlayerSpec("human")
    return (human, bot) if human_player == 1 else (bot, human)


class CarcassonneUI:
    def __init__(self, page: ft.Page, config: Optional[PlayUiConfig] = None):
        self.page = page
        self.config = config or PlayUiConfig()
        self.player_specs = self.config.player_specs
        self.engine: Optional[CppCarcassonneAdapter] = None
        self.state = None
        self.bot_game_started = False
        self.ai_running = False

        self.selected_move: Optional[Move] = None
        # What refresh() found the human can choose: the meeple phase's choices by cell,
        # {UI cell: side} for a dragon step, and how each cell is coloured.
        self.cell_options: Dict[str, List[Tuple[int, int]]] = {}
        self.dragon_moves: Dict[Tuple[int, int], int] = {}
        self.cell_highlights: Dict[Tuple[int, int], str] = {}
        self.board_zoom = 1.0
        self.board_viewport: Optional[Tuple[float, float]] = None
        self.center_board_pending = True
        # The engine's view the grid was last drawn from; it moves with the tiles.
        self.shown_view_origin: Optional[Tuple[int, int]] = None
        self._view_lock = asyncio.Lock()

        self._build_setup_panel()
        self._build_game_panel()

        # Rows and cells sit edge to edge (spacing 0) so neighbouring tiles touch.
        # The rows are built once and refresh() swaps in only the cells that changed.
        # Rebuilding all 441 cells (and their images) on every refresh made the client
        # re-create hundreds of image widgets per turn; tiles flickered blank while
        # they reloaded and the browser's render process could crash under the churn.
        self.moves_by_cell: Dict[Tuple[int, int], List[int]] = {}
        self._cell_keys: Dict[Tuple[int, int], tuple] = {}
        self.grid_rows = [ft.Row([], spacing=0, tight=True) for _ in range(BOARD_SIZE)]
        self.grid_column = ft.Column(self.grid_rows, spacing=0, tight=True)
        # Drag to pan, mouse wheel / pinch to zoom. It keeps its transform while
        # refresh() swaps the rows inside it.
        self.board_viewer = ft.InteractiveViewer(
            content=self.grid_column,
            constrained=False,
            min_scale=MIN_BOARD_SCALE,
            max_scale=MAX_BOARD_SCALE,
            # Flutter web may report a mouse wheel as a trackpad; either way scrolling zooms.
            trackpad_scroll_causes_scale=True,
            boundary_margin=_margin(CELL_SIZE * 4),
            expand=True,
            on_size_change=self.on_board_resize,
        )
        self.board_panel = ft.Container(
            expand=True,
            padding=_padding(right=12),
            content=ft.Column(
                controls=[
                    ft.Row(
                        [
                            ft.Text("Carcassonne", size=24, weight=ft.FontWeight.BOLD),
                            ft.IconButton(ft.Icons.ZOOM_IN, tooltip="Zoom in", on_click=self.on_zoom_in),
                            ft.IconButton(ft.Icons.ZOOM_OUT, tooltip="Zoom out", on_click=self.on_zoom_out),
                            ft.IconButton(
                                ft.Icons.CENTER_FOCUS_STRONG, tooltip="Center on tiles", on_click=self.on_reset_view
                            ),
                        ],
                        vertical_alignment=ft.CrossAxisAlignment.CENTER,
                    ),
                    ft.Container(
                        expand=True,
                        bgcolor="#f3f4f6",
                        border=_border(1, "#d0d7de"),
                        border_radius=8,
                        clip_behavior=ft.ClipBehavior.HARD_EDGE,
                        content=self.board_viewer,
                    ),
                ],
                spacing=8,
                expand=True,
            ),
        )
        self.side_panel = ft.Container(
            width=340,
            padding=ft.Padding(left=8, top=8, right=8, bottom=8),
            border=_border(1, "#d0d7de"),
            border_radius=8,
            # Both panels stay mounted and only their visibility flips. Swapping
            # side_panel.content instead left the re-attached Record list frozen:
            # after "New game" it no longer showed any record.
            content=ft.Column([self.setup_column, self.game_column], spacing=0, expand=True),
        )
        self.root = ft.Row(
            controls=[self.board_panel, self.side_panel],
            expand=True,
            # Stretch so the side panel is window-tall and the Record box can fill the rest.
            vertical_alignment=ft.CrossAxisAlignment.STRETCH,
        )

        self.page.title = "Carcassonne"
        self.page.padding = 10
        self.page.add(self.root)

        if has_bot_player(self.player_specs):
            # Opponents given on the command line: skip the setup panel.
            self.start_game(self.player_specs, self.config.seed)
        else:
            self.show_setup()

    # ----------------------------------------------------------------- setup

    def _build_setup_panel(self) -> None:
        bot_index = next((i for i, spec in enumerate(self.player_specs) if spec.is_bot), 1)
        bot_spec = self.player_specs[bot_index]
        opponent = bot_spec.label if bot_spec.is_bot else "az"
        az_path = bot_spec.az_path or os.getenv("CARCASSONNE_AZ_PATH", "")
        checkpoint = bot_spec.az_checkpoint
        if checkpoint is None:
            checkpoint = int(os.getenv("CARCASSONNE_AZ_CHECKPOINT", "-1"))
        simulations = bot_spec.max_simulations or DEFAULT_SIMULATIONS

        self.seat_group = ft.RadioGroup(
            value="2" if bot_index == 0 else "1",
            content=ft.Column(
                [ft.Radio(value="1", label="Play first (P1)"), ft.Radio(value="2", label="Play second (P2)")],
                spacing=0,
            ),
        )
        self.opponent_dropdown = ft.Dropdown(
            label="Opponent",
            value=opponent,
            options=[ft.dropdown.Option(key=name, text=name) for name in OPPONENT_TYPES],
            width=300,
        )
        self.az_path_field = ft.TextField(label="Model directory (training --path)", value=az_path, width=300)
        self.checkpoint_field = ft.TextField(label="Checkpoint (-1 = latest)", value=str(checkpoint), width=300)
        self.simulations_field = ft.TextField(label="Simulations per move", value=str(simulations), width=300)
        self.seed_field = ft.TextField(label="Seed (0 = random)", value=str(self.config.seed or 0), width=300)
        # One dropdown per expansion: off, tiles (its tiles alone) or on (with its rules).
        self.expansion_dropdowns: Dict[str, ft.Dropdown] = {}
        for name in EXPANSION_NAMES[1:]:
            self.expansion_dropdowns[name] = ft.Dropdown(
                label=EXPANSION_LABELS.get(name, name),
                value=self.config.expansions.get(name, "off"),
                options=[ft.dropdown.Option(key=mode, text=mode) for mode in expansion_modes(name)],
                width=300,
            )
        self.setup_error = ft.Text("", color="#d1242f")

        self.setup_column = ft.Column(
            controls=[
                ft.Text("New game", size=20, weight=ft.FontWeight.BOLD),
                ft.Text("Your seat", weight=ft.FontWeight.W_600),
                self.seat_group,
                self.opponent_dropdown,
                self.az_path_field,
                self.checkpoint_field,
                self.simulations_field,
                self.seed_field,
                ft.Text("Expansions", weight=ft.FontWeight.W_600),
                *self.expansion_dropdowns.values(),
                BUTTON("Start game", on_click=self.on_setup_start),
                self.setup_error,
            ],
            spacing=10,
            scroll=ft.ScrollMode.AUTO,
            expand=True,
        )

    def _show_side_panel(self, setup: bool) -> None:
        self.setup_column.visible = setup
        self.game_column.visible = not setup

    def show_setup(self) -> None:
        self._show_side_panel(setup=True)
        self.refresh()

    def on_setup_start(self, _: ft.ControlEvent) -> None:
        try:
            checkpoint = int(self.checkpoint_field.value or "-1")
            simulations = int(self.simulations_field.value or DEFAULT_SIMULATIONS)
            seed = int(self.seed_field.value or "0")
            if simulations < 1:
                raise ValueError("Simulations must be positive.")
            specs = build_player_specs(
                human_player=int(self.seat_group.value or "1"),
                opponent=self.opponent_dropdown.value or "az",
                az_path=self.az_path_field.value or "",
                az_checkpoint=checkpoint,
                max_simulations=simulations,
            )
        except ValueError as exc:
            self.setup_error.value = str(exc)
            self.page.update()
            return
        expansions = {
            name: dropdown.value
            for name, dropdown in self.expansion_dropdowns.items()
            if dropdown.value not in (None, "off")
        }
        self.setup_error.value = ""
        self.start_game(specs, None if seed == 0 else seed, expansions)

    # ------------------------------------------------------------------ game

    def _build_game_panel(self) -> None:
        self.status = ft.Text("Ready.")
        self.ai_text = ft.Text("")
        self.turn_text = ft.Text()
        self.score_text = ft.Text()
        self.meeple_text = ft.Text()
        self.piece_legend = ft.Text(size=12, color="#57606a", visible=False)
        self.goods_text = ft.Text(visible=False)
        self.dragon_text = ft.Text(visible=False)
        self.fairy_text = ft.Text(visible=False)
        self.value_text = ft.Text(visible=False)
        self.thinking_row = ft.Row(
            [ft.ProgressRing(width=16, height=16, stroke_width=2), ft.Text("AI thinking...")],
            visible=False,
        )
        self.holding_image = ft.Image(src="tiles/1.png", width=100, height=100, fit=IMAGE_FIT.CONTAIN)
        self.records_column = ft.Column(spacing=4, scroll=ft.ScrollMode.AUTO, expand=True)

        self.start_game_btn = BUTTON("Start Game", on_click=self.on_start_game)
        self.new_game_btn = ft.OutlinedButton("New game", on_click=self.on_new_game)
        self.confirm_btn = BUTTON("Confirm Tile", on_click=self.on_confirm_tile)
        self.skip_btn = ft.OutlinedButton("Skip Meeple", on_click=lambda _: self.on_apply_move(-1))
        # The meeple phase's tabs, built for each game with the pieces and choices by
        # cell it has (_build_meeple_tabs).
        self.meeple_tabs_slot = ft.Container()
        self._build_meeple_tabs(PIECE_KINDS[:1])
        # The second step of a choice by cell: one button per move on the cell chosen.
        self.spot_row = ft.Row([], wrap=True, visible=False)
        self._spot_key: Optional[tuple] = None

        self.game_column = ft.Column(
            controls=[
                ft.Text("Game Info", size=20, weight=ft.FontWeight.BOLD),
                ft.Row([self.start_game_btn, self.new_game_btn], wrap=True),
                self.turn_text,
                self.ai_text,
                self.thinking_row,
                self.status,
                ft.Text("Tile in hand (click a green cell again to rotate)", weight=ft.FontWeight.W_600),
                ft.Container(
                    width=120,
                    height=120,
                    border=_border(1, "#cccccc"),
                    alignment=ALIGN_CENTER,
                    content=self.holding_image,
                ),
                self.score_text,
                self.value_text,
                self.meeple_text,
                self.piece_legend,
                self.goods_text,
                self.dragon_text,
                self.fairy_text,
                ft.Row([self.confirm_btn, self.skip_btn], wrap=True),
                self.meeple_tabs_slot,
                self.spot_row,
                ft.Text("Record", size=18, weight=ft.FontWeight.BOLD),
                # Takes whatever height is left and scrolls on its own. A fixed-height box at
                # the end of a scrolling column fell off the bottom of shorter windows.
                ft.Container(
                    width=320,
                    expand=True,
                    border=_border(1, "#d0d7de"),
                    border_radius=4,
                    padding=ft.Padding(left=6, top=6, right=6, bottom=6),
                    content=self.records_column,
                ),
            ],
            spacing=8,
            expand=True,
        )

    def _build_meeple_tabs(self, kinds: Sequence[str]) -> None:
        """A tab per piece and per choice by cell; a piece's tab holds its buttons, a
        choice's a hint, and only the selected tab's show. The tab bar shows only with
        more than one tab, so a base game looks as it always did."""
        self.tab_kinds = list(kinds)
        self.meeple_tab = self.tab_kinds[0]
        self.meeple_buttons = {
            pos: BUTTON(label, on_click=lambda _, pos=pos: self.on_apply_move(pos))
            for pos, label in meeple_button_labels().items()
            if meeple_piece(pos) in self.tab_kinds
        }
        self.meeple_groups = {kind: ft.Row([], wrap=True) for kind in PIECE_KINDS if kind in self.tab_kinds}
        for pos, button in self.meeple_buttons.items():
            self.meeple_groups[meeple_piece(pos)].controls.append(button)
        self.choice_hint = ft.Text(size=12, color="#57606a", visible=False)
        # Flet draws a disabled tab like any other, so its label is greyed out by hand.
        labels = {**PIECE_TAB_LABELS, **CHOICE_TAB_LABELS}
        self.meeple_tab_labels = {kind: ft.Text(labels[kind]) for kind in self.tab_kinds}
        self.meeple_tab_bar = ft.TabBar(
            tabs=[ft.Tab(label=self.meeple_tab_labels[kind]) for kind in self.tab_kinds],
            label_padding=_padding(left=4, right=4),
            # Seven tabs do not fit the side panel.
            scrollable=len(self.tab_kinds) > len(PIECE_KINDS),
            visible=False,
        )
        self.meeple_tabs = ft.Tabs(
            length=len(self.tab_kinds),
            selected_index=0,
            on_change=self.on_meeple_tab_change,
            content=ft.Column(
                [self.meeple_tab_bar, *self.meeple_groups.values(), self.choice_hint], spacing=4, tight=True
            ),
        )
        self.meeple_tabs_slot.content = self.meeple_tabs

    def start_game(
        self,
        player_specs: Tuple[PlayerSpec, PlayerSpec],
        seed: Optional[int],
        expansions: Optional[Dict[str, str]] = None,
    ) -> None:
        self._close_engine()
        self.player_specs = player_specs
        self.bot_game_started = False
        self.status.value = "Ready."
        # The UI runs bot turns itself (off the UI thread), so the adapter must not.
        self.engine = CppCarcassonneAdapter(
            seed=seed,
            player_specs=player_specs,
            auto_run_bots=False,
            expansions=self.config.expansions if expansions is None else expansions,
        )
        self.state = self.engine.state
        self._build_meeple_tabs(["meeple", *self.state.piece_kinds, *self.state.choice_kinds])
        self.status.value = self._prompt() or "Ready."
        self._show_side_panel(setup=False)
        self.refresh()
        self._center_board()
        self._start_ai_turns()

    def _close_engine(self) -> None:
        if self.engine is not None:
            self.engine.close()
        self.engine = None
        self.state = None
        self.shown_view_origin = None
        self.ai_running = False
        self.selected_move = None

    def on_new_game(self, _: ft.ControlEvent) -> None:
        self._close_engine()
        self.show_setup()

    def _human_turn(self) -> bool:
        return (
            self.engine is not None
            and not self.ai_running
            and not self.state.game_over
            and not self.engine.is_ai_turn()
        )

    def refresh(self) -> None:
        if self.engine is None:
            self.moves_by_cell = {}
            self.cell_highlights = {}
            self._render_grid()
            self.page.update()
            return

        self.state = self.engine.state
        # While the AI thread runs, leave the engine alone and show no moves.
        decision = None if self.ai_running else self.engine.decision()

        moves_by_cell: Dict[Tuple[int, int], List[int]] = {}
        for move in self.engine.get_valid_moves() if decision == "tile" else []:
            moves_by_cell.setdefault((move.x, move.y), []).append(move.rotation)
        for pos in moves_by_cell:
            moves_by_cell[pos].sort()
        self.moves_by_cell = moves_by_cell
        meeple_options = self.engine.meeple_options() if decision == "piece" else []
        self.cell_options = self.engine.cell_options() if decision == "piece" else {}
        self.dragon_moves = {cell: side for side, cell in self.engine.dragon_options()} if decision == "dragon" else {}
        highlights = {cell: "tile" for cell in moves_by_cell}
        highlights.update({cell: self.meeple_tab for cell in self.cell_options.get(self.meeple_tab, [])})
        highlights.update({cell: "dragon" for cell in self.dragon_moves})
        spot_choice, spot_positions = self.engine.spot_options() if decision == "spot" else ("", [])
        spot_cell = self.engine.spot_cell() if decision == "spot" else None
        if spot_cell is not None:
            highlights[spot_cell] = spot_choice  # the tile whose spot is being chosen
        self.cell_highlights = highlights

        self.confirm_btn.disabled = self.selected_move is None or decision != "tile"
        self.skip_btn.visible = decision == "piece"
        self.start_game_btn.visible = should_show_start_game(self.player_specs, self.bot_game_started)
        self.start_game_btn.disabled = self.state.game_over
        self.thinking_row.visible = self.ai_running and not self.state.game_over

        for pos, btn in self.meeple_buttons.items():
            btn.visible = decision == "piece" and pos in meeple_options
        offered = offered_tabs(meeple_options, self.cell_options)
        for kind, tab in zip(self.tab_kinds, self.meeple_tab_bar.tabs):
            tab.disabled = kind not in offered
            self.meeple_tab_labels[kind].color = None if kind in offered else "#c4c9d0"
        for kind, group in self.meeple_groups.items():
            group.visible = decision == "piece" and kind == self.meeple_tab
        self.choice_hint.visible = decision == "piece" and self.meeple_tab in CHOICE_KINDS
        self.choice_hint.value = CHOICE_HINTS.get(self.meeple_tab, "")
        # No tabs in a game with only meeples, or when only skip is left.
        self.meeple_tab_bar.visible = decision == "piece" and len(self.tab_kinds) > 1 and bool(offered)
        self._show_spot_buttons(spot_choice, spot_positions)

        seat = human_seat(self.player_specs)
        player_label = f"P{self.state.current_player}"
        if seat is not None:
            player_label += " (you)" if self.state.current_player == seat else " (AI)"
        self.turn_text.value = f"Turn: {self.state.turn} | To move: {player_label}"
        if self.state.builder_extra_tile:
            self.turn_text.value += " · builder: extra tile"
        if self.state.phase == PHASE_DRAGON:
            self.turn_text.value += f" · dragon step {self.state.dragon_steps + 1}/{DRAGON_STEPS}"
        self.ai_text.value = f"Mode: {self.engine.mode_label()}"
        if self.engine.ai_status:
            self.ai_text.value += f"\nAI: {summarize_ai_status(self.engine.ai_status)}"

        if self.state.holding_tile_id is None:
            self.holding_image.visible = False
        else:
            self.holding_image.src = f"tiles/{self.state.holding_tile_id}.png"
            self.holding_image.visible = True
        rotation = self.selected_move.rotation if self.selected_move is not None else 0
        self.holding_image.rotate = ft.Rotate(angle=rotation * QUARTER_TURN)

        self.score_text.value = f"Scores -> P1: {self.state.scores[1]} | P2: {self.state.scores[2]}"
        self.meeple_text.spans = self._meeple_spans()
        kinds = ("meeple", *self.state.piece_kinds)
        self.piece_legend.visible = bool(self.state.piece_kinds)
        self.piece_legend.value = "  ".join(f"{PIECE_GLYPHS[kind]} {PIECE_TAB_LABELS[kind].lower()}" for kind in kinds)
        self.goods_text.visible = bool(self.state.goods)
        self.goods_text.value = "\n".join(
            f"P{player} goods  {format_goods(goods)}" for player, goods in sorted(self.state.goods.items())
        )
        # The Princess & the Dragon's rules come together: the choices by cell, the dragon, the fairy.
        self.dragon_text.visible = bool(self.state.choice_kinds)
        self.dragon_text.value = dragon_status(self.state)
        self.fairy_text.visible = "fairy" in self.state.choice_kinds
        self.fairy_text.value = fairy_status(self.state)
        bot_value = self.engine.last_bot_value
        self.value_text.visible = bot_value is not None
        self.value_text.value = format_bot_value(bot_value) if bot_value is not None else ""
        record_controls: List[ft.Control] = [
            ft.Text(format_move_record(record), selectable=True) for record in self.engine.move_records
        ]
        self.records_column.controls = record_controls or [ft.Text("No records.")]

        self._render_grid()
        # The engine's view moves to stay centred on the tiles, so the grid shifts
        # under them; pan the other way so they stay put on screen, keeping any zoom.
        if self.shown_view_origin is not None and self.engine.view_origin != self.shown_view_origin:
            self.page.run_task(self._pan_board, *view_shift_pan(self.shown_view_origin, self.engine.view_origin))
        self.shown_view_origin = self.engine.view_origin

        if self.state.game_over:
            self.status.value = f"Game over. {self._result_text()} {self._saved_text()}"

        self.page.update()

    def _show_spot_buttons(self, choice: str, positions: List[int]) -> None:
        key = (choice, tuple(positions))
        if key != self._spot_key:
            self._spot_key = key
            self.spot_row.controls = [
                BUTTON(spot_button_label(choice, pos), on_click=lambda _, pos=pos: self.on_spot(pos))
                for pos in positions
            ]
        self.spot_row.visible = bool(positions)

    def _prompt(self) -> str:
        """What the human is to do next, if anything."""
        decision = self.engine.decision() if self.engine is not None else None
        return {
            "tile": "Your turn.",
            "piece": "Choose meeple position or skip.",
            "spot": "Choose which one on that tile.",
            "dragon": "Your dragon step: click an orange tile.",
        }.get(decision, "")

    def _meeple_spans(self) -> List[ft.TextSpan]:
        """A symbol per piece still in hand, in the player's colour (pieces_in_hand)."""
        spans: List[ft.TextSpan] = []
        for player in (1, 2):
            pieces = pieces_in_hand(self.state.meeples_remaining[player], self.state.pieces_remaining.get(player, {}))
            spans.append(ft.TextSpan(f"{'' if player == 1 else chr(10)}P{player} meeples  "))
            spans.append(
                ft.TextSpan(pieces, style=ft.TextStyle(color=self._player_color(player), letter_spacing=1))
            )
        return spans

    def _result_text(self) -> str:
        scores = self.state.scores
        seat = human_seat(self.player_specs)
        if scores[1] == scores[2]:
            return "Draw."
        winner = 1 if scores[1] > scores[2] else 2
        if seat is None:
            return f"P{winner} wins."
        return "You win!" if winner == seat else "You lose."

    def _saved_text(self) -> str:
        if self.engine.saved_game_path is not None:
            return f"Saved to {self.engine.saved_game_path}"
        return self.engine.save_error

    def _render_grid(self) -> None:
        for y, row in enumerate(self.grid_rows):
            if not row.controls:
                row.controls = [ft.Container() for _ in range(BOARD_SIZE)]
            for x in range(BOARD_SIZE):
                key = self._cell_key(x, y)
                if self._cell_keys.get((x, y)) != key:
                    row.controls[x] = self._build_cell(x, y)
                    self._cell_keys[(x, y)] = key

    def _cell_tile(self, x: int, y: int):
        if self.engine is None:
            return None
        return self.state.board.get(self.engine.to_engine_coords(x, y))

    def _cell_key(self, x: int, y: int) -> tuple:
        """Everything _build_cell draws from, so an unchanged cell can be kept."""
        tile = self._cell_tile(x, y)
        tile_key = None
        if tile is not None:
            tile_key = (tile.tile_id, tile.rotation, tile.tile_owner, tuple(tile.meeple_markers))
        is_selected = self.selected_move is not None and (self.selected_move.x, self.selected_move.y) == (x, y)
        preview = None
        if is_selected and self.state is not None:
            preview = (self.state.holding_tile_id, self.selected_move.rotation)
        return tile_key, self.cell_highlights.get((x, y)), is_selected, preview, self._cell_figures(x, y)

    def _cell_figures(self, x: int, y: int) -> tuple:
        """The dragon and the fairy on this cell: dragon here, visited by the dragon's
        move under way, and the fairy's meeple spot (None for no fairy)."""
        if self.engine is None or self.state is None:
            return False, False, None
        cell = self.engine.to_engine_coords(x, y)
        fairy = self.state.fairy
        return (
            self.state.dragon == cell,
            cell in self.state.dragon_visited,
            fairy[2] if fairy is not None and fairy[:2] == cell else None,
        )

    def _build_cell(self, x: int, y: int) -> ft.Container:
        pos = (x, y)
        tile = self._cell_tile(x, y)
        highlight_kind = self.cell_highlights.get(pos)
        is_selected = self.selected_move is not None and (self.selected_move.x, self.selected_move.y) == pos
        dragon_here, visited, fairy_spot = self._cell_figures(x, y)

        bg = "#ffffff"
        border_color = "#e5e7eb"
        border_width = 1
        if highlight_kind == "tile":
            bg = TILE_MOVE_BG
            border_color = TILE_MOVE_BORDER
        if is_selected:
            bg = "#fff3cd"
            border_color = "#d18e00"

        # Tiles fill the whole cell with no border of their own, so placed tiles touch.
        # Highlights are drawn as an overlay on top instead of shrinking the tile.
        layers: List[ft.Control] = []
        highlight: Optional[Tuple[int, str]] = None
        if tile is not None:
            layers.append(self._build_tile_image(tile.tile_id, tile.rotation))
            for owner, pos_marker in tile.meeple_markers:
                layers.append(self._build_meeple_marker(owner, pos_marker))
            if visited:
                layers.append(self._build_tint(DRAGON_VISITED_COLOR, 0.35))
            if fairy_spot is not None:
                layers.append(self._build_figure("fairy", fairy_alignment(fairy_spot), FAIRY_SIZE, FAIRY_SIZE))
            if dragon_here:
                layers.append(self._build_figure("dragon", (0.0, 0.0), DRAGON_WIDTH, DRAGON_HEIGHT))
            if highlight_kind in HIGHLIGHT_COLORS:
                # A tile a choice by cell or the dragon can take.
                layers.append(self._build_tint(HIGHLIGHT_COLORS[highlight_kind], 0.25))
                highlight = (3, HIGHLIGHT_COLORS[highlight_kind])
            elif tile.tile_owner is not None:
                highlight = (3, self._player_color(tile.tile_owner))
        elif is_selected and self.state.holding_tile_id is not None and self.selected_move is not None:
            layers.append(
                ft.Container(
                    opacity=0.55,
                    content=self._build_tile_image(self.state.holding_tile_id, self.selected_move.rotation),
                )
            )
            highlight = (2, border_color)

        on_click = lambda _: self.on_cell_click(x, y)
        if not layers:
            return ft.Container(
                width=CELL_SIZE,
                height=CELL_SIZE,
                bgcolor=bg,
                border=_border(border_width, border_color),
                on_click=on_click,
            )
        if highlight is not None:
            width, color = highlight
            layers.append(ft.Container(width=CELL_SIZE, height=CELL_SIZE, border=_border(width, color)))
        return ft.Container(
            width=CELL_SIZE,
            height=CELL_SIZE,
            bgcolor=bg,
            content=ft.Stack(controls=layers, width=CELL_SIZE, height=CELL_SIZE),
            on_click=on_click,
        )

    def _build_tile_image(self, tile_id: int, rotation: int) -> ft.Image:
        # The tile art is not quite square (about 156x151); FILL stretches it over the
        # whole cell so there is no strip of background between neighbouring tiles.
        return ft.Image(
            src=f"tiles/{tile_id}.png",
            width=CELL_SIZE,
            height=CELL_SIZE,
            fit=IMAGE_FIT.FILL,
            rotate=ft.Rotate(angle=rotation * QUARTER_TURN),
        )

    def _player_color(self, owner: int) -> str:
        return "#3b82f6" if owner == 1 else "#ef4444"

    def _build_meeple_marker(self, owner: int, meeple_pos: int) -> ft.Container:
        """A standing meeple on a road, city or monastery, a lying one for a farmer (the big
        meeple drawn larger), and the builder or pig on its feature or field."""
        image, size = meeple_marker(meeple_pos)
        align_x, align_y = meeple_alignment(meeple_spot(meeple_pos))
        return ft.Container(
            width=CELL_SIZE,
            height=CELL_SIZE,
            alignment=ft.alignment.Alignment(align_x, align_y),
            content=ft.Image(
                src=f"meeples/{image}_p{owner}.png",
                width=size,
                height=size,
                fit=IMAGE_FIT.CONTAIN,
            ),
        )

    def _build_tint(self, color: str, opacity: float) -> ft.Container:
        return ft.Container(width=CELL_SIZE, height=CELL_SIZE, bgcolor=color, opacity=opacity)

    def _build_figure(self, image: str, alignment: Tuple[float, float], width: int, height: int) -> ft.Container:
        """The dragon or the fairy (meeples/<image>.png), which belong to no one."""
        return ft.Container(
            width=CELL_SIZE,
            height=CELL_SIZE,
            alignment=ft.alignment.Alignment(*alignment),
            content=ft.Image(src=f"meeples/{image}.png", width=width, height=height, fit=IMAGE_FIT.CONTAIN),
        )

    # InteractiveViewer.zoom() scales about the content origin, and Python cannot read
    # the transform back after the user drags or wheels. So the buttons start from a
    # known transform: reset to identity, zoom, then pan the placed tiles to the middle.

    def on_board_resize(self, e: ft.LayoutSizeChangeEvent) -> None:
        self.board_viewport = (e.width, e.height)
        if self.center_board_pending:
            self._center_board()

    def _center_board(self) -> None:
        self.board_zoom = 1.0
        if self.board_viewport is None:
            self.center_board_pending = True  # on_board_resize finishes it
            return
        self.center_board_pending = False
        self.page.run_task(self._show_board, self.board_zoom)

    def _tiles_center(self) -> Tuple[float, float]:
        cells = list(self.state.board) if self.state is not None else []
        if not cells:
            return BOARD_SIZE * CELL_SIZE / 2, BOARD_SIZE * CELL_SIZE / 2
        origin_x, origin_y = self.engine.view_origin
        xs = [x - origin_x for x, _ in cells]
        ys = [y - origin_y for _, y in cells]
        return (min(xs) + max(xs) + 1) * CELL_SIZE / 2, (min(ys) + max(ys) + 1) * CELL_SIZE / 2

    # Each of these runs several viewer calls in a row; the lock keeps two of them
    # (a zoom click and a view shift, say) from interleaving and compounding.

    async def _show_board(self, zoom: float) -> None:
        async with self._view_lock:
            await self.board_viewer.reset()
            if zoom != 1.0:
                await self.board_viewer.zoom(zoom)
            if self.board_viewport is not None:
                width, height = self.board_viewport
                center_x, center_y = self._tiles_center()
                # pan() moves in content units: the viewport shows s * (p + d), s = zoom.
                await self.board_viewer.pan(width / 2 / zoom - center_x, height / 2 / zoom - center_y)

    async def _pan_board(self, dx: float, dy: float) -> None:
        async with self._view_lock:
            await self.board_viewer.pan(dx, dy)

    def _step_zoom(self, factor: float) -> None:
        self.board_zoom = min(MAX_BOARD_SCALE, max(MIN_BOARD_SCALE, self.board_zoom * factor))
        self.page.run_task(self._show_board, self.board_zoom)

    def on_zoom_in(self, _: ft.ControlEvent) -> None:
        self._step_zoom(ZOOM_STEP)

    def on_zoom_out(self, _: ft.ControlEvent) -> None:
        self._step_zoom(1 / ZOOM_STEP)

    def on_reset_view(self, _: ft.ControlEvent) -> None:
        self.board_zoom = 1.0
        self.page.run_task(self._show_board, self.board_zoom)

    def on_cell_click(self, x: int, y: int) -> None:
        if not self._human_turn():
            return
        decision = self.engine.decision()
        if decision == "dragon":
            side = self.dragon_moves.get((x, y))
            if side is not None:
                self._apply_move(lambda: self.engine.apply_dragon(side))
            return
        if decision == "piece":
            choice = self.meeple_tab
            if (x, y) in self.cell_options.get(choice, []):
                self._apply_move(lambda: self.engine.apply_cell(choice, x, y))
            return
        if decision != "tile":
            return
        rots = self.moves_by_cell.get((x, y))
        if not rots:
            return

        if self.selected_move and (self.selected_move.x, self.selected_move.y) == (x, y):
            current_index = rots.index(self.selected_move.rotation)
            next_rotation = rots[(current_index + 1) % len(rots)]
            self.selected_move = Move(x=x, y=y, rotation=next_rotation)
        else:
            self.selected_move = Move(x=x, y=y, rotation=rots[0])

        self.status.value = f"Selected ({x}, {y}) rotation={self.selected_move.rotation}."
        self.refresh()

    def on_confirm_tile(self, _: ft.ControlEvent) -> None:
        if self.selected_move is None or not self._human_turn():
            return
        try:
            meeple_options = self.engine.confirm_tile(self.selected_move)
        except ValueError as exc:
            self.status.value = str(exc)
            self.selected_move = None
            self.refresh()
            return
        # The tile is down. The selection names a cell of the view, which may move
        # now and leave it highlighting an empty cell.
        self.selected_move = None

        # Open the first tab with a move: the meeple's, unless only an expansion's piece
        # or a choice by cell is left.
        offered = offered_tabs(meeple_options, self.engine.cell_options())
        self._select_meeple_tab(offered[0] if offered else self.tab_kinds[0])
        self.status.value = self._prompt()
        self.refresh()

    def _select_meeple_tab(self, kind: str) -> None:
        self.meeple_tab = kind
        self.meeple_tabs.selected_index = self.tab_kinds.index(kind)

    def on_meeple_tab_change(self, e: ft.ControlEvent) -> None:
        index = e.control.selected_index
        if index is None and e.data is not None:
            index = int(e.data)
        kind = self.tab_kinds[int(index or 0)]
        offered = offered_tabs(self.engine.meeple_options(), self.engine.cell_options()) if self.engine else []
        # A tab with no move stays shut: go back to the one that was open.
        self._select_meeple_tab(kind if kind in offered else self.meeple_tab)
        self.refresh()

    def on_apply_move(self, meeple_pos: int) -> None:
        if self.engine is None or self.engine.decision() != "piece":
            return
        self._apply_move(lambda: self.engine.apply_meeple(meeple_pos))

    def on_spot(self, pos: int) -> None:
        if self.engine is None or self.engine.decision() != "spot":
            return
        self._apply_move(lambda: self.engine.apply_spot(pos))

    def _apply_move(self, apply: Callable[[], None]) -> None:
        """A human move after the tile; then the bot's turn, if it is one."""
        old_scores = dict(self.state.scores)
        try:
            apply()
        except ValueError as exc:
            self.status.value = str(exc)
            self.refresh()
            return

        self.state = self.engine.state
        p1_gain = self.state.scores[1] - old_scores[1]
        p2_gain = self.state.scores[2] - old_scores[2]
        if p1_gain or p2_gain:
            self.status.value = f"Scored: P1 +{p1_gain}, P2 +{p2_gain}"
        else:
            self.status.value = self._prompt() or "Move applied."

        self.selected_move = None
        self.refresh()
        self._start_ai_turns()

    def _start_ai_turns(self) -> None:
        """Let the bot reply in the background when it is a human-vs-bot game."""
        if self.engine is None or self.ai_running or is_bot_vs_bot(self.player_specs):
            return
        if self.state.game_over or not self.engine.is_ai_turn():
            return
        self.ai_running = True
        self.refresh()
        self.page.run_task(self._run_ai_turns, self.engine)

    async def _run_ai_turns(self, engine: CppCarcassonneAdapter) -> None:
        try:
            while self.engine is engine and not engine.state.game_over and engine.is_ai_turn():
                played_turns = await asyncio.to_thread(engine.run_ai_turns, 1)
                if self.engine is not engine:
                    return  # "New game" was pressed while the bot was thinking.
                self.state = engine.state
                self.refresh()
                if played_turns <= 0:
                    break
        finally:
            if self.engine is engine:
                self.ai_running = False
                if not engine.state.game_over:
                    if engine.is_ai_turn():
                        # A bot that is still to move here failed; ai_status holds why.
                        print(f"Bot failed: {engine.ai_status}", flush=True)
                        self.status.value = summarize_ai_status(engine.ai_status)
                    else:
                        self.status.value = self._prompt()
                self.refresh()

    def on_start_game(self, _: ft.ControlEvent) -> None:
        if self.engine is None or not is_bot_vs_bot(self.player_specs) or self.bot_game_started:
            return
        self.bot_game_started = True
        self.engine.ai_status = "Bot game running..."
        self.refresh()
        self.page.run_task(self._run_bot_game, self.engine)

    async def _run_bot_game(self, engine: CppCarcassonneAdapter) -> None:
        while self.engine is engine and not engine.state.game_over and engine.is_ai_turn():
            played_turns = await asyncio.to_thread(engine.run_ai_turns, 1)
            if self.engine is not engine:
                return
            self.state = engine.state
            self.selected_move = None
            self.refresh()
            if played_turns <= 0:
                break
            await asyncio.sleep(0.2)


def main(page: ft.Page, config: Optional[PlayUiConfig] = None) -> None:
    CarcassonneUI(page, config)


def run_app(argv: Optional[Sequence[str]] = None) -> None:
    config = parse_ui_config(argv)
    assets_dir = str(Path(__file__).resolve().parent.parent.parent)
    mode = os.getenv("CARCASSONNE_UI_MODE", "web").lower()
    app_main = lambda page: main(page, config)
    if mode == "desktop":
        ft.run(app_main, assets_dir=assets_dir, view=ft.AppView.FLET_APP)
        return

    host = os.getenv("CARCASSONNE_UI_HOST", "127.0.0.1")
    port = int(os.getenv("CARCASSONNE_UI_PORT", os.getenv("FLET_SERVER_PORT", "8550")))
    os.environ.setdefault("FLET_FORCE_WEB_SERVER", "true")
    print(f"Carcassonne UI: http://{host}:{port}")
    ft.run(app_main, assets_dir=assets_dir, host=host, port=port, view=ft.AppView.WEB_BROWSER)
