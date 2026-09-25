from __future__ import annotations

import argparse
import asyncio
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import flet as ft

try:
    from domain import Move, MoveRecord
    from engine import BOARD_SIZE, CppCarcassonneAdapter, PlayerSpec
except ImportError:  # pragma: no cover - package import fallback
    from ..domain import Move, MoveRecord
    from ..engine import BOARD_SIZE, CppCarcassonneAdapter, PlayerSpec


IMAGE_FIT = getattr(ft, "ImageFit", ft.BoxFit)
# Flet 1.0 dropped ElevatedButton; older releases have both.
BUTTON = getattr(ft, "Button", None) or ft.ElevatedButton
ALIGN_CENTER = ft.alignment.Alignment(0, 0)
ALIGN_TOP_CENTER = ft.alignment.Alignment(0, -1)
ALIGN_CENTER_RIGHT = ft.alignment.Alignment(1, 0)
ALIGN_BOTTOM_CENTER = ft.alignment.Alignment(0, 1)
ALIGN_CENTER_LEFT = ft.alignment.Alignment(-1, 0)

CELL_SIZE = 40
MEEPLE_SIZE = 10
MEEPLE_GLYPH = "■"  # ■
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
    )


def format_move_record(record: MoveRecord) -> str:
    base = f"P{record.player}({record.tile_id},{record.x},{record.y},{record.rotation},{record.meeple_pos})"
    nonzero_deltas = {player: delta for player, delta in sorted(record.score_deltas.items()) if delta}
    if not nonzero_deltas:
        return f"{base} +0(得分)"
    if set(nonzero_deltas) == {record.player}:
        points = nonzero_deltas[record.player]
        return f"{base} {points:+d}(得分)"

    deltas = "/".join(f"P{player}{delta:+d}" for player, delta in nonzero_deltas.items())
    return f"{base} {deltas}(得分)"


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
        self.awaiting_meeple = False
        self.meeple_options: List[int] = []
        self.board_zoom = 1.0
        self.board_viewport: Optional[Tuple[float, float]] = None
        self.center_board_pending = True

        self._build_setup_panel()
        self._build_game_panel()

        # Rows and cells sit edge to edge (spacing 0) so neighbouring tiles touch.
        self.grid_column = ft.Column(spacing=0, tight=True)
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
        self.setup_error.value = ""
        self.start_game(specs, None if seed == 0 else seed)

    # ------------------------------------------------------------------ game

    def _build_game_panel(self) -> None:
        self.status = ft.Text("Ready.")
        self.ai_text = ft.Text("")
        self.turn_text = ft.Text()
        self.score_text = ft.Text()
        self.meeple_text = ft.Text()
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
        self.meeple_buttons = {
            0: BUTTON("Meeple: Up", on_click=lambda _: self.on_apply_move(0)),
            1: BUTTON("Meeple: Right", on_click=lambda _: self.on_apply_move(1)),
            2: BUTTON("Meeple: Down", on_click=lambda _: self.on_apply_move(2)),
            3: BUTTON("Meeple: Left", on_click=lambda _: self.on_apply_move(3)),
            4: BUTTON("Meeple: Center", on_click=lambda _: self.on_apply_move(4)),
        }

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
                self.meeple_text,
                ft.Row([self.confirm_btn, self.skip_btn], wrap=True),
                ft.Row(list(self.meeple_buttons.values()), wrap=True),
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

    def start_game(self, player_specs: Tuple[PlayerSpec, PlayerSpec], seed: Optional[int]) -> None:
        self._close_engine()
        self.player_specs = player_specs
        self.bot_game_started = False
        self.status.value = "Ready."
        # The UI runs bot turns itself (off the UI thread), so the adapter must not.
        self.engine = CppCarcassonneAdapter(seed=seed, player_specs=player_specs, auto_run_bots=False)
        self.state = self.engine.state
        self._show_side_panel(setup=False)
        self.refresh()
        self._center_board()
        self._start_ai_turns()

    def _close_engine(self) -> None:
        if self.engine is not None:
            self.engine.close()
        self.engine = None
        self.state = None
        self.ai_running = False
        self.selected_move = None
        self.awaiting_meeple = False
        self.meeple_options = []

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
            self.grid_column.controls = [self._build_row(y, {}) for y in range(BOARD_SIZE)]
            self.page.update()
            return

        self.state = self.engine.state
        # While the AI thread runs, leave the engine alone and show no moves.
        valid_moves = [] if self.ai_running else self.engine.get_valid_moves()

        moves_by_cell: Dict[Tuple[int, int], List[int]] = {}
        for move in valid_moves:
            moves_by_cell.setdefault((move.x, move.y), []).append(move.rotation)
        for pos in moves_by_cell:
            moves_by_cell[pos].sort()

        human_turn = self._human_turn()
        self.confirm_btn.disabled = self.selected_move is None or self.awaiting_meeple or not human_turn
        self.skip_btn.visible = self.awaiting_meeple and human_turn
        self.start_game_btn.visible = should_show_start_game(self.player_specs, self.bot_game_started)
        self.start_game_btn.disabled = self.state.game_over
        self.thinking_row.visible = self.ai_running and not self.state.game_over

        for pos, btn in self.meeple_buttons.items():
            btn.visible = self.awaiting_meeple and pos in self.meeple_options
            btn.disabled = not human_turn

        seat = human_seat(self.player_specs)
        player_label = f"P{self.state.current_player}"
        if seat is not None:
            player_label += " (you)" if self.state.current_player == seat else " (AI)"
        self.turn_text.value = f"Turn: {self.state.turn} | To move: {player_label}"
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
        record_controls: List[ft.Control] = [
            ft.Text(format_move_record(record), selectable=True) for record in self.engine.move_records
        ]
        self.records_column.controls = record_controls or [ft.Text("No records.")]

        self.grid_column.controls = [self._build_row(y, moves_by_cell) for y in range(BOARD_SIZE)]

        if self.state.game_over:
            self.status.value = f"Game over. {self._result_text()}"
            self.awaiting_meeple = False

        self.page.update()

    def _meeple_spans(self) -> List[ft.TextSpan]:
        """One square per meeple still in hand, in the player's colour."""
        spans: List[ft.TextSpan] = []
        for player in (1, 2):
            left = self.state.meeples_remaining[player]
            spans.append(ft.TextSpan(f"{'' if player == 1 else chr(10)}P{player} meeples  "))
            spans.append(
                ft.TextSpan(
                    MEEPLE_GLYPH * left if left else "-",
                    style=ft.TextStyle(color=self._player_color(player), letter_spacing=1),
                )
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

    def _build_row(self, y: int, moves_by_cell: Dict[Tuple[int, int], List[int]]) -> ft.Row:
        row_controls = [self._build_cell(x, y, moves_by_cell) for x in range(BOARD_SIZE)]
        return ft.Row(row_controls, spacing=0, tight=True)

    def _build_cell(self, x: int, y: int, moves_by_cell: Dict[Tuple[int, int], List[int]]) -> ft.Container:
        pos = (x, y)
        tile = None
        if self.engine is not None:
            tile = self.state.board.get(self.engine.to_engine_coords(x, y))
        is_valid = pos in moves_by_cell
        is_selected = self.selected_move is not None and (self.selected_move.x, self.selected_move.y) == pos

        bg = "#ffffff"
        border_color = "#e5e7eb"
        border_width = 1
        if is_valid:
            bg = "#ecfdf3"
            border_color = "#63b36f"
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
            if tile.tile_owner is not None:
                highlight = (3, self._player_color(tile.tile_owner))
        elif is_selected and self.state.holding_tile_id is not None and self.selected_move is not None:
            layers.append(
                ft.Container(
                    opacity=0.55,
                    content=self._build_tile_image(self.state.holding_tile_id, self.selected_move.rotation),
                )
            )
            highlight = (2, border_color)

        on_click = lambda _: self.on_cell_click(x, y, moves_by_cell)
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
        if owner == 0:
            return "#111111"
        return "#3b82f6" if owner == 1 else "#ef4444"

    def _build_meeple_marker(self, owner: int, meeple_pos: int) -> ft.Container:
        color = self._player_color(owner)
        position_align = {
            0: ALIGN_TOP_CENTER,
            1: ALIGN_CENTER_RIGHT,
            2: ALIGN_BOTTOM_CENTER,
            3: ALIGN_CENTER_LEFT,
            4: ALIGN_CENTER,
        }
        padding_map = {
            0: _padding(top=3),
            1: _padding(right=3),
            2: _padding(bottom=3),
            3: _padding(left=3),
            4: _padding(),
        }
        return ft.Container(
            width=CELL_SIZE,
            height=CELL_SIZE,
            alignment=position_align.get(meeple_pos, ALIGN_CENTER),
            padding=padding_map.get(meeple_pos, _padding()),
            content=ft.Container(
                width=MEEPLE_SIZE,
                height=MEEPLE_SIZE,
                bgcolor=color,
                border=_border(1, "#ffffff"),
                border_radius=2,
            ),
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

    async def _show_board(self, zoom: float) -> None:
        await self.board_viewer.reset()
        if zoom != 1.0:
            await self.board_viewer.zoom(zoom)
        if self.board_viewport is not None:
            width, height = self.board_viewport
            center_x, center_y = self._tiles_center()
            # pan() moves in content units: the viewport shows s * (p + d), s = zoom.
            await self.board_viewer.pan(width / 2 / zoom - center_x, height / 2 / zoom - center_y)

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

    def on_cell_click(self, x: int, y: int, moves_by_cell: Dict[Tuple[int, int], List[int]]) -> None:
        if self.awaiting_meeple or not self._human_turn():
            return
        rots = moves_by_cell.get((x, y))
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
            self.meeple_options = self.engine.confirm_tile(self.selected_move)
        except ValueError as exc:
            self.status.value = str(exc)
            self.selected_move = None
            self.refresh()
            return

        if not self.meeple_options:
            self.awaiting_meeple = True
            self.on_apply_move(-1)
            return

        self.awaiting_meeple = True
        self.status.value = "Choose meeple position or skip."
        self.refresh()

    def on_apply_move(self, meeple_pos: int) -> None:
        if not self.awaiting_meeple or self.engine is None:
            return
        old_scores = dict(self.state.scores)
        try:
            self.engine.apply_meeple(meeple_pos)
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
            self.status.value = "Move applied."

        self.selected_move = None
        self.awaiting_meeple = False
        self.meeple_options = []
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
                        self.status.value = "Your turn."
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
            self.awaiting_meeple = False
            self.meeple_options = []
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
