from __future__ import annotations

import argparse
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
ALIGN_CENTER = ft.alignment.Alignment(0, 0)
ALIGN_TOP_CENTER = ft.alignment.Alignment(0, -1)
ALIGN_CENTER_RIGHT = ft.alignment.Alignment(1, 0)
ALIGN_BOTTOM_CENTER = ft.alignment.Alignment(0, 1)
ALIGN_CENTER_LEFT = ft.alignment.Alignment(-1, 0)


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
    base = f"P{record.player}({record.x},{record.y},{record.rotation},{record.meeple_pos})"
    nonzero_deltas = {player: delta for player, delta in sorted(record.score_deltas.items()) if delta}
    if len(nonzero_deltas) <= 1:
        points = next(iter(nonzero_deltas.values()), 0)
        return f"{base} {points:+d}(得分)"

    deltas = "/".join(f"P{player}{delta:+d}" for player, delta in nonzero_deltas.items())
    return f"{base} {deltas}(得分)"


def is_bot_vs_bot(player_specs: Tuple[PlayerSpec, PlayerSpec]) -> bool:
    return all(spec.is_bot for spec in player_specs)


def should_show_start_game(player_specs: Tuple[PlayerSpec, PlayerSpec], started: bool) -> bool:
    return is_bot_vs_bot(player_specs) and not started


class CarcassonneUI:
    def __init__(self, page: ft.Page, config: Optional[PlayUiConfig] = None):
        self.page = page
        self.config = config or PlayUiConfig()
        self.bot_game_started = False
        self.engine = self._new_engine()
        self.state = self.engine.state
        self.view_origin = self.engine.view_origin

        self.selected_move: Optional[Move] = None
        self.awaiting_meeple = False
        self.meeple_options: List[int] = []

        self.status = ft.Text("Ready.")
        self.ai_text = ft.Text("")
        self.turn_text = ft.Text()
        self.holding_text = ft.Text("Holding tile")
        self.score_text = ft.Text()
        self.meeple_text = ft.Text()
        self.holding_image = ft.Image(src="tiles/1.png", width=140, height=140, fit=IMAGE_FIT.CONTAIN)
        self.records_column = ft.Column(spacing=4, scroll=ft.ScrollMode.AUTO)

        self.start_game_btn = ft.ElevatedButton("Start Game", on_click=self.on_start_game)
        self.confirm_btn = ft.ElevatedButton("Confirm Tile", on_click=self.on_confirm_tile)
        self.skip_btn = ft.OutlinedButton("Skip Meeple", on_click=lambda _: self.on_apply_move(-1))

        self.meeple_buttons = {
            0: ft.ElevatedButton("Meeple: Up", on_click=lambda _: self.on_apply_move(0)),
            1: ft.ElevatedButton("Meeple: Right", on_click=lambda _: self.on_apply_move(1)),
            2: ft.ElevatedButton("Meeple: Down", on_click=lambda _: self.on_apply_move(2)),
            3: ft.ElevatedButton("Meeple: Left", on_click=lambda _: self.on_apply_move(3)),
            4: ft.ElevatedButton("Meeple: Center", on_click=lambda _: self.on_apply_move(4)),
        }

        self.grid_column = ft.Column(spacing=1, scroll=ft.ScrollMode.ALWAYS, expand=True)

        self.board_panel = ft.Container(
            expand=True,
            padding=ft.padding.only(right=12),
            content=ft.Column(
                controls=[
                    ft.Text("Carcassonne (Flet)", size=24, weight=ft.FontWeight.BOLD),
                    self.grid_column,
                ],
                spacing=8,
                expand=True,
            ),
        )

        self.info_panel = ft.Container(
            width=320,
            padding=ft.padding.all(8),
            border=ft.border.all(1, "#d0d7de"),
            border_radius=8,
            content=ft.Column(
                controls=[
                    ft.Text("Game Info", size=20, weight=ft.FontWeight.BOLD),
                    ft.Row([self.start_game_btn], wrap=True),
                    self.turn_text,
                    self.ai_text,
                    self.holding_text,
                    ft.Text("Holding Preview", weight=ft.FontWeight.W_600),
                    ft.Container(
                        width=160,
                        height=160,
                        border=ft.border.all(1, "#cccccc"),
                        alignment=ALIGN_CENTER,
                        content=self.holding_image,
                    ),
                    self.score_text,
                    self.meeple_text,
                    ft.Row([self.confirm_btn, self.skip_btn], wrap=True),
                    ft.Row(list(self.meeple_buttons.values()), wrap=True),
                    ft.Text("Record", size=18, weight=ft.FontWeight.BOLD),
                    ft.Container(
                        height=220,
                        border=ft.border.all(1, "#d0d7de"),
                        border_radius=4,
                        padding=ft.padding.all(6),
                        content=self.records_column,
                    ),
                ],
                spacing=8,
                scroll=ft.ScrollMode.AUTO,
            ),
        )

        self.root = ft.Row(
            controls=[self.board_panel, self.info_panel],
            expand=True,
            vertical_alignment=ft.CrossAxisAlignment.START,
        )

        self.page.title = "Carcassonne"
        self.page.padding = 10
        self.page.scroll = ft.ScrollMode.AUTO
        self.page.add(self.root)
        self.refresh()

    def _new_engine(self) -> CppCarcassonneAdapter:
        return CppCarcassonneAdapter(
            seed=self.config.seed,
            player_specs=self.config.player_specs,
            auto_run_bots=not is_bot_vs_bot(self.config.player_specs),
        )

    def refresh(self) -> None:
        valid_moves = self.engine.get_valid_moves()
        self.state = self.engine.state
        self.view_origin = self.engine.view_origin

        moves_by_cell: Dict[Tuple[int, int], List[int]] = {}
        for move in valid_moves:
            moves_by_cell.setdefault((move.x, move.y), []).append(move.rotation)
        for pos in moves_by_cell:
            moves_by_cell[pos].sort()

        ai_turn = self.engine.is_ai_turn()
        self.confirm_btn.disabled = self.selected_move is None or self.awaiting_meeple or self.state.game_over or ai_turn
        self.skip_btn.visible = self.awaiting_meeple and not ai_turn
        self.start_game_btn.visible = should_show_start_game(self.config.player_specs, self.bot_game_started)
        self.start_game_btn.disabled = self.state.game_over

        for pos, btn in self.meeple_buttons.items():
            btn.visible = self.awaiting_meeple and pos in self.meeple_options
            btn.disabled = ai_turn

        self.turn_text.value = f"Turn: {self.state.turn} | Player: P{self.state.current_player}"
        self.ai_text.value = f"Mode: {self.engine.mode_label()}"
        if self.engine.ai_status:
            self.ai_text.value += f"\nAI: {self.engine.ai_status}"
        self.holding_text.value = "Holding tile"
        if self.state.holding_tile_id is None:
            self.holding_image.visible = False
        else:
            self.holding_image.src = f"tiles/{self.state.holding_tile_id}.png"
            self.holding_image.visible = True
        self.holding_image.rotate = ft.Rotate(angle=0)
        self.score_text.value = f"Scores -> P1: {self.state.scores[1]} | P2: {self.state.scores[2]}"
        self.meeple_text.value = (
            f"Meeples -> P1: {self.state.meeples_remaining[1]} | P2: {self.state.meeples_remaining[2]}"
        )
        record_controls: List[ft.Control] = [
            ft.Text(format_move_record(record), selectable=True) for record in self.engine.move_records
        ]
        self.records_column.controls = record_controls or [ft.Text("No records.")]

        self.grid_column.controls = [self._build_row(y, moves_by_cell) for y in range(BOARD_SIZE)]

        if self.state.game_over:
            winner = "Draw"
            if self.state.scores[1] > self.state.scores[2]:
                winner = "P1 wins"
            elif self.state.scores[2] > self.state.scores[1]:
                winner = "P2 wins"
            self.status.value = f"Game over. {winner}."
            self.awaiting_meeple = False

        self.page.update()

    def _build_row(self, y: int, moves_by_cell: Dict[Tuple[int, int], List[int]]) -> ft.Row:
        row_controls = [self._build_cell(x, y, moves_by_cell) for x in range(BOARD_SIZE)]
        return ft.Row(row_controls, spacing=1)

    def _build_cell(self, x: int, y: int, moves_by_cell: Dict[Tuple[int, int], List[int]]) -> ft.Container:
        pos = (x, y)
        engine_pos = self.engine.to_engine_coords(x, y)
        tile = self.state.board.get(engine_pos)
        is_valid = pos in moves_by_cell
        is_selected = self.selected_move is not None and (self.selected_move.x, self.selected_move.y) == pos

        bg = "#ffffff"
        border_color = "#cccccc"
        border_width = 1
        if is_valid:
            bg = "#ecfdf3"
            border_color = "#63b36f"
        if is_selected:
            bg = "#fff3cd"
            border_color = "#d18e00"
        if tile is not None and tile.tile_owner is not None:
            border_color = self._player_color(tile.tile_owner)
            border_width = 3

        content: ft.Control
        if tile is not None:
            overlay_controls: List[ft.Control] = [
                ft.Image(
                    src=f"tiles/{tile.tile_id}.png",
                    width=34,
                    height=34,
                    fit=IMAGE_FIT.COVER,
                    rotate=ft.Rotate(angle=tile.rotation * 1.57079632679),
                )
            ]
            if tile.meeple_markers:
                for owner, pos_marker in tile.meeple_markers:
                    overlay_controls.append(self._build_meeple_marker(owner, pos_marker))
            elif tile.meeple_owner is not None and tile.meeple_pos is not None:
                overlay_controls.append(self._build_meeple_marker(tile.meeple_owner, tile.meeple_pos))
            content = ft.Stack(controls=overlay_controls)
        elif is_selected and self.state.holding_tile_id is not None and self.selected_move is not None:
            content = ft.Container(
                opacity=0.55,
                content=ft.Image(
                    src=f"tiles/{self.state.holding_tile_id}.png",
                    width=34,
                    height=34,
                    fit=IMAGE_FIT.COVER,
                    rotate=ft.Rotate(angle=self.selected_move.rotation * 1.57079632679),
                ),
            )
        else:
            content = ft.Text("")

        return ft.Container(
            width=36,
            height=36,
            bgcolor=bg,
            border=ft.border.all(border_width, border_color),
            alignment=ALIGN_CENTER,
            content=content,
            on_click=lambda _: self.on_cell_click(x, y, moves_by_cell),
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
            0: ft.padding.only(top=2),
            1: ft.padding.only(right=2),
            2: ft.padding.only(bottom=2),
            3: ft.padding.only(left=2),
            4: ft.padding.all(0),
        }
        return ft.Container(
            alignment=position_align.get(meeple_pos, ALIGN_CENTER),
            padding=padding_map.get(meeple_pos, ft.padding.all(0)),
            content=ft.Container(
                width=9,
                height=9,
                bgcolor=color,
                border=ft.border.all(1, "#ffffff"),
                border_radius=2,
            ),
        )

    def on_cell_click(self, x: int, y: int, moves_by_cell: Dict[Tuple[int, int], List[int]]) -> None:
        if self.awaiting_meeple or self.state.game_over or self.engine.is_ai_turn():
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
        if self.selected_move is None or self.state.game_over:
            return
        try:
            self.meeple_options = self.engine.confirm_tile(self.selected_move)
        except ValueError as exc:
            self.status.value = str(exc)
            self.selected_move = None
            self.refresh()
            return

        if not self.meeple_options:
            self.on_apply_move(-1)
            return

        self.awaiting_meeple = True
        self.status.value = "Choose meeple position or skip."
        self.refresh()

    def on_apply_move(self, meeple_pos: int) -> None:
        if not self.awaiting_meeple and meeple_pos != -1:
            return
        old_scores = dict(self.state.scores)
        try:
            if self.engine.has_bot_players():
                self.status.value = "Bot thinking..."
                self.page.update()
            self.engine.apply_meeple(meeple_pos)
        except ValueError as exc:
            self.status.value = str(exc)
            self.refresh()
            return

        self.state = self.engine.state
        p1_gain = self.state.scores[1] - old_scores[1]
        p2_gain = self.state.scores[2] - old_scores[2]
        if self.engine.ai_status:
            self.status.value = self.engine.ai_status
        elif p1_gain or p2_gain:
            self.status.value = f"Scored: P1 +{p1_gain}, P2 +{p2_gain}"
        else:
            self.status.value = "Move applied."

        self.selected_move = None
        self.awaiting_meeple = False
        self.meeple_options = []
        self.refresh()

    def on_start_game(self, _: ft.ControlEvent) -> None:
        if not is_bot_vs_bot(self.config.player_specs) or self.bot_game_started:
            return
        self.bot_game_started = True
        self.status.value = "Bot thinking..."
        try:
            self.engine.run_ai_turns()
        except ValueError as exc:
            self.status.value = str(exc)
            self.refresh()
            return
        self.state = self.engine.state
        self.view_origin = self.engine.view_origin
        self.selected_move = None
        self.awaiting_meeple = False
        self.meeple_options = []
        self.refresh()


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
