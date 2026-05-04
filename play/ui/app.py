from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import flet as ft

try:
    from domain import Move
    from engine import BOARD_SIZE, CppCarcassonneAdapter
except ImportError:  # pragma: no cover - package import fallback
    from ..domain import Move
    from ..engine import BOARD_SIZE, CppCarcassonneAdapter


IMAGE_FIT = getattr(ft, "ImageFit", ft.BoxFit)
ALIGN_CENTER = ft.alignment.Alignment(0, 0)
ALIGN_TOP_CENTER = ft.alignment.Alignment(0, -1)
ALIGN_CENTER_RIGHT = ft.alignment.Alignment(1, 0)
ALIGN_BOTTOM_CENTER = ft.alignment.Alignment(0, 1)
ALIGN_CENTER_LEFT = ft.alignment.Alignment(-1, 0)


class CarcassonneUI:
    def __init__(self, page: ft.Page):
        self.page = page
        self.engine = CppCarcassonneAdapter()
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
        self.view_text = ft.Text()
        self.holding_image = ft.Image(src="tiles/1.png", width=140, height=140, fit=IMAGE_FIT.CONTAIN)

        self.opponent_dropdown = ft.Dropdown(
            value="player",
            label="Opponent",
            options=[
                ft.DropdownOption(key="player", text="Player"),
                ft.DropdownOption(key="random", text="Random"),
                ft.DropdownOption(key="mcts", text="MCTS"),
                ft.DropdownOption(key="alphazero", text="AlphaZero"),
            ],
            width=180,
            on_select=self.on_opponent_change,
        )
        self.new_game_btn = ft.ElevatedButton("New Game", on_click=self.on_new_game)
        self.confirm_btn = ft.ElevatedButton("Confirm Tile", on_click=self.on_confirm_tile)
        self.skip_btn = ft.OutlinedButton("Skip Meeple", on_click=lambda _: self.on_apply_move(-1))
        self.pan_up_btn = ft.OutlinedButton("Up", on_click=lambda _: self.on_pan(0, -1), width=88)
        self.pan_left_btn = ft.OutlinedButton("Left", on_click=lambda _: self.on_pan(-1, 0), width=88)
        self.pan_right_btn = ft.OutlinedButton("Right", on_click=lambda _: self.on_pan(1, 0), width=88)
        self.pan_down_btn = ft.OutlinedButton("Down", on_click=lambda _: self.on_pan(0, 1), width=88)

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
                    ft.Row([self.opponent_dropdown, self.new_game_btn], wrap=True),
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
                    ft.Divider(),
                    self.status,
                    ft.Divider(),
                    ft.Text("Viewport", size=18, weight=ft.FontWeight.BOLD),
                    self.view_text,
                    ft.Row([ft.Container(width=88), self.pan_up_btn, ft.Container(width=88)]),
                    ft.Row([self.pan_left_btn, self.pan_right_btn], alignment=ft.MainAxisAlignment.CENTER),
                    ft.Row([ft.Container(width=88), self.pan_down_btn, ft.Container(width=88)]),
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
        self.new_game_btn.disabled = False

        for pos, btn in self.meeple_buttons.items():
            btn.visible = self.awaiting_meeple and pos in self.meeple_options
            btn.disabled = ai_turn

        self.turn_text.value = f"Turn: {self.state.turn} | Player: P{self.state.current_player}"
        if self.engine.ai_status:
            self.ai_text.value = f"AI: {self.engine.ai_status}"
        elif self.engine.opponent_mode == "player":
            self.ai_text.value = "Mode: Player vs Player"
        else:
            self.ai_text.value = f"Mode: Player vs {self.engine.opponent_mode}"
        self.holding_text.value = "Holding tile"
        self.view_text.value = (
            f"View X: {self.view_origin[0]}-{self.view_origin[0] + BOARD_SIZE - 1} | "
            f"Y: {self.view_origin[1]}-{self.view_origin[1] + BOARD_SIZE - 1}"
        )
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
        pan_disabled = self.awaiting_meeple or self.state.game_over or ai_turn
        self.pan_up_btn.disabled = pan_disabled or not self.engine.can_pan(0, -1)
        self.pan_left_btn.disabled = pan_disabled or not self.engine.can_pan(-1, 0)
        self.pan_right_btn.disabled = pan_disabled or not self.engine.can_pan(1, 0)
        self.pan_down_btn.disabled = pan_disabled or not self.engine.can_pan(0, 1)

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
        if is_valid:
            bg = "#ecfdf3"
            border_color = "#63b36f"
        if is_selected:
            bg = "#fff3cd"
            border_color = "#d18e00"

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
            border=ft.border.all(1, border_color),
            alignment=ALIGN_CENTER,
            content=content,
            on_click=lambda _: self.on_cell_click(x, y, moves_by_cell),
        )

    def _build_meeple_marker(self, owner: int, meeple_pos: int) -> ft.Container:
        color = "#3b82f6" if owner == 1 else "#ef4444"
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

    def on_pan(self, dx: int, dy: int) -> None:
        if self.awaiting_meeple or self.state.game_over:
            return
        if not self.engine.pan(dx, dy):
            return
        self.selected_move = None
        self.status.value = "Viewport moved."
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
            if self.engine.opponent_mode != "player":
                self.status.value = f"{self.engine.opponent_mode} thinking..."
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

    def on_opponent_change(self, _: ft.ControlEvent) -> None:
        self.status.value = "Opponent mode will apply to the next new game."
        self.refresh()

    def on_new_game(self, _: ft.ControlEvent) -> None:
        mode = self.opponent_dropdown.value or "player"
        try:
            self.engine = CppCarcassonneAdapter(opponent_mode=mode)
        except ValueError as exc:
            self.status.value = str(exc)
            self.refresh()
            return
        self.state = self.engine.state
        self.view_origin = self.engine.view_origin
        self.selected_move = None
        self.awaiting_meeple = False
        self.meeple_options = []
        self.status.value = f"New game: Player vs {mode}."
        self.refresh()


def main(page: ft.Page) -> None:
    CarcassonneUI(page)


def run_app() -> None:
    assets_dir = str(Path(__file__).resolve().parent.parent.parent)
    mode = os.getenv("CARCASSONNE_UI_MODE", "web").lower()
    if mode == "desktop":
        ft.run(main, assets_dir=assets_dir, view=ft.AppView.FLET_APP)
        return

    host = os.getenv("CARCASSONNE_UI_HOST", "127.0.0.1")
    port = int(os.getenv("CARCASSONNE_UI_PORT", os.getenv("FLET_SERVER_PORT", "8550")))
    os.environ.setdefault("FLET_FORCE_WEB_SERVER", "true")
    print(f"Carcassonne UI: http://{host}:{port}")
    ft.run(main, assets_dir=assets_dir, host=host, port=port, view=ft.AppView.WEB_BROWSER)
