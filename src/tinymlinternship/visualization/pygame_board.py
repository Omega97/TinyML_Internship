"""Render chess positions with pygame."""

from __future__ import annotations

import os
import chess
import chess.pgn
import pygame

LIGHT_SQ = (240, 217, 181)
DARK_SQ = (181, 136, 99)
HIGHLIGHT = (186, 202, 68)
CHECK_COLOR = (235, 97, 80)
MARGIN_BG = (45, 45, 45)
TEXT_COLOR = (230, 230, 230)

PIECE_GLYPHS: dict[tuple[bool, int], str] = {
    (chess.WHITE, chess.KING): "♔",
    (chess.WHITE, chess.QUEEN): "♕",
    (chess.WHITE, chess.ROOK): "♖",
    (chess.WHITE, chess.BISHOP): "♗",
    (chess.WHITE, chess.KNIGHT): "♘",
    (chess.WHITE, chess.PAWN): "♙",
    (chess.BLACK, chess.KING): "♚",
    (chess.BLACK, chess.QUEEN): "♛",
    (chess.BLACK, chess.ROOK): "♜",
    (chess.BLACK, chess.BISHOP): "♝",
    (chess.BLACK, chess.KNIGHT): "♞",
    (chess.BLACK, chess.PAWN): "♟",
}


def _pick_piece_font(size: int) -> pygame.font.Font:
    for name in ("segoeuisymbol", "Segoe UI Symbol", "dejavusans", "arial"):
        path = pygame.font.match_font(name)
        if path:
            return pygame.font.Font(path, size)
    return pygame.font.SysFont(None, size)


class PygameBoardRenderer:
    """Draw a ``chess.Board`` and optionally show it in a pygame window."""

    def __init__(
        self,
        *,
        square_size: int = 72,
        header_height: int = 36,
        headless: bool = False,
    ) -> None:
        if headless:
            os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        if not pygame.get_init():
            pygame.init()
            pygame.font.init()

        self.square_size = square_size
        self.header_height = header_height
        self.board_px = square_size * 8
        self.width = self.board_px
        self.height = self.board_px + header_height
        self._piece_font = _pick_piece_font(int(square_size * 0.78))
        self._label_font = _pick_piece_font(18)
        self._screen: pygame.Surface | None = None
        if not headless:
            self._screen = pygame.display.set_mode((self.width, self.height))
            pygame.display.set_caption("SARDINE — engine game")

    def render(
        self,
        board: chess.Board,
        *,
        last_move: chess.Move | None = None,
        caption: str = "",
    ) -> pygame.Surface:
        surface = pygame.Surface((self.width, self.height))
        surface.fill(MARGIN_BG)
        self._draw_header(surface, board, caption)
        self._draw_board(surface, board, last_move=last_move)
        return surface

    def show(
        self,
        board: chess.Board,
        *,
        last_move: chess.Move | None = None,
        caption: str = "",
        delay_ms: int = 350,
        pump_events: bool = True,
    ) -> pygame.Surface:
        frame = self.render(board, last_move=last_move, caption=caption)
        if self._screen is not None:
            self._screen.blit(frame, (0, 0))
            pygame.display.flip()
            if pump_events:
                pygame.event.pump()
            if delay_ms > 0:
                pygame.time.wait(delay_ms)
        return frame

    def frames_from_game(
        self,
        game: chess.pgn.Game,
        *,
        caption_prefix: str = "SARDINE v0.1",
    ) -> list[pygame.Surface]:
        board = game.board()
        frames: list[pygame.Surface] = []
        frames.append(
            self.render(board, caption=f"{caption_prefix} — start")
        )
        last_move: chess.Move | None = None
        for node in game.mainline():
            if node.move is None:
                continue
            last_move = node.move
            board.push(last_move)
            side = "White" if not board.turn else "Black"
            frames.append(
                self.render(
                    board,
                    last_move=last_move,
                    caption=f"{caption_prefix} — {side} played {last_move.uci()}",
                )
            )
        return frames

    def _draw_header(self, surface: pygame.Surface, board: chess.Board, caption: str) -> None:
        if not caption:
            side = "White" if board.turn == chess.WHITE else "Black"
            caption = f"{side} to move"
            if board.is_check():
                caption += " (check)"
        text = self._label_font.render(caption, True, TEXT_COLOR)
        surface.blit(text, (8, 8))

    def _draw_board(
        self,
        surface: pygame.Surface,
        board: chess.Board,
        *,
        last_move: chess.Move | None,
    ) -> None:
        highlight: set[int] = set()
        if last_move is not None:
            highlight.add(last_move.from_square)
            highlight.add(last_move.to_square)

        king_sq = board.king(board.turn)
        in_check = board.is_check() and king_sq is not None

        for rank in range(8):
            for file in range(8):
                square = chess.square(file, 7 - rank)
                x = file * self.square_size
                y = self.header_height + rank * self.square_size
                light = (file + rank) % 2 == 0
                color = LIGHT_SQ if light else DARK_SQ
                if square in highlight:
                    color = HIGHLIGHT
                if in_check and square == king_sq:
                    color = CHECK_COLOR
                pygame.draw.rect(surface, color, (x, y, self.square_size, self.square_size))

                piece = board.piece_at(square)
                if piece is None:
                    continue
                glyph = PIECE_GLYPHS[(piece.color, piece.piece_type)]
                text = self._piece_font.render(glyph, True, (20, 20, 20))
                rect = text.get_rect(center=(x + self.square_size // 2, y + self.square_size // 2))
                surface.blit(text, rect)

    def quit(self) -> None:
        if pygame.get_init():
            pygame.quit()