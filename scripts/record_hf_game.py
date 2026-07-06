#!/usr/bin/env python3
"""Self-play with a Hugging Face value teacher and export GIF."""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import chess
import chess.pgn
import torch
import torch.nn as nn
import torch.nn.functional as F

from tinymlinternship.config.settings import ARTORIA_SMALL_CKPT, ARTORIA_SMALL_CONFIG, CHESS_LITE_WEIGHTS
from tinymlinternship.engine import ENGINE_VERSION, search
from tinymlinternship.engine.eval_chess_lite import ChessLiteEvaluator
from tinymlinternship.visualization import PygameBoardRenderer, export_game_gif


class LiveMovePrinter:
    def print_move(self, board: chess.Board, move: chess.Move) -> None:
        san = board.san(move)
        if board.turn == chess.WHITE:
            print(f"{board.fullmove_number}. {san}", end="", flush=True)
        else:
            print(f" {san}", flush=True)
        board.push(move)

    def finish(self, board: chess.Board) -> None:
        if board.turn == chess.BLACK:
            print(flush=True)


@dataclass
class HfTeacher:
    name: str
    evaluate_cp: object


@dataclass
class ChessModelConfig:
    vocab_size: int = 256
    d_model: int = 256
    n_layers: int = 8
    n_heads: int = 8
    max_seq_len: int = 79
    num_classes: int = 128
    dropout: float = 0.0
    epsilon: float = 1e-5


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weight * (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps))


class SwiGLU(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        hidden_dim = int(2 * (4 * d_model) / 3)
        self.w1 = nn.Linear(d_model, hidden_dim, bias=False)
        self.w2 = nn.Linear(d_model, hidden_dim, bias=False)
        self.w3 = nn.Linear(hidden_dim, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


class TransformerBlock(nn.Module):
    def __init__(self, config: ChessModelConfig) -> None:
        super().__init__()
        self.attention_norm = RMSNorm(config.d_model, config.epsilon)
        self.attention = nn.MultiheadAttention(
            embed_dim=config.d_model,
            num_heads=config.n_heads,
            dropout=config.dropout,
            batch_first=True,
        )
        self.ffn_norm = RMSNorm(config.d_model, config.epsilon)
        self.ffn = SwiGLU(config.d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.attention_norm(x)
        attn_out, _ = self.attention(h, h, h, need_weights=False)
        x = x + attn_out
        return x + self.ffn(self.ffn_norm(x))


class GrandmasterChessModel(nn.Module):
    def __init__(self, config: ChessModelConfig) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(config.max_seq_len, config.d_model)
        self.layers = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        self.final_norm = RMSNorm(config.d_model, config.epsilon)
        self.output_head = nn.Linear(config.d_model, config.num_classes, bias=False)
        self.value_head = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.ReLU(),
            nn.Linear(config.d_model, 1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        position_ids = torch.arange(x.size(1), device=x.device).unsqueeze(0)
        x = self.token_embedding(x) + self.position_embedding(position_ids)
        for layer in self.layers:
            x = layer(x)
        pooled = self.final_norm(x).mean(dim=1)
        return self.output_head(pooled), self.value_head(pooled)


class ChessTokenizer:
    def __init__(self) -> None:
        valid_moves: list[str] = []
        for from_sq in range(64):
            for to_sq in range(64):
                if from_sq == to_sq:
                    continue
                uci = chess.Move(from_sq, to_sq).uci()
                valid_moves.append(uci)
                if to_sq >= 56 and 48 <= from_sq <= 55:
                    for promo in "qrnb":
                        valid_moves.append(uci + promo)
                if to_sq <= 7 and 8 <= from_sq <= 15:
                    for promo in "qrnb":
                        valid_moves.append(uci + promo)
        valid_moves = sorted(set(valid_moves))
        self.num_actions = len(valid_moves)

    def tokenize(self, fen: str) -> torch.Tensor:
        parts = fen.split(" ")
        board_rows = parts[0].split("/")
        expanded = ""
        for row in board_rows:
            for char in row:
                expanded += "." * int(char) if char.isdigit() else char
        castling = parts[2].ljust(4, ".") if parts[2] != "-" else "...."
        ep = parts[3] if parts[3] != "-" else "-."
        token_str = (
            expanded
            + parts[1]
            + castling
            + ep
            + parts[4].rjust(2, ".")
            + parts[5].rjust(3, ".")
        )
        tokens = [ord(c) for c in token_str]
        if len(tokens) < 79:
            tokens += [46] * (79 - len(tokens))
        else:
            tokens = tokens[:79]
        return torch.tensor(tokens, dtype=torch.long)


class ArtoriaEvaluator:
    def __init__(self, checkpoint: Path, config_path: Path) -> None:
        with config_path.open(encoding="utf-8") as handle:
            config = ChessModelConfig(**json.load(handle))
        self._tokenizer = ChessTokenizer()
        config.num_classes = self._tokenizer.num_actions
        self._model = GrandmasterChessModel(config)
        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
        self._model.load_state_dict(ckpt["model_state_dict"])
        self._model.eval()

    def evaluate_cp(self, board: chess.Board) -> int:
        with torch.no_grad():
            tokens = self._tokenizer.tokenize(board.fen()).unsqueeze(0)
            _, value = self._model(tokens)
        reward = float(value.item())
        white_reward = reward if board.turn == chess.WHITE else -reward
        return int(round(white_reward * 1000))


def bench_eval_ms(evaluate_cp: object, *, rounds: int = 30) -> float:
    board = chess.Board()
    t0 = time.perf_counter()
    for _ in range(rounds):
        evaluate_cp(board)
    return (time.perf_counter() - t0) / rounds * 1000.0


def pick_teacher(requested: str) -> tuple[HfTeacher, float]:
    chess_lite = ChessLiteEvaluator()
    artoria = ArtoriaEvaluator(ARTORIA_SMALL_CKPT, ARTORIA_SMALL_CONFIG)
    lite_ms = bench_eval_ms(chess_lite.evaluate_cp)
    artoria_ms = bench_eval_ms(artoria.evaluate_cp)
    print(f"Bench eval (startpos): chess_lite {lite_ms:.1f} ms · artoria-small {artoria_ms:.1f} ms")

    if requested == "auto":
        if lite_ms <= artoria_ms:
            return HfTeacher("chess_lite", chess_lite.evaluate_cp), lite_ms
        return HfTeacher("artoria-small", artoria.evaluate_cp), artoria_ms
    if requested == "chess_lite":
        return HfTeacher("chess_lite", chess_lite.evaluate_cp), lite_ms
    return HfTeacher("artoria-small", artoria.evaluate_cp), artoria_ms


def play_hf_game(
    teacher: HfTeacher,
    *,
    max_plies: int,
    depth: int,
    live: LiveMovePrinter,
) -> chess.pgn.Game:
    def eval_fn(board: chess.Board) -> int:
        return teacher.evaluate_cp(board)

    board = chess.Board()
    game = chess.pgn.Game()
    game.headers["Event"] = "SARDINE HF teacher self-play"
    game.headers["White"] = teacher.name
    game.headers["Black"] = teacher.name
    game.headers["Annotator"] = f"SARDINE {ENGINE_VERSION} ({teacher.name}, {depth}-ply)"

    node = game
    plies = 0
    while not board.is_game_over() and plies < max_plies:
        result = search(board, depth, eval_fn=eval_fn, quiescence=True)
        if result is None:
            break
        live.print_move(board, result.move)
        node = node.add_variation(result.move)
        plies += 1

    live.finish(board)
    outcome = board.outcome()
    game.headers["Result"] = outcome.result() if outcome else "*"
    return game


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Record HF teacher self-play GIF")
    parser.add_argument(
        "--model",
        choices=("auto", "chess_lite", "artoria"),
        default="auto",
        help="HF teacher (default: auto = faster of chess_lite vs artoria-small)",
    )
    parser.add_argument("--depth", type=int, default=1)
    parser.add_argument("--max-plies", type=int, default=48)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--frame-ms", type=int, default=500)
    parser.add_argument("--exporter", choices=("gifpgn", "chess_gif", "pygame"), default="gifpgn")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)

    if not CHESS_LITE_WEIGHTS.exists() or not ARTORIA_SMALL_CKPT.exists():
        print("Missing weights — run: py -3.12 scripts/download_hf_teacher.py")
        return 1

    teacher, eval_ms = pick_teacher(args.model)
    print(f"Using {teacher.name} (~{eval_ms:.1f} ms/eval)")

    if args.output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"hf_{teacher.name}_depth{args.depth}_{timestamp}.gif"
        args.output = Path(__file__).parent.parent / "images" / "games" / filename

    live = LiveMovePrinter()
    print(f"Playing {teacher.name} {args.depth}-ply self-play (max {args.max_plies} plies) …")
    print()
    game = play_hf_game(teacher, max_plies=args.max_plies, depth=args.depth, live=live)

    print()
    moves = max(0, game.end().ply() - game.ply())
    print(f"Game finished: {moves} plies, result {game.headers.get('Result', '*')}")

    renderer = PygameBoardRenderer(headless=args.headless)
    pygame_frames: list = []
    try:
        board = game.board()
        pygame_frames.append(renderer.show(board, caption="Start", delay_ms=200))
        last_move = None
        for node in game.mainline():
            if node.move is None:
                continue
            last_move = node.move
            board.push(last_move)
            pygame_frames.append(
                renderer.show(
                    board,
                    last_move=last_move,
                    caption=(
                        f"{board.fullmove_number}{'.' if board.turn == chess.BLACK else '...'} "
                        f"{last_move.uci()}"
                    ),
                    delay_ms=0,
                )
            )
    finally:
        renderer.quit()

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    export_game_gif(
        game,
        output,
        exporter=args.exporter,
        frame_duration=args.frame_ms / 1000.0,
        board_size=480,
        pygame_frames=pygame_frames if args.exporter == "pygame" else None,
    )
    print(f"Saved GIF: {output} ({output.stat().st_size:,} bytes)")

    pgn_path = output.with_suffix(".pgn")
    with pgn_path.open("w", encoding="utf-8") as handle:
        print(game, file=handle)
    print(f"Saved PGN: {pgn_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())