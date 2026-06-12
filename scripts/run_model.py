#!/usr/bin/env python3
"""
Run a prepared chess model on a sample board state.

This is a standalone inference tool for models produced by the export pipeline
(TinyPolicy or the tiny value nets).

Usage examples:
  # Value net (recommended for Wio)
  python .\scripts\run_model.py --checkpoint models/checkpoints/tiny_value_wio.pt --type value --fen "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"

  # Policy net
  python .\scripts\run_model.py --checkpoint models/checkpoints/tiny_chess_policy_lab.pt  --type policy --fen "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3" --top-k 3

"""
import argparse
import sys
from pathlib import Path
import chess
import torch

# Make the package importable when running the script directly
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.tinymlinternship.datasets.featurizer import fen_to_tensor, get_legal_mask
from src.tinymlinternship.models.policy import TinyPolicy
from src.tinymlinternship.models.value import TinyValueMLP, UltraTinyValueMLP


def load_model(checkpoint: Path, model_type: str) -> torch.nn.Module:
    """Load a model from checkpoint. Supports value and policy variants."""
    checkpoint = checkpoint.resolve()
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    if model_type == "value":
        # Try the smallest first (most common for Wio), then larger
        for ModelClass in (UltraTinyValueMLP, TinyValueMLP):
            try:
                model = ModelClass()
                model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
                model.eval()
                print(f"Loaded {ModelClass.__name__} from {checkpoint.name}")
                return model
            except (RuntimeError, KeyError):
                # Wrong architecture or hidden size
                continue
        raise RuntimeError(
            f"Could not load value model from {checkpoint}. "
            "Check that the checkpoint matches TinyValueMLP or UltraTinyValueMLP."
        )

    elif model_type == "policy":
        model = TinyPolicy()
        model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
        model.eval()
        print(f"Loaded TinyPolicy from {checkpoint.name}")
        return model

    else:
        raise ValueError(f"Unknown model_type: {model_type}")


def run_value_model(model: torch.nn.Module, fen: str) -> float:
    """Run a value net and return the scalar output (in approx [-1, +1])."""
    x = fen_to_tensor(fen, flatten=True).unsqueeze(0)  # (1, 768)
    with torch.no_grad():
        value = model(x).item()
    return value


def run_policy_model(model: torch.nn.Module, fen: str, top_k: int = 1) -> list:
    """
    Run a policy net and return top-k legal moves with their probabilities.
    Returns list of (uci_move, probability).
    """
    board = chess.Board(fen)
    x = fen_to_tensor(fen, flatten=False).unsqueeze(0)  # (1, 12, 8, 8)
    mask = get_legal_mask(fen)  # (4096,)

    with torch.no_grad():
        logits = model(x).squeeze(0)  # (4096,)
        logits = logits.masked_fill(~mask, float("-inf"))
        probs = torch.softmax(logits, dim=-1)

    top_indices = torch.topk(probs, k=min(top_k, mask.sum().item())).indices.tolist()

    results = []
    for idx in top_indices:
        from_sq = idx // 64
        to_sq = idx % 64
        move = chess.Move(from_sq, to_sq)
        # Handle promotion if needed (simple default to queen for now)
        if move not in board.legal_moves:
            for promo in [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]:
                promo_move = chess.Move(from_sq, to_sq, promotion=promo)
                if promo_move in board.legal_moves:
                    move = promo_move
                    break
        prob = probs[idx].item()
        results.append((move.uci(), prob))

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Run a prepared chess model on a board position."
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        type=Path,
        help="Path to the .pt checkpoint (e.g. models/checkpoints/my_tiny_model.pt)",
    )
    parser.add_argument(
        "--fen",
        default=chess.STARTING_FEN,
        help="FEN string of the position to evaluate (default: starting position)",
    )
    parser.add_argument(
        "--type",
        choices=["value", "policy"],
        default="value",
        help="Model type (default: value)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="For policy models: show top-K legal moves (default: 3)",
    )
    args = parser.parse_args()

    model = load_model(args.checkpoint, args.type)

    board = chess.Board(args.fen)
    print(f"\nPosition:\n{board}\nFEN: {args.fen}\n")

    if args.type == "value":
        value = run_value_model(model, args.fen)
        print(f"Model value (positive = good for side to move): {value:+.4f}")
        if value > 0.5:
            print("  → Model thinks the side to move has a clear advantage.")
        elif value < -0.5:
            print("  → Model thinks the side to move is in trouble.")
        else:
            print("  → Model evaluates the position as roughly equal.")

    else:  # policy
        results = run_policy_model(model, args.fen, top_k=args.top_k)
        print(f"Top {len(results)} suggested moves:")
        for i, (uci, prob) in enumerate(results, 1):
            print(f"  {i}. {uci}  (prob={prob:.4f})")

    print()


if __name__ == "__main__":
    main()
