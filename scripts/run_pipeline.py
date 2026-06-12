#!/usr/bin/env python3
"""
Central entry point for the TinyML Chess export / training / inference pipeline.

This is the single place to configure and run experiments.

Current capabilities:
- Run inference on a prepared model (value or policy) on a board position.

Future (planned):
- create: build a model with given architecture
- export: run TorchScript / ONNX / TFLite / C-header export
- train: run training (with possible pruning)
- full: create + train + export + (optional) run

Example usage:
    # Just run the current tiny value model on a position
    py -3.12 scripts/run_pipeline.py \
        --model-name my_tiny_model \
        --model-type value \
        --fen "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1" \
        --stage run

    # Future example with pruning
    # py -3.12 scripts/run_pipeline.py --model-name my_model --prune-method magnitude --stage full

You can also import from this module in notebooks or other scripts.
"""

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import chess

# Make local imports work when running the script directly
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

# Import inference helpers from the sibling module (now possible after the path insert above)
from run_model import load_model, run_value_model, run_policy_model


@dataclass
class PipelineConfig:
    """Central configuration for a full experiment.

    Add new fields here as we introduce new techniques (pruning, quantization, distillation, etc.).
    """
    model_name: str = "my_tiny_model"
    model_type: str = "value"          # "value" or "policy"
    checkpoint: Optional[Path] = None  # explicit path, otherwise derived from model_name

    # Inference options
    fen: str = chess.STARTING_FEN
    top_k: int = 3                     # for policy models

    # Future parameters (currently ignored but ready for use)
    target: str = "wio"                # "wio" | "xiao" | "general"
    quantize: str = "none"             # "none" | "dynamic" | "int8"
    prune_method: str = "none"         # "none" | "magnitude" | "structured" | ...
    prune_amount: float = 0.0

    # Training / data options (future)
    train: bool = False
    max_games: int = 500

    # Misc
    verbose: bool = True

    def get_checkpoint_path(self) -> Path:
        if self.checkpoint:
            return Path(self.checkpoint).resolve()
        return (Path("models/checkpoints") / f"{self.model_name}.pt").resolve()

    def get_exported_ts_path(self) -> Path:
        return (Path("models/exported") / f"{self.model_name}.ts.pt").resolve()


def run_inference(cfg: PipelineConfig) -> None:
    """Run the model on a board position. This is the only stage implemented today."""
    ckpt = cfg.get_checkpoint_path()
    model = load_model(ckpt, cfg.model_type)

    board = chess.Board(cfg.fen)
    print(f"\n=== Running inference ===")
    print(f"Model     : {cfg.model_name} ({cfg.model_type})")
    print(f"Checkpoint: {ckpt}")
    print(f"FEN       : {cfg.fen}")
    print(f"Board:\n{board}\n")

    if cfg.model_type == "value":
        value = run_value_model(model, cfg.fen)
        print(f"Value output: {value:+.4f}")
    else:
        results = run_policy_model(model, cfg.fen, top_k=cfg.top_k)
        print("Top policy suggestions:")
        for i, (uci, prob) in enumerate(results, 1):
            print(f"  {i}. {uci}   prob={prob:.4f}")


def main():
    parser = argparse.ArgumentParser(
        description="Central TinyML Chess pipeline runner."
    )

    # Core identification
    parser.add_argument("--model-name", default="my_tiny_model",
                        help="Base name for the model (used for checkpoint, exported files, etc.)")
    parser.add_argument("--model-type", choices=["value", "policy"], default="value",
                        help="Which family of model to use")

    # What to do
    parser.add_argument("--stage", choices=["run"], default="run",
                        help="Pipeline stage to execute (more stages will be added)")

    # Inference options (used by 'run' stage)
    parser.add_argument("--fen", default=chess.STARTING_FEN,
                        help="FEN of the position to evaluate")
    parser.add_argument("--top-k", type=int, default=3,
                        help="Number of top moves to show (policy models only)")

    # Future extensibility (ignored for now, but declared)
    parser.add_argument("--target", choices=["wio", "xiao", "general"], default="wio",
                        help="Target hardware (affects which architecture/export is used)")
    parser.add_argument("--quantize", choices=["none", "dynamic", "int8"], default="none")
    parser.add_argument("--prune-method", default="none",
                        help="Pruning technique (future)")
    parser.add_argument("--prune-amount", type=float, default=0.0)

    parser.add_argument("--checkpoint", type=Path, default=None,
                        help="Explicit path to checkpoint (overrides --model-name)")

    parser.add_argument("--train", action="store_true",
                        help="Run training as part of the pipeline (future)")

    parser.add_argument("--verbose", action="store_true", default=True)

    args = parser.parse_args()

    cfg = PipelineConfig(
        model_name=args.model_name,
        model_type=args.model_type,
        fen=args.fen,
        top_k=args.top_k,
        target=args.target,
        quantize=args.quantize,
        prune_method=args.prune_method,
        prune_amount=args.prune_amount,
        checkpoint=args.checkpoint,
        train=args.train,
        verbose=args.verbose,
    )

    if args.stage == "run":
        run_inference(cfg)
    else:
        print(f"Stage '{args.stage}' is not implemented yet.")
        print("Currently supported: run")

    # In the future you can do:
    # if args.stage in ("create", "all"):
    #     create_model(cfg)
    # if args.stage in ("export", "all"):
    #     export_model(cfg)
    # if args.stage in ("run", "all"):
    #     run_inference(cfg)


if __name__ == "__main__":
    main()
