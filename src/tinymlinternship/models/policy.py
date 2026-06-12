"""
scripts/models.py
Download and initialize policy networks for TinyML chess project.
- Creates a random TinyPolicy (matching our featurizer) and saves to models/checkpoints/
- Clones Policy-chess from GitHub and saves architecture stub to models/exported/
"""
import torch
import torch.nn as nn
import subprocess
from pathlib import Path
import logging
from tinymlinternship.config.settings import CHECKPOINTS_DIR, EXPORTED_DIR


# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


POLICY_CHESS_URL = "https://github.com/Zeta36/Policy-chess"
POLICY_CHESS_DIR = EXPORTED_DIR / "policy-chess"


class TinyPolicy(nn.Module):
    """
    Minimal CNN policy network for TinyML deployment.
    Input:  (B, 12, 8, 8)  → 12 channels: 6 piece types × 2 colors
    Output: (B, 4096)      → logits over from_sq*64 + to_sq moves
    TFLite-compatible: static shapes, no dynamic ops.
    """

    def __init__(self, hidden_channels: int = 16, policy_size: int = 4096):
        super().__init__()
        self.conv = nn.Conv2d(12, hidden_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(hidden_channels * 8 * 8, policy_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.conv(x))  # (B, 16, 8, 8)
        return self.fc(x.view(x.size(0), -1))  # (B, 4096)


def create_and_save_random_policy(model_name: str = "tiny_policy_v0.1") -> Path:
    """
    Initialize TinyPolicy with random weights and save state_dict.
    Returns path to saved .pt file.
    """
    model = TinyPolicy().eval()

    # Save state_dict only (portable, version-safe)
    save_path = CHECKPOINTS_DIR / f"{model_name}.pt"
    torch.save(model.state_dict(), save_path)

    # Log model stats
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    logger.info(f"✅ Saved random TinyPolicy to: {save_path}")
    logger.info(f"   Total params: {total_params:,} | Trainable: {trainable_params:,}")
    logger.info(f"   Input shape: (B, 12, 8, 8) → Output: (B, 4096)")

    return save_path


def download_policy_chess() -> Path:
    """
    Clone Policy-chess repository to models/exported/policy-chess/.
    Returns path to cloned repo.
    """
    if POLICY_CHESS_DIR.exists():
        logger.info(f"🔄 Policy-chess already exists at: {POLICY_CHESS_DIR}")
        return POLICY_CHESS_DIR

    logger.info(f"📥 Cloning Policy-chess from {POLICY_CHESS_URL}...")

    try:
        subprocess.run(
            ["git", "clone", "--depth", "1", POLICY_CHESS_URL, str(POLICY_CHESS_DIR)],
            check=True,
            capture_output=True,
            timeout=120
        )
        logger.info(f"✅ Cloned Policy-chess to: {POLICY_CHESS_DIR}")
        return POLICY_CHESS_DIR
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Failed to clone Policy-chess: {e.stderr.decode()}")
        raise
    except FileNotFoundError:
        logger.warning("⚠️ 'git' not found. Please install Git or manually clone the repo.")
        # Fallback: create placeholder
        POLICY_CHESS_DIR.mkdir(parents=True, exist_ok=True)
        (POLICY_CHESS_DIR / "README.txt").write_text(
            "Policy-chess repo placeholder.\n"
            "Manually clone from: https://github.com/Zeta36/Policy-chess\n"
            "Architecture: 8x8x8 board input → Conv layers → 4096-policy softmax"
        )
        return POLICY_CHESS_DIR


def create_policy_chess_stub() -> Path:
    """
    Create a PyTorch architecture stub matching Policy-chess (8x8x8 input).
    Useful for testing export pipeline before loading actual weights.
    Returns path to saved stub .pt file.
    """

    class PolicyChessStub(nn.Module):
        """
        Approximate Policy-chess architecture.
        Input:  (B, 8, 8, 8)  → 8 channels: piece types (no color separation)
        Output: (B, 4096)     → logits over legal moves
        Note: Actual Policy-chess may differ; inspect cloned repo for exact layers.
        """

        def __init__(self):
            super().__init__()
            # Typical small CNN for chess policy
            self.conv1 = nn.Conv2d(8, 32, kernel_size=3, padding=1)
            self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
            self.relu = nn.ReLU()
            self.fc = nn.Linear(64 * 8 * 8, 4096)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # x: (B, 8, 8, 8) → need to permute to (B, 8, 8, 8) for Conv2d
            # Assuming input is already (B, C, H, W) = (B, 8, 8, 8)
            x = self.relu(self.conv1(x))
            x = self.relu(self.conv2(x))
            return self.fc(x.view(x.size(0), -1))

    stub = PolicyChessStub().eval()
    save_path = EXPORTED_DIR / "policy_chess_stub.pt"
    torch.save(stub.state_dict(), save_path)

    params = sum(p.numel() for p in stub.parameters())
    logger.info(f"✅ Saved Policy-chess architecture stub to: {save_path}")
    logger.info(f"   Params: {params:,} | Input: (B, 8, 8, 8) → Output: (B, 4096)")

    return save_path


def main():
    """Run all model initialization tasks."""
    logger.info("🚀 Initializing models for TinyML chess project...")

    # Ensure directories exist
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    EXPORTED_DIR.mkdir(parents=True, exist_ok=True)

    # Task 1: Random TinyPolicy
    create_and_save_random_policy()

    # Task 2: Policy-chess download + stub
    download_policy_chess()
    create_policy_chess_stub()

    logger.info("✨ All model initialization tasks completed.")
    logger.info(f"📁 Checkpoints: {CHECKPOINTS_DIR}")
    logger.info(f"📁 Exports: {EXPORTED_DIR}")


if __name__ == "__main__":
    main()
