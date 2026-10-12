"""World-model encoder (self-supervised L1) — losses, flags, and roundtrips."""

from __future__ import annotations

import torch

from tinymlinternship.nnue.model import DualHiddenNNUE, l1_activation
from tinymlinternship.nnue.moe import (
    DualHiddenMoE,
    SoftGatedMoE,
    checkpoint_encoder_info,
    load_dual_hidden_checkpoint,
)
from tinymlinternship.nnue.world_model import (
    collapse_diagnostics,
    info_nce_loss,
    vicreg_loss,
)


def test_l1_activation_standard_vs_world_model():
    torch.manual_seed(0)
    z = torch.randn(8, 16)
    crelu_out = l1_activation(z, encoder="standard", crelu_clip=127.0)
    assert (crelu_out >= 0.0).all() and (crelu_out <= 127.0).all()
    silu = torch.nn.functional.silu(z)
    wm_out = l1_activation(z, encoder="world_model", normalize=False)
    assert torch.allclose(wm_out, silu, atol=1e-6)
    wm_norm = l1_activation(z, encoder="world_model", normalize=True)
    assert torch.allclose(wm_norm.norm(dim=-1), torch.ones(8), atol=1e-5)


def test_dual_hidden_world_model_flag_and_roundtrip(tmp_path):
    model = DualHiddenNNUE(hidden_dim=32, hidden2_dim=64, encoder="world_model", normalize_l1=True)
    assert model.is_world_model
    path = tmp_path / "best.pt"
    model.save(path)
    loaded = load_dual_hidden_checkpoint(path)
    assert loaded.is_world_model
    assert loaded.normalize_l1 is True
    assert loaded.hidden_dim == 32 and loaded.hidden2_dim == 64
    info = checkpoint_encoder_info(path)
    assert info is not None and info["encoder"] == "world_model"
    assert info["hidden_dim"] == 32 and info["hidden2_dim"] == 64


def test_legacy_checkpoint_without_flag_is_standard(tmp_path):
    model = DualHiddenNNUE(hidden_dim=32, hidden2_dim=64)
    path = tmp_path / "legacy.pt"
    torch.save(
        {
            "architecture": "dual_hidden_wdl",
            "hidden_dim": 32,
            "hidden2_dim": 64,
            "model_state_dict": model.state_dict(),
        },
        path,
    )
    info = checkpoint_encoder_info(path)
    assert info is not None and info["encoder"] == "standard"
    loaded = load_dual_hidden_checkpoint(path)
    assert not loaded.is_world_model


def test_info_nce_and_vicreg_reduce_to_scalar():
    torch.manual_seed(0)
    z1 = torch.randn(16, 32)
    z2 = torch.randn(16, 32)
    nce = info_nce_loss(z1, z2, 0.1)
    assert nce.ndim == 0
    vic = vicreg_loss(z1, z2)
    assert vic.ndim == 0 and float(vic) >= 0.0


def test_collapse_diagnostics():
    torch.manual_seed(0)
    z = torch.randn(64, 32)
    diag = collapse_diagnostics(z)
    assert "effective_rank" in diag and "singular_values" in diag
    assert diag["effective_rank"] > 0.0
    # a collapsed (constant) latent has effective rank ~1
    diag_c = collapse_diagnostics(torch.ones(64, 32))
    assert diag_c["effective_rank"] < 1.5


def test_moe_propagates_world_model_encoder():
    base = DualHiddenNNUE(hidden_dim=32, hidden2_dim=64, encoder="world_model", normalize_l1=True)
    moe = DualHiddenMoE.from_base(base, n_experts=4)
    assert moe.encoder == "world_model"
    assert moe.normalize_l1 is True
    sg = SoftGatedMoE(base, n_experts=2, top_k=1)
    assert sg.encoder == "world_model"
    assert sg.normalize_l1 is True
