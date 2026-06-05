"""Day 6: Teacher Assistant — TSM MobileNetV2 (uni-directional) with per-frame head.

Architecture (Plan Section 1 + Day 6):
  * Backbone: torchvision MobileNetV2 (3.5M params), ImageNet pretrained
  * Temporal modeling: TSM uni-directional shift inserted on conv[0] of every
    InvertedResidual whose `use_res_connect=True` (10 sites, matching MIT TSM
    convention `place='blockres'`)
  * Shift kernel: causal — only `t-1 -> t` direction; first `C // n_div`
    channels carry the cache, remaining channels stay put. The "left shift"
    branch of MIT bi-directional TSM is removed (acausal -> incompatible with
    streaming deployment).
  * Head: per-frame `Linear(1280, 27)` shared across T time steps;
    NO temporal averaging — output is `[B, T, n_classes]`, consumed by the
    weighted per-frame KD loss in train_ta.py.

Notes:
  * BatchNorm2d statistics across N*T frames work because frames are merged
    into the batch dim (B*T) at the backbone input. Sync between DDP ranks
    via `nn.SyncBatchNorm.convert_sync_batchnorm` (called in training script).
  * Pretrained weights are loaded from torchvision URL on cold start; cached
    at $TORCH_HOME (project Lustre dir) so compute nodes (no internet) can
    reload from cache.

Causal-shift formulation:
  Standard MIT TSM shifts 2/n_div of channels (1/n_div left + 1/n_div right).
  We collapse the left branch and keep the same 1/n_div = C/8 right-shift
  channels per default. This matches the "online TSM" formulation: cache
  buffer size per layer = C/8, which is also the per-cache memory footprint
  the U5 streaming inference loop needs to maintain.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torchvision.models import MobileNet_V2_Weights, mobilenet_v2
from torchvision.models.mobilenetv2 import InvertedResidual


# ---- Uni-directional (causal) temporal shift -------------------------------

class CausalTemporalShift(nn.Module):
    """Wraps an inner module; performs causal channel shift on input first.

    Expects input shape `(B*T, C, H, W)` with frames merged into batch dim.
    `n_segment` = T. Reshapes to `(B, T, C, H, W)`, performs the shift in the
    temporal dim, then flattens back and forwards through `net`.

    Causal semantics:
      out[:, 0,  :fold] = 0           # no past frame for t=0
      out[:, 1:, :fold] = x[:, :-1, :fold]   # right-shift: t-1 -> t
      out[:, :,  fold:] = x[:, :,    fold:]  # remaining channels unchanged
    """

    def __init__(self, net: nn.Module, n_segment: int, n_div: int = 8) -> None:
        super().__init__()
        self.net = net
        self.n_segment = n_segment
        self.n_div = n_div

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        nt, c, h, w = x.shape
        n_batch = nt // self.n_segment
        x = x.view(n_batch, self.n_segment, c, h, w)
        fold = c // self.n_div

        # Causal right-shift: first `fold` channels carry cache (t-1 -> t).
        # Use slicing + cat so autograd graph stays clean (no in-place writes
        # into a zero tensor) and the cache region for t=0 is zero-padded.
        shifted = x[:, :-1, :fold]                              # (B, T-1, fold, H, W)
        pad = torch.zeros_like(x[:, :1, :fold])                 # zero for t=0
        left = torch.cat([pad, shifted], dim=1)                 # (B, T, fold, H, W)
        right = x[:, :, fold:]                                  # (B, T, C-fold, H, W)
        out = torch.cat([left, right], dim=2)                   # (B, T, C, H, W)

        out = out.reshape(nt, c, h, w)
        return self.net(out)


def inject_tsm(backbone: nn.Module, n_segment: int, n_div: int = 8) -> int:
    """Wrap conv[0] of every InvertedResidual with use_res_connect=True.

    Returns the number of injection sites (sanity counter; expect 10 for
    MobileNetV2 at any reasonable input size).
    """
    n_inj = 0
    for module in backbone.modules():
        if isinstance(module, InvertedResidual) and module.use_res_connect:
            module.conv[0] = CausalTemporalShift(
                module.conv[0], n_segment=n_segment, n_div=n_div
            )
            n_inj += 1
    return n_inj


# ---- TA model --------------------------------------------------------------

class TaTSMMobileNetV2(nn.Module):
    """Teacher Assistant: TSM-MobileNetV2 with per-frame classification head.

    Args:
        n_classes: 27 (Jester)
        n_segment: T (frames per clip) = 8
        n_div: 8 (causal-shift channel fraction = 1/n_div)
        dropout: applied before per-frame FC
        pretrained: load torchvision ImageNet weights on init
    """

    def __init__(
        self,
        n_classes: int = 27,
        n_segment: int = 8,
        n_div: int = 8,
        dropout: float = 0.5,
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        self.n_segment = n_segment
        self.n_div = n_div

        weights = MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
        mbv2 = mobilenet_v2(weights=weights)
        # Backbone features only — discard classifier (1000-way ImageNet)
        self.features = mbv2.features
        self.last_channel = mbv2.last_channel  # 1280

        n_inj = inject_tsm(self.features, n_segment=n_segment, n_div=n_div)
        # Expected at MobileNetV2: 10 InvertedResidual blocks with use_res_connect=True
        assert n_inj == 10, f"Unexpected TSM injection count: {n_inj} (expected 10)"
        self.n_tsm_sites = n_inj

        self.dropout = nn.Dropout(p=dropout)
        # Per-frame head: shared Linear applied to each time step's GAP feature
        self.fc = nn.Linear(self.last_channel, n_classes)

        # Re-init head (don't inherit ImageNet classifier weights, dim mismatch
        # anyway). MIT TSM uses normal_(std=0.001) for fresh head; mirror that.
        nn.init.normal_(self.fc.weight, mean=0.0, std=0.001)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, 3, H, W) -> logits (B, T, n_classes)."""
        B, T, C, H, W = x.shape
        assert T == self.n_segment, f"T mismatch: got {T}, expected {self.n_segment}"
        # Fold time into batch for 2D backbone; CausalTemporalShift internally
        # re-folds to (B, T, ...) for the shift, then back.
        x = x.reshape(B * T, C, H, W)
        feat = self.features(x)                  # (B*T, 1280, h, w)
        feat = feat.mean(dim=(2, 3))             # GAP -> (B*T, 1280)
        feat = self.dropout(feat)
        logits = self.fc(feat)                   # (B*T, n_classes)
        logits = logits.view(B, T, -1)           # (B, T, n_classes)
        return logits


# ---- Quick smoke test (run as script for sanity) ---------------------------

if __name__ == "__main__":
    torch.manual_seed(0)
    m = TaTSMMobileNetV2(n_classes=27, n_segment=8, n_div=8, pretrained=True)
    n_params = sum(p.numel() for p in m.parameters())
    print(f"params = {n_params / 1e6:.3f} M (expect ~3.5M)")
    print(f"TSM injection sites = {m.n_tsm_sites}")
    x = torch.randn(2, 8, 3, 112, 112)
    y = m(x)
    print(f"forward: x={tuple(x.shape)} -> y={tuple(y.shape)}")
    assert y.shape == (2, 8, 27), y.shape

    # Causality probe: change frame t=5; verify earlier frames' logits unchanged
    m.eval()
    with torch.no_grad():
        y1 = m(x)
        x_perturbed = x.clone()
        x_perturbed[:, 5] += 0.1 * torch.randn_like(x_perturbed[:, 5])
        y2 = m(x_perturbed)
        diff_per_frame = (y1 - y2).abs().mean(dim=(0, 2))  # mean abs over batch+class
        print("per-frame change after perturbing t=5:")
        for t in range(8):
            print(f"  t={t}: mean|Δlogit|={diff_per_frame[t].item():.6f}")
        # Frames 0..4 should be ~0 (causal); frames 5..7 should be > 0.
        assert diff_per_frame[:5].max() < 1e-6, (
            f"Causality violated: frames 0..4 changed by {diff_per_frame[:5].tolist()}"
        )
        assert diff_per_frame[5:].min() > 0, (
            f"Frames 5..7 unaffected (TSM not working): {diff_per_frame[5:].tolist()}"
        )
    print("causality check: PASSED")
