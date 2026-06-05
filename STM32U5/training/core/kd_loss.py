"""Day 6: Per-frame weighted KD loss (Swin-T video-level -> TA per-frame).

Implements the exact formula from Project_Plan_ch.md Day 6:

    L_t = alpha * CE(y_true, softmax(z_ta_t))
        + (1-alpha) * T^2 * KL(softmax(z_swin/T), softmax(z_ta_t/T))

    L_total = sum_t w_t * L_t

with defaults:
    alpha = 0.5
    T (temperature) = 4
    w_t = [0.0, 0.0, 0.0, 0.0, 0.05, 0.10, 0.20, 0.65]
          ── front frames not supervised (forcing the video label on causally
             starved early frames pollutes the representation); back frames
             get heavier weight (uni-directional shift gives them more info,
             so their prediction is closest to the video-level truth)

KL direction note (matching standard KD literature):
    F.kl_div(input=log(student_soft), target=teacher_soft) computes
        sum_k teacher_soft_k * (log teacher_soft_k - log student_soft_k)
    i.e. KL(teacher || student). This is the direction used in
    Hinton et al. 2015. The Plan writes KL(softmax(z_swin/T), softmax(z_ta_t/T))
    which is also KL(teacher || student).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# Plan default weights (Day 6); sum=1.0 by design (so per-clip loss scale is
# comparable across hyperparameter changes).
DEFAULT_FRAME_WEIGHTS = (0.0, 0.0, 0.0, 0.0, 0.05, 0.10, 0.20, 0.65)


class PerFrameKDLoss(nn.Module):
    """Weighted per-frame CE + KL distillation loss.

    Args:
        n_classes: 27
        temperature: KD softmax temperature (Plan: 4)
        alpha: CE weight (Plan: 0.5)
        frame_weights: tuple of length T (Plan: see DEFAULT_FRAME_WEIGHTS).
            Will be normalized via softmax of (sum=1) — but we keep the raw
            values as written in the Plan; sum=1.00 already.

    Inputs to forward():
        z_ta:  (B, T, n_classes)  TA per-frame logits (FP32 or AMP-cast)
        z_swin:(B, n_classes)     Super Teacher video-level logits (FP16 ok)
        y_true:(B,) int64         video-level hard label

    Returns a dict with:
        loss: scalar — sum over frames of w_t * L_t
        ce:   scalar — frame-weighted CE term only (for logging)
        kd:   scalar — frame-weighted KL term only (for logging)
    """

    def __init__(
        self,
        n_classes: int = 27,
        temperature: float = 4.0,
        alpha: float = 0.5,
        frame_weights: tuple[float, ...] = DEFAULT_FRAME_WEIGHTS,
    ) -> None:
        super().__init__()
        self.n_classes = n_classes
        self.T = float(temperature)
        self.alpha = float(alpha)
        # Register as buffer so it follows .to(device)
        self.register_buffer(
            "frame_weights",
            torch.tensor(frame_weights, dtype=torch.float32),
        )
        assert self.frame_weights.numel() > 0, "frame_weights must be non-empty"

    def forward(
        self,
        z_ta: torch.Tensor,
        z_swin: torch.Tensor,
        y_true: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        B, T, K = z_ta.shape
        assert T == self.frame_weights.numel(), (
            f"T mismatch with frame_weights: T={T}, weights={self.frame_weights.numel()}"
        )
        assert z_swin.shape == (B, K), z_swin.shape
        assert y_true.shape == (B,), y_true.shape

        # Cast everything to FP32 for loss math (AMP-safe). z_swin may come
        # from a FP16 npy; we use it as the KD target (pre-softmax logits).
        z_ta_f = z_ta.float()
        z_swin_f = z_swin.float()

        # --- CE per frame -----------------------------------------------------
        # CE expects (N, K) logits + (N,) labels. We expand label across T.
        z_ta_flat = z_ta_f.reshape(B * T, K)
        y_rep = y_true.view(B, 1).expand(B, T).reshape(B * T)
        ce_per_sample = F.cross_entropy(z_ta_flat, y_rep, reduction="none")  # (B*T,)
        ce_per_bt = ce_per_sample.view(B, T)

        # --- KL per frame -----------------------------------------------------
        # log_prob_ta_t at temperature T
        log_prob_ta = F.log_softmax(z_ta_f / self.T, dim=-1)            # (B,T,K)
        # teacher soft target: video-level, broadcast across T
        prob_swin = F.softmax(z_swin_f / self.T, dim=-1).unsqueeze(1)   # (B,1,K)
        prob_swin = prob_swin.expand(-1, T, -1)                          # (B,T,K)
        # KL(teacher || student) — Hinton-style — sum over class dim
        kl_per_bt = F.kl_div(
            log_prob_ta, prob_swin, reduction="none"
        ).sum(dim=-1)                                                   # (B,T)

        # Multiplied by T^2 (standard scale-restoration term in soft-label KD)
        kd_per_bt = (self.T * self.T) * kl_per_bt

        # --- Per-frame mix + weighted sum ------------------------------------
        per_frame = self.alpha * ce_per_bt + (1.0 - self.alpha) * kd_per_bt   # (B,T)
        w = self.frame_weights.to(per_frame.device, dtype=per_frame.dtype)    # (T,)
        # Sum over T with frame weights, then mean over batch.
        loss = (per_frame * w.unsqueeze(0)).sum(dim=1).mean()

        # Logging-only decomposition (same weighting; useful to track relative
        # contribution of hard-label vs soft-label term).
        ce_total = (self.alpha * ce_per_bt * w.unsqueeze(0)).sum(dim=1).mean()
        kd_total = ((1.0 - self.alpha) * kd_per_bt * w.unsqueeze(0)).sum(dim=1).mean()

        return {"loss": loss, "ce": ce_total, "kd": kd_total}


# ---- Sanity test ----------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    loss_fn = PerFrameKDLoss()
    B, T, K = 4, 8, 27
    z_ta = torch.randn(B, T, K, requires_grad=True)
    z_swin = torch.randn(B, K)
    y = torch.randint(0, K, (B,))
    out = loss_fn(z_ta, z_swin, y)
    print(f"loss={out['loss'].item():.4f}  ce={out['ce'].item():.4f}  kd={out['kd'].item():.4f}")
    out["loss"].backward()
    print(f"grad on z_ta exists: {z_ta.grad is not None}")
    # Frame 0..3 should have zero gradient contribution (weight=0).
    gnorm_per_t = z_ta.grad.norm(dim=(0, 2))
    print("grad norm per frame:", [f"{x.item():.4f}" for x in gnorm_per_t])
    assert gnorm_per_t[:4].max().item() == 0, "frames 0..3 must have zero gradient"
    assert gnorm_per_t[4:].min().item() > 0, "frames 4..7 must have nonzero gradient"
    print("PASSED")
