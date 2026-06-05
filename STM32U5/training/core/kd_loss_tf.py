"""Day 9-10: KD losses for TF/Keras student training (per-frame + video-level).

Formula (Plan Day 8):

    L_frame_t = alpha * CE(y_true, z_student_t)
              + (1 - alpha) * T^2 * KL(softmax(z_ta_t/T), softmax(z_student_t/T))

    L_total = sum_{r in R} w_r * L_frame_{frame_idx_student(r)}

with defaults:
    alpha = 0.3            (CE weight; 1 - alpha = KD weight, KD-dominated)
    T_temperature = 4.0
    R = (0.25, 0.50, 0.75, 1.0)
    frame_weights = (0.25, 0.25, 0.25, 0.25)   # uniform first pass

The student may have a different temporal length T_s from the TA (T_ta = 8).
We supervise student frame `frame_idx_student(r)` with TA frame
`frame_idx_ta(r)` — both computed by `student_frame_indices` (Plan Day 6).

KL direction (matches Hinton et al. 2015): teacher distribution as the
target — KL(p_teacher || p_student) penalizes student probability mass
**away** from teacher modes.
"""

from __future__ import annotations

from dataclasses import dataclass

import tensorflow as tf


@dataclass(frozen=True)
class KDConfig:
    """Day 10 default: INCREASING frame weights.

    Day 9 measured: TA per-frame[0] argmax matches train label only **10.7%** of
    the time (frame[7] = 95.5%). Uniform weights would route 25% of the KD signal
    through that noisy early-frame supervision -- effectively teaching the student
    to mimic ~random predictions on obs=0.25.

    Increasing weights [0.10, 0.20, 0.30, 0.40] put 70% of the loss on the
    reliable later frames (obs 0.75 & 1.0) and only 10% on the noisy obs=0.25.
    This matches the Plan Day 8 "scheme A" recommendation.
    """
    alpha: float = 0.3
    temperature: float = 4.0
    student_frame_idx: tuple = (1, 3, 5, 7)
    ta_frame_idx: tuple = (1, 3, 5, 7)
    frame_weights: tuple = (0.10, 0.20, 0.30, 0.40)


def per_frame_kd_loss(
    z_student: tf.Tensor,
    z_ta: tf.Tensor,
    y_true: tf.Tensor,
    cfg: KDConfig,
) -> dict:
    """Frame-weighted CE + KL distillation.

    Args:
        z_student: (B, T_s, K) student per-frame logits (FP32)
        z_ta:      (B, T_ta, K) TA per-frame logits (FP32; soft labels)
        y_true:    (B,) int32/int64 hard labels
        cfg:       KDConfig

    Returns:
        dict with keys 'loss' (scalar), 'ce' (scalar), 'kd' (scalar) — `loss`
        is the value to backprop, 'ce' and 'kd' are the unweighted-sum
        components for logging.
    """
    z_student = tf.cast(z_student, tf.float32)
    z_ta = tf.cast(z_ta, tf.float32)
    y_true = tf.cast(y_true, tf.int32)

    T = cfg.temperature
    alpha = cfg.alpha

    ce_total = tf.constant(0.0)
    kd_total = tf.constant(0.0)
    loss_total = tf.constant(0.0)
    for w, s_idx, t_idx in zip(cfg.frame_weights, cfg.student_frame_idx, cfg.ta_frame_idx):
        z_s = z_student[:, s_idx]       # (B, K)
        z_t = z_ta[:, t_idx]            # (B, K)

        ce = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_true, logits=z_s)
        ce = tf.reduce_mean(ce)

        p_t = tf.nn.softmax(z_t / T, axis=-1)
        log_p_s = tf.nn.log_softmax(z_s / T, axis=-1)
        log_p_t = tf.math.log(p_t + 1e-9)
        kl_per = tf.reduce_sum(p_t * (log_p_t - log_p_s), axis=-1)
        kl = tf.reduce_mean(kl_per) * (T * T)

        ce_total = ce_total + w * ce
        kd_total = kd_total + w * kl
        loss_total = loss_total + w * (alpha * ce + (1.0 - alpha) * kl)
    return {"loss": loss_total, "ce": ce_total, "kd": kd_total}


# ----------------------------------------------------------------------------
# Day 10 Path C2: video-level KD (single logit vector instead of per-frame T).
# ----------------------------------------------------------------------------

def video_level_kd_loss(
    z_student: tf.Tensor,
    z_ta: tf.Tensor,
    y_true: tf.Tensor,
    alpha: float = 0.3,
    temperature: float = 4.0,
    ta_reduce: str = "last",
) -> dict:
    """Video-level CE + KL KD (Path C2: Stacked-frame MobileNet).

    The C2 student predicts ONE logit vector per clip (not per frame). The TA
    output is (B, T_ta, K); we reduce it to (B, K) via `ta_reduce`:
      * 'last' (default): take z_ta[:, -1, :] -- uni-directional TA's last
        frame is its most confident prediction (95.5% match with hard label).
      * 'mean': average over T_ta -- TSN-style ensembling.

    Args:
        z_student: (B, K) student logits
        z_ta:      (B, T_ta, K) TA per-frame logits
        y_true:    (B,) int32
        alpha:     CE weight
        temperature: KD softmax temperature
        ta_reduce: 'last' | 'mean'
    """
    z_student = tf.cast(z_student, tf.float32)
    z_ta = tf.cast(z_ta, tf.float32)
    y_true = tf.cast(y_true, tf.int32)

    if ta_reduce == "last":
        z_ta_video = z_ta[:, -1, :]
    elif ta_reduce == "mean":
        z_ta_video = tf.reduce_mean(z_ta, axis=1)
    else:
        raise ValueError(f"unknown ta_reduce: {ta_reduce}")

    T = temperature
    ce = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_true, logits=z_student)
    ce = tf.reduce_mean(ce)

    p_t = tf.nn.softmax(z_ta_video / T, axis=-1)
    log_p_s = tf.nn.log_softmax(z_student / T, axis=-1)
    log_p_t = tf.math.log(p_t + 1e-9)
    kl_per = tf.reduce_sum(p_t * (log_p_t - log_p_s), axis=-1)
    kl = tf.reduce_mean(kl_per) * (T * T)

    loss = alpha * ce + (1.0 - alpha) * kl
    return {"loss": loss, "ce": ce, "kd": kl}


# ---- Sanity ---------------------------------------------------------------
if __name__ == "__main__":
    tf.keras.utils.set_random_seed(42)
    B, T_s, T_ta, K = 4, 8, 8, 27
    z_s = tf.random.normal((B, T_s, K)) * 0.3
    z_t = tf.random.normal((B, T_ta, K)) * 0.3
    y = tf.constant([0, 5, 13, 26], dtype=tf.int32)
    cfg = KDConfig()
    out = per_frame_kd_loss(z_s, z_t, y, cfg)
    print(f"loss={out['loss'].numpy():.4f}  ce={out['ce'].numpy():.4f}  kd={out['kd'].numpy():.4f}")
    # Identity: z_s == z_t should give kd = 0 (perfect distillation)
    out2 = per_frame_kd_loss(z_t, z_t, y, cfg)
    assert abs(out2["kd"].numpy()) < 1e-4, f"identity KD should be ~0, got {out2['kd'].numpy()}"
    print("identity KD check PASSED")
    # T_s=6 variant
    cfg6 = KDConfig(student_frame_idx=(1, 2, 4, 5), ta_frame_idx=(1, 3, 5, 7))
    z_s6 = tf.random.normal((B, 6, K)) * 0.3
    out3 = per_frame_kd_loss(z_s6, z_t, y, cfg6)
    print(f"T=6 variant loss={out3['loss'].numpy():.4f}")
    print("ALL KD-LOSS SANITY CHECKS PASSED")
