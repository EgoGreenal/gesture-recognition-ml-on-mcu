# Day 8 TSM 部署路径探针实验报告

**日期**: 2026-05-12
**目标**: 决定 Day 13 部署路径 — Path A.streaming（单 TFLite + cache-as-IO）vs Path B（multi-network fallback）
**结论**: ✅ **Path A.streaming — GO**（A.streaming 主路径，Day 9-10 训练全部 7 个 variant V0-V6 都可推进）

---

## 探针模型设计

最小可行 streaming student（**故意做小**，只为验证拓扑 + 量化通路；不是 student variant 候选）：

```
frame_in [1, 32, 32, 1]                                ch=1   spatial 32x32
   │
   │  Conv2D(8, 3, stride=2) → BN → ReLU                ch=8   spatial 16x16
   │
   │  TSMStreamingShift  ← cache_0_in [1, 16, 16, 1]    ch=8 (n_shift=1)
   │                    → cache_0_out [1, 16, 16, 1]
   │  Conv2D(16, 3, stride=2) → BN → ReLU               ch=16  spatial 8x8
   │
   │  TSMStreamingShift  ← cache_1_in [1, 8, 8, 2]      ch=16 (n_shift=2)
   │                    → cache_1_out [1, 8, 8, 2]
   │  Conv2D(32, 3, stride=2) → BN → ReLU               ch=32  spatial 4x4
   │
   │  GlobalAveragePool                                 (1, 32)
   │  Dense(27, no activation)                          (1, 27) logits
   ▼
[logits, cache_0_out, cache_1_out]
```

**关键设计点**:
- 输入通道选取保证 `n_shift = C / n_div = C/8` 在每个 TSM 边界都是整数（8/8=1, 16/8=2）— 避免 TFLite Slice 量化 scale 不齐的 corner case
- 每个 `Conv → BN → ReLU` 三元组结尾是 ReLU，TSM cut 点在 ReLU 之后 — 满足 Plan Day 8 "BN folding 边界规则"
- TSM shift 用纯 `tf.split + tf.concat` 实现，不需要 custom op

详见 [scripts/tsm_streaming_layer.py](scripts/tsm_streaming_layer.py) + [scripts/tsm_probe.py](scripts/tsm_probe.py)。

---

## Check 1 — INT8 量化保真度

**目标**: FP32 Keras 模型 → INT8 TFLite (post-training quantization) 后输出分布偏差 < KL 0.05

**流程**:
1. Keras model → `tf.saved_model.save` → `tf.lite.TFLiteConverter.from_saved_model`
2. `representative_dataset` 用 dict-keyed yield（多输入模型 calibrator 输入顺序坑）
3. `target_spec.supported_ops = [TFLITE_BUILTINS_INT8]` + `inference_input_type = INT8` + `inference_output_type = INT8` → fully quantized I/O
4. 32 个随机 [-1, 1] uniform 样本，每个跑 FP32 Keras + INT8 TFLite Interpreter，对比 logits 的 softmax KL

**结果**:

| 指标 | 值 | 评注 |
|---|---|---|
| samples | 32 | |
| KL(FP32 ‖ INT8) | **0.000001** | 阈值 < 0.05 ✓ |
| KL(INT8 ‖ FP32) | 0.000001 | 对称对照 |
| argmax 一致率 | **1.0000** | 全部 32 个样本预测一致 |

**verdict: PASS** ✓

注：模型未训练（随机初始化），输出 logits 接近均匀分布，量化几乎没东西可丢。**这里验证的是拓扑 + 量化通路本身**，不是精度。真实 student variant 训完后 INT8 KL 应该上升但仍远低于 0.05（Day 11 QAT 还会再压一次）。

---

## Check 2 — stedgeai analyze（X-CUBE-AI 多 I/O + ops 支持）

**目标**: ST Edge AI Core 4.0 (= STM32CubeAI 12.0) 能否：
1. 接受多输入多输出 INT8 TFLite（3 inputs / 3 outputs）
2. 把 TSM 的 STRIDED_SLICE + CONCATENATION 正确量化保真
3. 给出 MAC / RAM / ROM 估算

**命令**:
```bash
stedgeai analyze \
    --model streaming_probe_int8.tflite \
    --target stm32 \
    --name streaming_probe \
    --workspace logs/tsm_probe/stedgeai/workspace \
    --output logs/tsm_probe/stedgeai/output \
    --verbosity 2
```

**结果**:

| 指标 | 值 |
|---|---|
| stedgeai 版本 | ST Edge AI Core v4.0.0-20500 (STM32CubeAI 12.0.0) |
| returncode | **0** ✓ |
| multi-I/O 接受 | **True** ✓ |
| 模型类型 | tflite (fully INT8, `ss/sa per channel`) |
| 模型 hash | `0x96cc70c7739a60adb4adc6f87ad69071` |
| params # | 6,752 items (**6.76 KiB**) |
| inputs total | 1.38 KBytes (frame 1024 + cache_0 256 + cache_1 128) |
| outputs total | 411 Bytes (logits 27 + cache_0_out 256 + cache_1_out 128) |
| MACs | **170,419** |
| weights (RO) | 7,028 B (**6.86 KiB**) — 比 FP32 模型 -74.0% |
| activations (RW) | 11,136 B (**10.88 KiB**) |
| RAM 总和 | 11,136 B (10.88 KiB) |
| ROM (AI RT 代码) | 未给出（需要 `arm-none-eabi-gcc` 在 PATH 才能算 runtime 代码体积；Day 13 烧板前再装 ARM toolchain） |
| 工具耗时 | 20.82 s |
| analyze 进度 | 162/171 PASS（剩 9 项 skipped 因为没装 GCC） |

**verdict: PASS** ✓

**stedgeai 输出节选**（关键拓扑被识别）:
```
m_id   layer (original)                  oshape                 param/size    macc
0      serving_default_frame_in0         [b:1,h:32,w:32,c:1]
       conv2d_0 (CONV_2D)                [b:1,h:16,w:16,c:8]    80/104         18,440
       nl_0_nl (CONV_2D)                 [b:1,h:16,w:16,c:8]                    2,048   <- ReLU as fused-conv
1      slice_1 (STRIDED_SLICE)           [b:1,h:16,w:16,c:1]                            <- TSM cache 切
2      slice_2 (STRIDED_SLICE)           [b:1,h:16,w:16,c:7]                            <- TSM static 切
3      concat_3 (CONCATENATION)          [b:1,h:16,w:16,c:8]                            <- TSM 拼回
...
```

**关键发现**:
1. **多 I/O 完全 OK**：3 inputs（frame + 2 cache）、3 outputs（logits + 2 cache_out）都被识别为正常张量，没有触发任何 op 不支持错误
2. **TSM 的 SLICE/CONCAT 干净保留**：每个 TSM 边界分解为 `STRIDED_SLICE×2 + CONCATENATION×1`，stedgeai 把它们识别为标准 ops 并量化（per-channel ss/sa scale）
3. **量化 scale 跨 SLICE/CONCAT 节点自动同步**：cache_0_in 和 cache_0_out 都是 `QLinear(0.008567031, -11, int8)` — scale + zero point 完全一致，意味着 cache buffer 在 C 端可以**直接 memcpy** 不需要 rescale
4. **Plan section 3 要求 ≥ X-CUBE-AI 9.1（即 STM32CubeAI 9.x）**；实测装的 STM32CubeAI 12.0 是更新版本，op 支持更完整

---

## Check 3 — U5 板上 DWT 实测

**状态**: **跳过**（HPC Leonardo 上无 U5 板）

**用估算公式代替**（Plan Section 5 修正公式）:
```
latency_ms = MACs / 300e6 × 1000 + N_ai_run_calls × 5ms
```
- Probe model MACs = 170,419 ≈ 0.17M
- 8 frames × 1 ai_run/frame = 8 ai_run calls
- 估算: `0.17M / 300M × 1000 + 8 × 5 = 0.6 + 40 = ~40.6 ms` for 8-frame clip

Probe model 是故意做小的，跑这点 latency 不代表 V0-V6 的真实数字。但 stedgeai 拿到的 MACs 是 **per-frame** 而不是 per-clip，对 Day 9-10 各 variant 跑同样 analyze 拿真实 MACs，再代入公式得每帧延迟。

---

## 决策与下一步

### 部署路径锁定: **Path A.streaming**

Day 13 部署用单 TFLite + C 端 cache buffer 管理，单 `ai_run()` per frame，每帧 overhead 仅 ~5ms × 8 = 40ms。Multi-network fallback (Path B) **不需要训练 V0/V1 收缩**。

### Day 9-10 variant 池放开到全 7 个

按 Plan Section 5 表格，A.streaming 单 NN 路径下：

| Variant | 估算 latency | 可行 |
|---|---|---|
| V0 Ultra-Tiny | 140 ms | ✓ |
| V1 Tiny | 97 ms | ✓ |
| V2 Small | 307 ms | ⚠ 超 200ms |
| V3 Small-DW | 190 ms | ✓ 边界 |
| V4 Deep | 373 ms | ✗ |
| V5 Wide | 973 ms | ✗ |
| V6 HiRes | 473 ms | ✗ |

实际跑训练时，估算可能 ±30%。Day 9-10 跑完 fp32 训练后，对每个 variant 都跑一次 `stedgeai analyze` 拿真 MACs / RAM，**用真实数字决定 Day 11 QAT top-3**。

### 不需要的 fallback 工作
- ❌ 不需要 implement `block_exporter.py`（Path B 才需要的多 sub-network 切分）
- ❌ 不需要在 block 边界插 identity FakeQuant + 自定义 `tfmot.QuantizeConfig`（QAT scale 一致性 — Path B 才需要）
- ❌ 不需要写 `tsm_shift.c`（C 端 memcpy shift — Path B 才需要）
- ✅ 仍然需要：student model factory + frame-level KD loss + 训练脚本

---

## Probe 工具产出

- [scripts/tsm_streaming_layer.py](scripts/tsm_streaming_layer.py) — TSMStreamingShift Keras layer + `causal_tsm_clip` clip-form helper（训练时用 clip-form，部署时 unroll 成 streaming step）
- [scripts/tsm_probe.py](scripts/tsm_probe.py) — probe 全流程脚本（build → INT8 export → Interpreter cross-check → stedgeai analyze → JSON 报告）
- `logs/tsm_probe/` — 完整产出目录：
  - `streaming_probe_fp32.tflite` — 30.6 KB
  - `streaming_probe_int8.tflite` — 12.9 KB
  - `saved_model/` — TF SavedModel 中间格式
  - `stedgeai/workspace/` + `stedgeai/output/` — stedgeai 完整工作目录
  - `stedgeai/analyze_log.txt` — stedgeai 完整运行日志
  - `tsm_probe_results.json` — 机器可读结果摘要
- [tools/stedgeai/4.0/](tools/stedgeai/4.0/) — ST Edge AI Core 4.0 (= STM32CubeAI 12.0) 完整安装树
- [scripts/env_setup.sh](scripts/env_setup.sh) — 加了 `stedgeai` 到 PATH（append-not-prepend，避开 bundled python 3.9.13 shadow tf_env 的 3.11.7）
