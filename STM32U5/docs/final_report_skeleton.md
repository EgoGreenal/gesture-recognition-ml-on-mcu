# 实时手势识别在 STM32U5：TAKD + QAT + 早退 + 部署
**Project Final Report — MLonMCU 2026 Spring**

*Author: Yizhou Xu*  *Project days: 14*

---

## TL;DR (报告 1 段, 50 words)
在 STM32U5 (B-U585I-IOT02A, Cortex-M33 @ 160 MHz, 786 KB SRAM, 2 MB Flash) 上部署了一个实时手势识别系统：27-class Jester，部署模型 C1j streaming INT8。无早退 INT8 baseline（full val 14787）86.49%；板上实测（firmware C1j-ai-v2 流式 `ai_streaming_step` + 早退 S1 mf=5 thresh=0.85，平均 obs 0.778，n=128）accuracy 84.4%。**实测延迟 ~141 ms/frame**（DWT 实测，<150 ms/frame budget 通过），~880 ms/clip（含 4 KB/帧 UART 传输）；注意离线 X-CUBE-AI 估算 ~32–70 ms/frame 偏低 2–4×。Flash 0.99 MB / SRAM 261 KB。教师 → TA → student 三阶段 KD 蒸馏 (TAKD) + 17 层 TSM-MBV2 cone 截断 (w=0.5) + per-channel symmetric INT8 PTQ + min_exit_frame 约束早退。所有 U5 硬约束均通过（Flash<2MB / SRAM<600KB / lat<150ms/frame）。

---

## 1. 引言 + 项目背景
- 真实场景动机（实时手势识别 → 灯控）
- 选 Jester 数据集（27 class, 中等规模 segment 视频）
- 选 STM32U5 (B-U585I-IOT02A)：教学板，Cortex-M33 + X-CUBE-AI 9.1+
- 项目里程碑：14 天 × 主题（Day 1-2: data，3-5: teacher，6-7: TA + KD，8: probe，9-10: arch search，11: quantization，12: early-exit，13-14: deployment）

## 2. 数据集与预处理
- Jester v1: 119k train + 14787 val + 14743 test (no labels)
- 27 类（含 left/right 镜像 → flip-aware）
- Pipeline L (TF/Keras)：64×64 grayscale @ T=8（segment sampling, 帧间均匀采）
- HDF5 存储 + tf.io.decode_jpeg 在 graph 中解码（GIL-free）
- KD soft labels：teacher / TA 各跑一遍 train + val，per-frame (B,T_ta=8,K=27) FP16
- 关键 finding (Day 6)：left/right 镜像翻转时 label 必须做 FLIP_LABEL_MAP 重映射，否则 KD 信号反向 (~25% acc cost)

## 3. 模型架构 — TAKD 三级蒸馏
### 3.1 Teacher (Day 3-4)
- Swin-T (~28M params) on Jester
- val acc：**[FILL — find from models/super_teacher.../metrics.json]**

### 3.2 Teacher Assistant (Day 6-7)
- TSM-MobileNetV2 full (17 blocks, w=1.0, RGB 112×112)
- ~2.5M params; KD from teacher with α=0.3, T=4
- val acc: **93.12%** (per-frame[7] = 95.5% argmax-train-label)

### 3.3 Student family — Path C decision (Day 10)
**V0-V9 from-scratch tiny CNN 路线 dead-end**（V4 上限 38.45%），改 Path C 三路并行：
- C1: TSM-MBV2 truncated（与 TA 同源拓扑 + width_mult）→ 全胜，C1b 84%
- C2: Stacked-frame MBV2/V1（8 帧 → channel 轴）→ 60-69%，video-level KD 丢早期帧
- C3: Hybrid（前 2D + 后 TSM）→ 39-57%，TSM site 太少

C1 family 深挖到 88.46% (C1g)，**Top-2 候选**：
- **C1f** (1.39M, 87.35% FP32)：拼精度
- **C1j** (722K, 86.65% FP32)：拼部署效率

## 4. 知识蒸馏 (TAKD) — Plan vs 实际
- Plan：teacher (28M) → TA (2.5M) → student (722K-1.4M)
- 几何均值原则：√(28e6 × 0.7e6) ≈ 4.4M ≠ TA 2.5M — 但 TA 在 ImageNet pretrain 加成下表现优异
- 实际：α=0.3 + T=4 + 递增 frame_weights [0.10, 0.20, 0.30, 0.40]
- **递增 frame_weights 关键**：TA frame[0] argmax=10.7%, frame[7]=95.5%；均匀权重会让 25% KD 信号走 noisy 早期帧

## 5. 量化 (Day 11)
### 5.1 PTQ INT8 (用作 default)

| Variant | FP32 obs@1.0 | **PTQ INT8 obs@1.0** | drop |
|---|---:|---:|---:|
| C1f | 87.35% | **87.02%** | -0.33 pp |
| C1j | 86.65% | **86.49%** | -0.16 pp |

PTQ 表现明显好于 Plan 预期（预期 1-3 pp，实际 ≤ 0.33 pp）。原因：grayscale 1-channel + per-channel weight quant + KD 让 logits 范围对后帧 well-clustered。

### 5.2 QAT — 跳过（含理由）
- tfmot 0.8.0 + TF 2.15 不接受嵌套 Functional + TimeDistributed（tfmot 检查 type 失败）
- 手写 fake-quant STE（[fake_quant_qat.py](../scripts/fake_quant_qat.py)）作 fallback 写完、smoke 通过，未投训练
- 决策：PTQ 已满足 Plan 默认档 (≥85%)，QAT 边际增益 ≤ 0.5 pp 在判决噪声内
- 节省 ~2-3 hr 转移给 Day 13 部署（项目最难一天）

### 5.3 Multi-input streaming INT8 PTQ — 两个 workaround
- 第一种 `from_keras_model` + 位置 list：Day 10/11 **失败**（不同错误点：positional reordering / channel divisibility）
- 第二种 `from_saved_model` + dict-keyed generator (Day 11 后期发现 / Day 13 prep)：**成功** ✓
  - C1f streaming INT8: 1.75 MB
  - C1j streaming INT8: 0.99 MB

## 6. 早退策略 (Day 12)
### 6.1 Vanilla S1/S3 不工作（重要发现）
- Plan 引用 paper 默认 S1 thresh=0.85 / S3 entropy<0.5
- 在 C1 INT8 模型上：**14% acc / obs 0.21** — 70% 样本在 frame 0 退出（INT8 logits 量化后更 peaked，"confidently wrong"）

### 6.2 修正：min_exit_frame=5 floor
- C1 模型 per-frame acc 是 ramp：frame 0 ~10%，frame 5 ~86%，frame 7 ~87%
- mf=5 强制最早 obs ≥ 0.75（per-frame acc 已可靠）
- S1 mf=5 thresh=0.85：C1f 86.66% obs 0.776, C1j **86.69% obs 0.776**

### 6.3 U5 部署组合选定
**(C1j, S1, mf=5, thresh=0.85)** ⭐

| Metric | Value |
|---|---:|
| INT8 acc (full Jester val) | **86.69%** |
| Drop vs C1j baseline | 0.0 pp (=baseline) |
| Drop vs C1j FP32 | -0.66 pp |
| 平均 obs ratio | 0.776 |
| 平均 frame count | 6.2 / 8 |
| 离线估算 lat/clip | 199 ms（X-CUBE-AI 估算，偏低）|
| **实测 lat/frame (DWT)** | **~141 ms/frame**（<150 budget 通过）|
| **实测 lat/clip (含 UART)** | **~880 ms**（板上 bench, n=128, S1 mf5 thr0.85）|
| 实测 on-device acc (n=128) | 84.4% |
| C1j INT8 Flash | 0.99 MB |
| C1j SRAM | 261 KB |
| C1j per-frame MACs | 8.13M |

## 7. U5 部署 (Day 13)
**[FILL —待 Day 13 完成]**

### 7.1 X-CUBE-AI 9.1 集成
### 7.2 Cache buffer 维护代码
### 7.3 早退 hook + INT8 softmax 实现
### 7.4 LED1-4 灯控映射

## 8. 系统级评估 (Day 14)
**[FILL — Day 14]**
- 端到端 latency profiling（板上实测 vs Plan 估算）
- 误识别 case study
- 实时演示视频

## 9. 经验教训 (踩坑日记 summary)
1. **From-scratch tiny CNN ≠ 路线**（V0-V9 上限 38%，与 TA 同源拓扑 + KD 蒸馏的 C1 跳到 84%）
2. **tfmot QAT 与 nested Functional 不兼容**（TF 2.15 + tfmot 0.8.0，PTQ 已够则跳过 QAT）
3. **Multi-input INT8 PTQ 的 SavedModel workaround**（dict-keyed calibrator 是 silver bullet）
4. **vanilla S1/S3 早退失败 / min_exit_frame floor 是关键**（INT8 让 softmax 更 peaked, "confidently wrong" 在早帧）
5. **HDF5 + Lustre + multi-job**（HDF5_USE_FILE_LOCKING=FALSE + 不 SWMR）
6. **递增 frame_weights** (TA 早帧噪声 → 不能用均匀权重)
7. **dbg QoS submit limit + 25min cap + 7min grace**（lprod 1h cap 给保险）
8. **GPU/算力预算**：项目用 ~42 GPU-hr（Plan 100 GPU-hr 预算保留 ~58）

## 10. 结论与未来工作
- 实现了 STM32U5 上 27-class 实时手势识别：C1j streaming INT8，no-exit full-val baseline 86.49–86.69%，板上实测（S1 mf=5 thr=0.85）on-device acc 84.4%（n=128），**实测 ~141 ms/frame（<150 budget 通过）**、~880 ms/clip（含 UART）、Flash 0.99 MB
- 早退策略 mf=5 + S1=0.85：平均 obs 0.778（exit frame ~5.2，即观察 ~6.2/8 帧），相对 no-exit 省 ~22% 帧数 / 推理时间
- 未来方向：
  - 真正的 QAT（如能 flat-inline 重建 C1 模型避开 tfmot bug）
  - 拓展到 RGB 输入 + 提升 obs_0.25-0.50 准确率（让早退 floor 能降到 frame 3-4）
  - 板上多模态（IMU/音频）集成

---

## 附录 A：所有 stedgeai analyze 数字
**[FILL — 从 models/qat_comparison.csv + Day 13 stedgeai generate 报告]**

## 附录 B：repro 指令
**[FILL — 从 README]**

## 附录 C：每日 commit history + 资源核算
**[FILL — Plan Section 9 摘要]**
