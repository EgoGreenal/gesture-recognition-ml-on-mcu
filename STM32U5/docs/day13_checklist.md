# Day 13 部署 Checklist — C1j INT8 streaming on STM32U5 (B-U585I-IOT02A)

**目标**：不做 live camera demo。Host PC 通过 UART 喂 Jester val INT8 帧到 U5，U5 跑 multi-input streaming ai_run + S1 mf=5 thresh=0.85 早退，UART 回 (pred, exit_frame, cycles)；Host 统计端到端 INT8 acc + 实测 lat。验证目标：
1. INT8 acc 与 host-side TFLite Interpreter 一致（**86.49% baseline / 86.69% with S1 mf=5 thresh=0.85**）
2. 实测 per-frame lat ≤ 150 ms（Plan budget）
3. Flash + SRAM 实测占用与 Day 12.5 stedgeai generate 报告一致（**723 KB Flash / 68 KB SRAM**）

---

## 准备 (~30 min)

- [ ] 拿到 B-U585I-IOT02A 板子 + USB Type-C 线 (ST-LINK V3)
- [ ] 装 STM32CubeIDE（Win/Mac/Linux 都可；最新版 ≥ 1.14）
- [ ] 装 X-CUBE-AI 9.1+ extension（CubeIDE → Help → Manage Embedded Software Packages → STMicroelectronics → X-CUBE-AI）
- [ ] 装 ST-LINK V3 driver（板子第一次连电脑会装）
- [ ] 装 PuTTY / minicom / pyserial（host 端串口工具）
- [ ] 把 `tools/arm-gnu-toolchain-13.2.Rel1-x86_64-arm-none-eabi/` 拷贝到本地（或者用 CubeIDE 自带的 toolchain，无需手装）
- [ ] **拷贝项目部署文件到本地**：
  - `models/student_C1J/deploy/C1j_ptq_streaming_sm_int8.tflite`（X-CUBE-AI 重 generate 用）
  - 或者：整个 `models/student_C1J/deploy/stedgeai_generate_int8/` 目录（含 5 个 generated C 文件 + network_data.c）
  - `deploy/main_template.c`（本目录）
  - `deploy/host_send_jester.py`（本目录）

## STM32CubeIDE 工程创建 (1-2 hr)

- [ ] **File → New → STM32 Project** → 选 Board Selector → search "B-U585I-IOT02A" → Next → Project Name e.g. "gesture_c1j" → Targeted Language: C → Finish
- [ ] 接受 default initialization for all peripherals (CubeMX)
- [ ] **配置 UART1**（数据流，向 host 收/发 4096-byte 帧数据）：CubeMX Pinout View → Connectivity → USART1 → Mode: Asynchronous, baud rate 921600（**注意**：默认 115200 太慢，4096 bytes × T=8 frames × 100 sample = 3.2 MB；115200 baud = ~11.5 KB/s = 280 秒/sample 测试集，太慢；921600 baud = ~92 KB/s = 35 秒/sample 测试集；2000000 baud 更好，但 U5 PCLK 看是否支持）
- [ ] **配置 USART2**（debug printf 输出，连接 ST-LINK virtual COM）：通常默认已开
- [ ] **配置 DCMI**: 禁用（不用 camera）
- [ ] **生成代码** → Project → Generate Code

## X-CUBE-AI 集成 (1 hr)

**方案 A**（推荐）：直接用 Day 12.5 已 generate 的 C 代码：
- [ ] 把 `models/student_C1J/deploy/stedgeai_generate_int8/` 下 5 个 `.h/.c` + `LICENSE.txt` 拷到 CubeIDE 工程 `Application/User/X-CUBE-AI/App/` 下
- [ ] **重要**：CubeIDE 的 X-CUBE-AI 版本必须能链接 ST Edge AI Core 4.0 生成的 `stai_*` API（应该 X-CUBE-AI 9.1+ 都支持，未确认）
- [ ] 在 CubeIDE 工程加 include path：`Application/User/X-CUBE-AI/App/`

**方案 B**（fallback）：让 CubeIDE 内的 X-CUBE-AI 重新跑一遍 INT8 TFLite：
- [ ] CubeMX → Software Packs → X-CUBE-AI → enable, mode "Add network"
- [ ] Add network → Model: TFLite → Browse → `C1j_ptq_streaming_sm_int8.tflite`
- [ ] Compression: lossless, Optimization: balanced
- [ ] Analyze → 验证数字匹配 Day 12.5 (macc 7.59M / weights 723 KB / activations 68 KB)
- [ ] Generate Code

## main.c 集成（复制 main_template.c）(2-3 hr)

- [ ] 用 `deploy/main_template.c` 替换 CubeIDE 生成的 `Core/Src/main.c`（保留 USER CODE BEGIN/END 区域里的 CubeMX init 代码）
- [ ] 修改 `#include` 路径让它能找到 `student_c1j_ptq_int8.h` 等 generated headers
- [ ] **验证 STM32U5 system clock = 160 MHz max**（CubeMX → Clock Configuration → SYSCLK = 160 MHz；如果是 80 MHz 改成 160）
- [ ] Build → 应该 0 error 0 warning
- [ ] Flash 占用应该是 ~723 KB weights + ~50 KB code = ~780 KB / 2 MB = **39%**
- [ ] SRAM 占用应该是 ~68 KB activations + ~10 KB stack/heap/UART buf = ~78 KB / 786 KB = **10%**

## 第一次烧板 + 串口连接验证 (0.5-1 hr)

- [ ] CubeIDE → Run → 选 ST-LINK，烧
- [ ] 打开 PuTTY / minicom：ST-LINK virtual COM (USART2)，baud 115200
- [ ] 板上 Reset → 看到 "MLonMCU gesture recognition ready"
- [ ] 板子的 USART1 通过 USB-UART 桥（或者直接用 STLink V3 的 secondary UART pin）连到 host

## Host-side INT8 accuracy 验证 (1-2 hr)

- [ ] 在 host 上：`python deploy/host_send_jester.py --port /dev/ttyUSB0 --baud 921600 --n_samples 100`
- [ ] 跑 100 个 Jester val sample，对比 host TFLite Interpreter 预测
- [ ] **acceptance**: agreement rate ≥ 99%（INT8 推理在 host vs board 应该 bit-exact，因为 cmsis-nn 用 int8 算）
- [ ] 如果 disagreement 多，检查：
  - input frame quantization scale 是否与 IN_1_SCALE=0.00784 / ZP=-1 一致
  - cache buffer mapping 是否正确（参考 main_template.c 里的 CACHE_PAIRS table）
  - 第一帧 cache 是否初始化为 zp 而非 0（int8 zero point shift）

## 实测延迟 profile (0.5-1 hr)

- [ ] DWT cycle counter 在 main_template.c 里已经接好，每帧 ai_run 前后读 DWT->CYCCNT
- [ ] Host 收到 "cycles=N" → 打印 ms = N / 160e6 × 1000
- [ ] Pareto budget check：
  - per-frame lat ≤ ~50 ms (Plan 估算 32ms 是 300 MHz 公式；160 MHz 实际 ~50-80 ms 可接受)
  - per-clip (T=8, no early-exit) ≤ ~400 ms
  - per-clip with S1 mf=5 thresh=0.85 (avg 6.2 frames): ≤ ~310 ms
- [ ] 验证 cmsis-nn / Helium SIMD 是否 active：build log 应该出现 `cmsis-nn` 优化字样；如果没有，CubeMX → X-CUBE-AI → Optimization 选 "Balanced (cmsis-nn)"

> **实测结果（已完成，覆盖上面估算）**：板上实测 **~141 ms/frame**（远超上面 ~50 ms 估算；
> 有效 Helium INT8 吞吐 ~57 MMAC/s 而非假设的 ~300）。per-clip ~880 ms（纯 compute，
> 不含 UART 线时；S1 mf=5 thresh=0.85，avg ~6.2 观察帧）。**per-frame 仍在 150 ms/frame 硬预算内**，但 per-clip 远高于
> 上面的 ≤400/≤310 ms 目标。权威数字见 [`../results/bench_summary.csv`](../results/bench_summary.csv)
> 与 [EARLY_EXIT.md](EARLY_EXIT.md)。

## 整理 Day 13 entry 写到 Project_Plan_ch.md

- [ ] 真实 Flash / SRAM 占用 (CubeIDE Build Log 报)
- [ ] 真实 per-frame / per-clip lat (DWT 数字 × 1000 / 160e6 ms)
- [ ] 100-sample agreement rate
- [ ] 如有，踩坑笔记（X-CUBE-AI 版本 mismatch / cmsis-nn off / cache 初始化错 / baud rate 不够等）

---

## 风险残量 + Fallback

| 风险 | Fallback |
|---|---|
| **CubeIDE 自带 X-CUBE-AI 版本不支持 ST Edge AI Core 4.0 API**（`stai_*` 函数找不到 link symbol）| 用方案 B：CubeIDE 内 X-CUBE-AI 重新 generate（产生 `ai_*` 旧 API），同步把 main_template.c 里 `stai_*` 替换为 `ai_*`（两套 API 等价，函数名映射可对照 Day 12.5 generated header） |
| **921600 baud 在 USART1 不稳，丢字节** | 降到 460800 或加流控（RTS/CTS）；接受测试时间翻倍 |
| **cache mapping 错（看 host vs board 推理结果不一致）** | main_template.c 里 print 每个 IN/OUT 的 `scale + zp + shape` 在 init 时，与 `student_c1j_ptq_int8.h` 里的 IN_n_SCALE / OUT_n_SCALE 对比，看哪个 OUT 实际对应 IN_n（理论上 scale+shape 匹配的就是 cache pair） |
| **实测 lat > 150 ms/frame budget** | 检查 cmsis-nn 是否 active；若 active 还慢，回退到 C1f → C1j 已经是 deploy-friendly variant，应该没问题 |
| **INT8 agreement < 99%（host vs board 推理不一致）** | 大概率是 quantization 边界处理 / 第一帧 cache init 错。逐 layer 比 host TFLite Interpreter vs board ai_run intermediate output（X-CUBE-AI Validation tool 支持） |
