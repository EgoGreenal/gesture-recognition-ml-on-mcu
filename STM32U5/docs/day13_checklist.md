# Day 13 Deployment Checklist — C1j INT8 streaming on STM32U5 (B-U585I-IOT02A)

**Goal**: no live-camera demo. The host PC feeds Jester val INT8 frames to the U5 over
UART; the U5 runs the multi-input streaming `ai_run` + S1 mf=5 thresh=0.85 early exit and
returns `(pred, exit_frame, cycles)` over UART; the host computes end-to-end INT8 accuracy
+ measured latency. Validation targets:
1. INT8 accuracy agrees with the host-side TFLite Interpreter (**86.49% baseline / 86.69% with S1 mf=5 thresh=0.85**)
2. Measured per-frame latency ≤ 150 ms (Plan budget)
3. Measured Flash + SRAM usage matches the Day 12.5 stedgeai generate report (**723 KB Flash / 68 KB SRAM**)

---

## Preparation (~30 min)

- [ ] Get a B-U585I-IOT02A board + USB Type-C cable (ST-LINK V3)
- [ ] Install STM32CubeIDE (Win/Mac/Linux; latest ≥ 1.14)
- [ ] Install the X-CUBE-AI 9.1+ extension (CubeIDE → Help → Manage Embedded Software Packages → STMicroelectronics → X-CUBE-AI)
- [ ] Install the ST-LINK V3 driver (installs on first board connect)
- [ ] Install PuTTY / minicom / pyserial (host serial tools)
- [ ] Copy `tools/arm-gnu-toolchain-13.2.Rel1-x86_64-arm-none-eabi/` locally (or use the toolchain bundled with CubeIDE, no manual install needed)
- [ ] **Copy the project deployment files locally**:
  - `models/student_C1J/deploy/C1j_ptq_streaming_sm_int8.tflite` (for X-CUBE-AI re-generate)
  - or: the whole `models/student_C1J/deploy/stedgeai_generate_int8/` dir (5 generated C files + network_data.c)
  - `deploy/main_template.c` (this directory)
  - `deploy/host_send_jester.py` (this directory)

## STM32CubeIDE project creation (1-2 hr)

- [ ] **File → New → STM32 Project** → Board Selector → search "B-U585I-IOT02A" → Next → Project Name e.g. "gesture_c1j" → Targeted Language: C → Finish
- [ ] Accept default initialization for all peripherals (CubeMX)
- [ ] **Configure UART1** (data stream, receive/send 4096-byte frame data to/from host): CubeMX Pinout View → Connectivity → USART1 → Mode: Asynchronous, baud rate 921600 (**note**: the default 115200 is too slow — 4096 bytes × T=8 frames × 100 samples = 3.2 MB; 115200 baud = ~11.5 KB/s = 280 s/sample test set, too slow; 921600 baud = ~92 KB/s = 35 s/sample test set; 2000000 baud is better if the U5 PCLK supports it)
- [ ] **Configure USART2** (debug printf output, to the ST-LINK virtual COM): usually on by default
- [ ] **Configure DCMI**: disable (no camera)
- [ ] **Generate code** → Project → Generate Code

## X-CUBE-AI integration (1 hr)

**Option A** (recommended): use the C code already generated on Day 12.5:
- [ ] Copy the 5 `.h/.c` + `LICENSE.txt` under `models/student_C1J/deploy/stedgeai_generate_int8/` into the CubeIDE project's `Application/User/X-CUBE-AI/App/`
- [ ] **Important**: the CubeIDE X-CUBE-AI version must be able to link the `stai_*` API produced by ST Edge AI Core 4.0 (X-CUBE-AI 9.1+ should support it, unconfirmed)
- [ ] Add the include path to the CubeIDE project: `Application/User/X-CUBE-AI/App/`

**Option B** (fallback): let the in-CubeIDE X-CUBE-AI re-run the INT8 TFLite:
- [ ] CubeMX → Software Packs → X-CUBE-AI → enable, mode "Add network"
- [ ] Add network → Model: TFLite → Browse → `C1j_ptq_streaming_sm_int8.tflite`
- [ ] Compression: lossless, Optimization: balanced
- [ ] Analyze → verify the numbers match Day 12.5 (macc 7.59M / weights 723 KB / activations 68 KB)
- [ ] Generate Code

## main.c integration (from main_template.c) (2-3 hr)

- [ ] Replace the CubeIDE-generated `Core/Src/main.c` with `deploy/main_template.c` (keep the CubeMX init code inside the USER CODE BEGIN/END regions)
- [ ] Fix the `#include` paths so it finds `student_c1j_ptq_int8.h` and the other generated headers
- [ ] **Verify the STM32U5 system clock = 160 MHz max** (CubeMX → Clock Configuration → SYSCLK = 160 MHz; if it is 80 MHz, change it to 160)
- [ ] Build → should be 0 error, 0 warning
- [ ] Flash usage should be ~723 KB weights + ~50 KB code = ~780 KB / 2 MB = **39%**
- [ ] SRAM usage should be ~68 KB activations + ~10 KB stack/heap/UART buf = ~78 KB / 786 KB = **10%**

## First flash + serial-link verification (0.5-1 hr)

- [ ] CubeIDE → Run → select ST-LINK, flash
- [ ] Open PuTTY / minicom: ST-LINK virtual COM (USART2), baud 115200
- [ ] Reset the board → see "MLonMCU gesture recognition ready"
- [ ] Connect the board's USART1 to the host via a USB-UART bridge (or the ST-LINK V3 secondary UART pin)

## Host-side INT8 accuracy verification (1-2 hr)

- [ ] On the host: `python deploy/host_send_jester.py --port /dev/ttyUSB0 --baud 921600 --n_samples 100`
- [ ] Run 100 Jester val samples, compare against the host TFLite Interpreter predictions
- [ ] **acceptance**: agreement rate ≥ 99% (INT8 inference should be bit-exact host vs board, since cmsis-nn computes in int8)
- [ ] If many disagreements, check:
  - the input frame quantization scale matches IN_1_SCALE=0.00784 / ZP=-1
  - the cache buffer mapping is correct (see the CACHE_PAIRS table in main_template.c)
  - the first-frame cache is initialized to zp, not 0 (int8 zero-point shift)

## Measured latency profile (0.5-1 hr)

- [ ] The DWT cycle counter is already wired in main_template.c; read DWT->CYCCNT around each frame's ai_run
- [ ] Host receives "cycles=N" → print ms = N / 160e6 × 1000
- [ ] Pareto budget check:
  - per-frame lat ≤ ~50 ms (Plan estimate 32ms is the 300 MHz formula; ~50-80 ms at 160 MHz is acceptable)
  - per-clip (T=8, no early-exit) ≤ ~400 ms
  - per-clip with S1 mf=5 thresh=0.85 (avg 6.2 frames): ≤ ~310 ms
- [ ] Verify cmsis-nn / Helium SIMD is active: the build log should show `cmsis-nn` optimisation; if not, CubeMX → X-CUBE-AI → Optimization → "Balanced (cmsis-nn)"

> **Measured result (done, supersedes the estimates above)**: the board measured
> **~141 ms/frame** (far above the ~50 ms estimate; effective Helium INT8 throughput
> ~57 MMAC/s, not the assumed ~300). per-clip ~880 ms (pure compute, **excluding UART
> wire time**; S1 mf=5 thresh=0.85, avg ~6.2 observed frames). **per-frame is still
> within the 150 ms/frame hard budget**, but per-clip is well above the ≤400/≤310 ms
> targets above. Authoritative numbers in
> [`../results/bench_summary.csv`](../results/bench_summary.csv) and
> [EARLY_EXIT.md](EARLY_EXIT.md).

## Write the Day 13 entry into Project_Plan_ch.md

- [ ] Real Flash / SRAM usage (from the CubeIDE build log)
- [ ] Real per-frame / per-clip latency (DWT cycles × 1000 / 160e6 ms)
- [ ] 100-sample agreement rate
- [ ] Any gotcha notes (X-CUBE-AI version mismatch / cmsis-nn off / wrong cache init / insufficient baud rate, etc.)

---

## Residual risks + fallbacks

| Risk | Fallback |
|---|---|
| **The bundled CubeIDE X-CUBE-AI version does not support the ST Edge AI Core 4.0 API** (`stai_*` functions have no link symbol) | Use Option B: re-generate inside CubeIDE's X-CUBE-AI (produces the older `ai_*` API), and replace `stai_*` with `ai_*` in main_template.c (the two APIs are equivalent; the name mapping is in the Day 12.5 generated header) |
| **921600 baud unstable on USART1, drops bytes** | Drop to 460800 or add flow control (RTS/CTS); accept doubled test time |
| **Wrong cache mapping (host vs board inference results differ)** | Have main_template.c print each IN/OUT `scale + zp + shape` at init and compare against IN_n_SCALE / OUT_n_SCALE in `student_c1j_ptq_int8.h` to find which OUT actually maps to which IN_n (the cache pair is the one whose scale+shape match) |
| **Measured lat > 150 ms/frame budget** | Check whether cmsis-nn is active; if it is active and still slow, fall back C1f → C1j is already a deploy-friendly variant, should be fine |
| **INT8 agreement < 99% (host vs board inference differ)** | Most likely a quantization-boundary or first-frame cache-init bug. Compare host TFLite Interpreter vs board ai_run intermediate output layer by layer (the X-CUBE-AI Validation tool supports this) |
