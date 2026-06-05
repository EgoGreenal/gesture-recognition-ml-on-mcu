# MCU_FIRMWARE — STM32U5 Track

The on-device firmware is the STM32CubeIDE project
`../mcu/firmware/gesture_c1j_U5_board/` (firmware version **`C1j-ai-v2`**), targeting
the **B-U585I-IOT02A** (STM32U585AI, Cortex-M33 @ 160 MHz, Helium/MVE).

## What it does

Runs the deployed **C1j streaming INT8** model **one frame at a time** and reports
per-frame logits + a DWT cycle count, so the host can drive a real-time, early-exit
recognition loop over UART.

```c
void ai_streaming_reset(void);                       // zero frame + all 10 caches
int  ai_streaming_step(const int8_t* frame_in,       // 64*64 INT8, NHWC (4096 B)
                       int8_t* logits_out,           // 27 INT8 logits
                       uint32_t* cycles_out);        // DWT cycles for this step
```

The streaming model has **10 TSM cache tensors** (one per uni-directional shift
boundary). After each `ai_run`, each `cache_out[i]` is copied back into the matching
`cache_in[i]` for the next frame (`s_cache_in_idx` / `s_cache_out_idx` /
`s_cache_bytes` tables in `X-CUBE-AI/App/app_x-cube-ai.c`). Total cache ≈ 1.2 KB.

## UART protocol (`Core/Src/main.c`, 1-byte command loop on USART1 @ 115200)

| Cmd | Action | Reply |
|---|---|---|
| `P` | ping | `PONG` |
| `V` | version + sysclk | `SMOKE_FW_VERSION` line |
| `C` | snapshot DWT->CYCCNT | cycle count |
| `I` | reset AI streaming cache | `AI_RESET ok` |
| `F` | read 4096 INT8 bytes, run one streaming step | `INF cycles=<u32> us=<u32> argmax=<0..26> v=<int8>` + `LOGITS <27 int8>` |

Frame tensor quantization: scale 0.007843138, zero-point −1 (float [−1,1] → ≈[−128,126]).
Output (logits) quantization: scale 0.187850028, zero-point 66.

## Early exit

The firmware emits per-frame logits; the **early-exit decision is applied host-side**
(`host_benchmark.py`: S1 max-softmax, threshold 0.85, min_exit_frame 5). Folding the
decision into the firmware is ~10 lines (softmax-max + compare) and is the one
remaining device-side step. See [EARLY_EXIT.md](EARLY_EXIT.md).

## Measured performance (on board)

- **~141 ms/frame** compute (≈22.6 M cycles @ 160 MHz, DWT) — within the 150 ms/frame
  budget. RAM ≈ 261 KiB, Flash within the 2 MB budget.
- Full numbers: `../results/bench_summary.csv` and the project's own
  `../mcu/firmware/gesture_c1j_U5_board/bench_*.json`.

## Build

Open the project in STM32CubeIDE (it pulls X-CUBE-AI runtime from
`Middlewares/ST/AI/`, incl. `NetworkRuntime*_CM33_GCC.a`). Build outputs (`Debug/`,
`*.elf`, `*.map`, `.ai/` cache) are git-ignored. Enable the cmsis-nn / Helium
optimisation in the X-CUBE-AI settings for the measured latency.

## Relationship to the other firmware files

- `../firmware/main.c` — earlier **smoke-test** (`C1j-smoke-v0`, board bring-up only,
  no inference). Superseded by this project.
- `../deploy/main_template.c` + `../deploy/host_inference_reference.c` — earlier
  Day-13 *template* for the streaming loop (a clip-batch `S` + 8-frame protocol). The
  shipped firmware uses the per-frame `F` protocol above; see [HOST_DEMO.md](HOST_DEMO.md).
