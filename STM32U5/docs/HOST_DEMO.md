# HOST_DEMO — STM32U5 Track

There is **no live-camera demo** on the U5 track (unlike MAX78000). Instead, the host
feeds Jester clips to the board over UART and collects predictions + measured latency.
This is the "demo" / evaluation harness.

## Shipped harness (firmware `C1j-ai-v2`, per-frame `F` protocol)

In `../mcu/firmware/gesture_c1j_U5_board/`:

- **`host_frame_test.py`** — smoke test of the `F` command. Sends `I` (reset) then
  `F` + 4096 INT8 bytes per frame, prints the board's `INF …` / `LOGITS …` reply.
  ```bash
  pip install pyserial numpy
  python host_frame_test.py --n-frames 8 --pattern tframe   # or --pattern zeros --no-reset
  ```
- **`host_benchmark.py`** — the real benchmark. Streams clips frame-by-frame, applies
  the **S1 early-exit** rule host-side (`min_exit_frame=5`, `threshold=0.85`), and
  reports accuracy, mean exit frame, observation ratio, and cycle-based latency.
  ```bash
  python host_benchmark.py --synthetic 10                              # no board data
  python host_benchmark.py --data jester_val_int8.npz --max-samples 100 \
                           --save-json bench_100.json
  ```
  Constants are locked to the current firmware (out scale 0.187850028 / zp 66, sysclk
  160 MHz; override with `--sysclk-hz`). **Its latency metric is inference-cycle based
  and excludes UART wire time by design.** Outputs are the committed `bench_*.json`
  (summarised in `../results/bench_summary.csv`).

## Earlier Day-13 template (clip-batch `S` protocol)

`../deploy/host_send_jester.py` is the companion to `../deploy/main_template.c`. It
uses a different, earlier protocol: send `S` then 8 frames; board replies `F <t>
<cycles>` per frame and `R <pred> <exit> <total_cycles>` at clip end. It also compares
board predictions against the host TFLite interpreter.

```bash
python deploy/host_send_jester.py --port /dev/ttyUSB0 --baud 921600 --n_samples 100
```

This template predates the shipped `F`-protocol firmware; use `host_benchmark.py`
against `gesture_c1j_U5_board` for current results. The two are kept for reference and
because the `S`-protocol design is closer to a self-contained on-board loop (the board
holds the exit decision) — the intended next firmware step (see
[MCU_FIRMWARE.md](MCU_FIRMWARE.md), [EARLY_EXIT.md](EARLY_EXIT.md)).

## Preparing input data

Clips are exported to INT8 `.npz` (the format `host_benchmark.py --data` expects) by
`dump_val_clips_int8.py` (a copy ships inside the firmware project).
