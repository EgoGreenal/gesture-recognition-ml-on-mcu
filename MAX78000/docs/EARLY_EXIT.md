# Early-Exit Policy

Early exit was explored as an energy-saving strategy for the sliding-window gesture-recognition pipeline.

## Baseline

The stable firmware keeps the camera history updated at the target backend cadence and invokes the CNN for every inference step.

## Experimental policy

The preserved early-stop variant is in:

```text
mcu/firmware/early_stop/
```

The policy idea is:

1. Run a shorter prefix model decision at 8 frames.
2. If confidence passes a class-specific threshold, hold that prediction and skip later CNN calls for a small number of frames.
3. Otherwise try a 10-frame prefix.
4. If 10 frames is still not confident, run the full 12-frame model.

The important constraint is that the video stream remains a sliding-window stream. The energy saving comes from skipping some CNN invocations after a confident accepted prediction, not from stopping camera capture.

## Limitation

Skipping CNN invocations saves compute, but it also lowers the temporal update rate while the prediction is held. This is why the policy is kept as an experimental firmware variant rather than the default stable demo.

## Current stable setting

The stable firmware keeps:

```c
#define ENABLE_EARLY_EXIT_POLICY 0
```

Use `mcu/firmware/early_stop/` when experimenting with early exit.
