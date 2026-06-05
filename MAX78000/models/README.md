# Models

This directory contains lightweight model metadata and source definitions.

- `best_deployable/`: selected MAX78000-compatible model metadata, labels, cache/training config, and ai8x model definition.
- `notes/`: architecture notes used during model comparison.

Large training checkpoints are not included here. The deployed firmware includes generated C weights in `../mcu/firmware/stable/weights.h`.
