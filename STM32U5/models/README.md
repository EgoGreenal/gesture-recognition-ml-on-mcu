# STM32U5 Models

This directory contains the selected STM32U5 deployment artifacts rather than the full set of training checkpoints.

- `selected_deployable/`
  - Primary STM32U5 deployment model (`C1j_ptq_streaming_sm_int8.tflite`)
  - Clip-form INT8 evaluation model (`C1j_ptq_clip_int8.tflite`)
  - Backup deployment model (`C1f_ptq_streaming_sm_int8.tflite`)
  - Generated ST Edge AI C sources for the selected C1j INT8 network
  - Compact metadata describing why this model was selected

Large `.h5` checkpoints, SavedModels, and generated workspaces are intentionally excluded.
