# STM32U5 Workflow Summary

This note condenses the STM32U5 path from `Project_Plan_ch.md` into the set of artifacts that matter for the public repository.

## End-to-End Flow

1. Teacher and TA preparation
   - A larger teacher and an assistant model provide soft labels for the student models.
   - The STM32U5 path uses the TAKD setup described in the project plan, with TensorFlow/Keras student models consuming exported soft labels.

2. Student architecture sweep
   - Multiple student variants were trained across the C1, C2, and C3 families.
   - The deployment-oriented path focused on compact INT8-ready models under STM32U5 memory and latency constraints.

3. Path selection
   - The streaming multi-I/O path was validated through the TSM probe and ST Edge AI analysis.
   - The compact path-selection summaries are in `../results/path_selection/`.
   - The probe write-up is in `tsm_probe_results.md`.

4. INT8 export and deployment packaging
   - The chosen primary deployment artifact is `C1j_ptq_streaming_sm_int8.tflite`.
   - A clip-form INT8 artifact is kept for host-side evaluation.
   - Generated ST Edge AI C sources for the selected model are stored under `../models/selected_deployable/stedgeai_generate_int8/`.

5. Early-exit evaluation
   - Early-exit sweeps were run on the selected deployment candidates.
   - Compact summary files are stored under `../results/early_exit/`.

## Selected STM32U5 Deployment Artifacts

- Primary model:
  - `../models/selected_deployable/C1j_ptq_streaming_sm_int8.tflite`
- Host evaluation model:
  - `../models/selected_deployable/C1j_ptq_clip_int8.tflite`
- Backup model:
  - `../models/selected_deployable/C1f_ptq_streaming_sm_int8.tflite`
- Generated C network package:
  - `../models/selected_deployable/stedgeai_generate_int8/`

## Why These Files Are In Git

These files are the smallest set that makes the STM32U5 path understandable and reusable:

- the final deployable model,
- the generated network sources needed for X-CUBE-AI integration,
- the result summaries used to justify model selection,
- and the board-side deployment notes.

Raw training checkpoints, dataset caches, and generated workspaces remain outside the repository because they are large and not needed to understand the final deployment path.
