# ai8x Quantization and Deployment Flow

The deployment path converts a PyTorch/QAT model into MAX78000 CNN accelerator C sources.

## Inputs

Required inputs for synthesis:

- ai8x-training model definition, for example `jestermotion12res.py`.
- Matching network YAML for ai8x-synthesis.
- QAT checkpoint.
- Sample input tensor with the same signed-motion 12x64x64 layout.
- Class labels for host-side display and firmware strings.

The selected model definition and policy are in:

```text
models/best_deployable/model_definition/
```

The broader set of reusable ai8x files is organized as:

```text
training/ai8x_custom_files/ai8x_training_models/
training/ai8x_custom_files/ai8x_training_datasets/
training/ai8x_custom_files/qat_policies/
training/ai8x_custom_files/synthesis_networks/
```

## Quantization semantics

The deployed CNN uses signed 8-bit quantization:

- Weights are signed int8.
- Input and intermediate activations are signed 8-bit tensors.
- C code may move bytes through `uint8_t` or packed `uint32_t` buffers, but those bytes represent signed 8-bit values when consumed by the CNN accelerator.
- CNN outputs are read as integer logits and then post-processed on the CPU with softmax/argmax.

## Synthesis

The generated firmware files are produced by `ai8xize.py`. The stable firmware currently includes:

```text
mcu/firmware/stable/cnn.c
mcu/firmware/stable/cnn.h
mcu/firmware/stable/weights.h
mcu/firmware/stable/softmax.c
```

The generated header comments in `cnn.c` record the original ai8xize command, checkpoint name, network YAML, and layer schedule.

## Regenerating sources

To regenerate the CNN sources:

1. Install ai8x-training and ai8x-synthesis outside this source package.
2. Copy the custom dataset, model, and policy files from `training/ai8x_custom_files/codex_uploads/`.
3. Run QAT or copy the selected QAT checkpoint into the ai8x-synthesis `trained/` directory.
4. Generate a matching sample input `.npy`.
5. Run `ai8xize.py` with `--device MAX78000`, `--mexpress`, `--compact-data`, and `--softmax`.
6. Copy the generated `cnn.c`, `cnn.h`, `weights.h`, and `softmax.c` into `mcu/firmware/stable/`.

Large checkpoints and `.npy` samples are deliberately excluded from this source package.
