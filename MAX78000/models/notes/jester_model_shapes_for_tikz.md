# 20BN-Jester Model Shapes for TikZ Diagrams

This document summarizes the model input/output sizes and layer shapes used in
the MAX78000/Jester experiments. It is written as a drawing reference for a
TikZ/LaTeX agent, not as training code.

## Shape Conventions

- 2D ai8x models use `NCHW`: `[batch, channels, height, width]`.
- The true 3D CNN probe uses `NCTHW`: `[batch, channels, time, height, width]`.
- All MAX78000-facing models use grayscale-derived planes, not RGB channels.
- All classifier outputs are logits of shape `[N, 27]`.
- On the board, the CNN accelerator computes the model body. Softmax and
  argmax are done in CPU software after logits are unloaded.
- MMAC counts Conv/Linear multiply-accumulates for one `64x64` inference.
  Pooling, activation, element-wise add, data movement, and softmax are not
  included.
- Parameter counts are checkpoint/report counts where available. The local
  PyTorch ai8x modules may include a few extra non-trainable control parameters.

## Input Encodings

| Encoding | Tensor shape | Plane meaning |
|---|---:|---|
| `raw8` | `[N, 8, 64, 64]` | 8 uniformly sampled grayscale frames |
| `raw12` | `[N, 12, 64, 64]` | 12 uniformly sampled grayscale frames |
| `motion8` | `[N, 8, 64, 64]` | plane 0 is grayscale anchor frame, planes 1-7 are absolute frame differences |
| `motion12` | `[N, 12, 64, 64]` | plane 0 is grayscale anchor frame, planes 1-11 are absolute frame differences |
| `signed_motion12` | `[N, 12, 64, 64]` | plane 0 is grayscale anchor frame, planes 1-11 are signed frame differences |
| `mixed12` | `[N, 12, 64, 64]` | 4 raw grayscale planes plus 8 absolute-motion planes |
| `true_3d_raw8` | `[N, 1, 8, 64, 64]` | raw grayscale frames represented as a temporal dimension |
| `true_3d_raw12` | `[N, 1, 12, 64, 64]` | raw grayscale frames represented as a temporal dimension |

## Model Summary Table

| Model / run family | Input encoding | Architecture template | Deployable on MAX78000 | Params | MMAC | Output |
|---|---|---|---:|---:|---:|---|
| TinyGesture software baseline | `raw8` | TinyGesture-2D | no | 196,091 | 16.65 | `[N, 27]` |
| JesterNet raw 8-frame QAT | `raw8` | JesterNet-2D | yes | 195,584 | 16.65 | `[N, 27]` |
| Jester12Wide raw 12-frame | `raw12` | BaseWide-2D | yes | 245,632 | 30.84 | `[N, 27]` |
| JesterMotion8 ablation | `motion8` | BaseWide-2D | yes | 244,768 | 27.30 | `[N, 27]` |
| JesterMotion12 baseline | `motion12` | BaseWide-2D | yes | 245,632 | 30.84 | `[N, 27]` |
| JesterMotion12 low-LR QAT | `motion12` | BaseWide-2D | yes | 245,632 | 30.84 | `[N, 27]` |
| JesterMixed12 | `mixed12` | BaseWide-2D | yes | 245,632 | 30.84 | `[N, 27]` |
| JesterSignedMotion12 | `signed_motion12` | BaseWide-2D | yes | 245,632 | 30.84 | `[N, 27]` |
| JesterMotion12 hard-class oversampling | `motion12` | BaseWide-2D | yes | 245,631 | 30.84 | `[N, 27]` |
| JesterMotion12 residual | `motion12` | Residual-2D | yes | 393,088 | 54.43 | `[N, 27]` |
| JesterMotion12 residual + signed motion | `signed_motion12` | Residual-2D | yes | 393,088 | 54.43 | `[N, 27]` |
| JesterMotion12 residual fc192 | `motion12` | ResidualFC192-2D | yes | 426,720 | 54.47 | `[N, 27]` |
| JesterMotion12 residual fc192 + signed motion | `signed_motion12` | ResidualFC192-2D | yes | 426,720 | 54.47 | `[N, 27]` |
| JesterMotion12 residual + extra conv | `motion12` | ResidualXConv-2D | yes | 429,952 | 56.79 | `[N, 27]` |
| JesterMotion12 residual + extra conv + signed motion | `signed_motion12` | ResidualXConv-2D | yes | 429,952 | 56.79 | `[N, 27]` |
| Big-teacher distilled student | `signed_motion12` | ResidualXConv-2D student | yes, if synthesis placement passes | 429,952 | 56.79 | `[N, 27]` |
| Huge-teacher distilled student / QAT sweep | `signed_motion12` | ResidualXConv-2D student | yes, if synthesis placement passes | 429,952 | 56.79 | `[N, 27]` |
| Big software teacher | `signed_motion12` | Teacher-Big-2D | no | 2,851,291 | 1,307.11 | `[N, 27]` |
| Huge software teacher | `signed_motion12` | Teacher-Huge-2D | no | 11,424,859 | 4,809.60 | `[N, 27]` |
| JesterLite3D raw 8-frame probe | `true_3d_raw8` | Lite-3D | no | 1,040,555 | 1,401.44 | `[N, 27]` |
| JesterLite3D raw 12-frame probe | `true_3d_raw12` | Lite-3D | no | 1,040,555 | 2,102.15 | `[N, 27]` |

## Template A: TinyGesture-2D Software Baseline

This was the early PyTorch software baseline. It is not an ai8x module, but it
uses the same high-level feature-map progression as JesterNet.

```text
Input raw8                         [N, 8, 64, 64]
ConvBlock 3x3, 8 -> 16             [N, 16, 64, 64]
MaxPool2d 2x2                      [N, 16, 32, 32]
ConvBlock 3x3, 16 -> 32            [N, 32, 32, 32]
MaxPool2d 2x2                      [N, 32, 16, 16]
ConvBlock 3x3, 32 -> 64            [N, 64, 16, 16]
MaxPool2d 2x2                      [N, 64, 8, 8]
ConvBlock 3x3, 64 -> 64            [N, 64, 8, 8]
MaxPool2d 2x2                      [N, 64, 4, 4]
Flatten                            [N, 1024]
Linear 1024 -> 128, ReLU           [N, 128]
Dropout
Linear 128 -> 27                   [N, 27]
```

## Template B: JesterNet-2D

Used by `JesterNet raw 8-frame ai8x QAT`.

```text
Input raw8                         [N, 8, 64, 64]
FusedConv2dReLU 3x3, 8 -> 16       [N, 16, 64, 64]
FusedMaxPoolConv2dReLU 2x2 + 3x3,
  16 -> 32                         [N, 32, 32, 32]
FusedMaxPoolConv2dReLU 2x2 + 3x3,
  32 -> 64                         [N, 64, 16, 16]
FusedMaxPoolConv2dReLU 2x2 + 3x3,
  64 -> 64                         [N, 64, 8, 8]
MaxPool2d 2x2                      [N, 64, 4, 4]
Flatten                            [N, 1024]
FusedLinearReLU 1024 -> 128        [N, 128]
Linear 128 -> 27                   [N, 27]
CPU softmax                        [N, 27]
```

## Template C: BaseWide-2D

Used by `Jester12Wide`, `JesterMotion8`, `JesterMotion12`, `JesterMixed12`,
`JesterSignedMotion12`, and the hard-class oversampling run. The only
architectural difference between 8-plane and 12-plane variants is the first
layer input channel count.

### 12-plane version

```text
Input C=12                         [N, 12, 64, 64]
FusedConv2dReLU 3x3, 12 -> 24      [N, 24, 64, 64]
FusedMaxPoolConv2dReLU 2x2 + 3x3,
  24 -> 48                         [N, 48, 32, 32]
FusedMaxPoolConv2dReLU 2x2 + 3x3,
  48 -> 64                         [N, 64, 16, 16]
FusedMaxPoolConv2dReLU 2x2 + 3x3,
  64 -> 64                         [N, 64, 8, 8]
MaxPool2d 2x2                      [N, 64, 4, 4]
Flatten                            [N, 1024]
FusedLinearReLU 1024 -> 160        [N, 160]
Linear 160 -> 27                   [N, 27]
CPU softmax                        [N, 27]
```

### 8-plane version

```text
Input C=8                          [N, 8, 64, 64]
FusedConv2dReLU 3x3, 8 -> 24       [N, 24, 64, 64]
Then same as 12-plane BaseWide-2D  [N, 27]
```

## Template D: Residual-2D

Used by `JesterMotion12 residual` and `JesterMotion12 residual + signed
motion`.

```text
Input C=12                         [N, 12, 64, 64]
FusedConv2dReLU 3x3, 12 -> 24      [N, 24, 64, 64]
FusedMaxPoolConv2dReLU 2x2 + 3x3,
  24 -> 48                         [N, 48, 32, 32]
FusedMaxPoolConv2dReLU 2x2 + 3x3,
  48 -> 64                         [N, 64, 16, 16]

Residual block 1 at 16x16:
  save residual                    [N, 64, 16, 16]
  FusedConv2dReLU 3x3, 64 -> 64    [N, 64, 16, 16]
  Conv2d 3x3, 64 -> 64             [N, 64, 16, 16]
  Add                              [N, 64, 16, 16]

FusedMaxPoolConv2dReLU 2x2 + 3x3,
  64 -> 64                         [N, 64, 8, 8]

Residual block 2 at 8x8:
  save residual                    [N, 64, 8, 8]
  FusedConv2dReLU 3x3, 64 -> 64    [N, 64, 8, 8]
  Conv2d 3x3, 64 -> 64             [N, 64, 8, 8]
  Add                              [N, 64, 8, 8]

MaxPool2d 2x2                      [N, 64, 4, 4]
Flatten                            [N, 1024]
FusedLinearReLU 1024 -> 160        [N, 160]
Linear 160 -> 27                   [N, 27]
CPU softmax                        [N, 27]
```

## Template E: ResidualFC192-2D

Used by `JesterMotion12 residual fc192` and signed-motion equivalent.

Same as Template D until flatten. The classifier changes:

```text
Flatten                            [N, 1024]
FusedLinearReLU 1024 -> 192        [N, 192]
Linear 192 -> 27                   [N, 27]
CPU softmax                        [N, 27]
```

## Template F: ResidualXConv-2D

Used by the best deployable student family: extra-conv residual model, signed
motion, and distilled students.

Same as Template D until `Residual block 2`, then one extra low-resolution
convolution is inserted before the final pool:

```text
After residual block 2             [N, 64, 8, 8]
FusedConv2dReLU 3x3, 64 -> 64      [N, 64, 8, 8]
MaxPool2d 2x2                      [N, 64, 4, 4]
Flatten                            [N, 1024]
FusedLinearReLU 1024 -> 160        [N, 160]
Linear 160 -> 27                   [N, 27]
CPU softmax                        [N, 27]
```

## Template G: Teacher-Big-2D

Software-only teacher used for distillation. Not MAX78000 deployable.

```text
Input signed_motion12              [N, 12, 64, 64]

Stage 1:
  ConvBNReLU 3x3, 12 -> 64         [N, 64, 64, 64]
  ResidualBlock x2, 64 -> 64       [N, 64, 64, 64]
  MaxPool2d 2x2                    [N, 64, 32, 32]

Stage 2:
  ConvBNReLU 3x3, 64 -> 96         [N, 96, 32, 32]
  ResidualBlock x2, 96 -> 96       [N, 96, 32, 32]
  MaxPool2d 2x2                    [N, 96, 16, 16]

Stage 3:
  ConvBNReLU 3x3, 96 -> 128        [N, 128, 16, 16]
  ResidualBlock x2, 128 -> 128     [N, 128, 16, 16]
  MaxPool2d 2x2                    [N, 128, 8, 8]

Stage 4:
  ConvBNReLU 3x3, 128 -> 192       [N, 192, 8, 8]
  ResidualBlock x2, 192 -> 192     [N, 192, 8, 8]
  AdaptiveAvgPool2d 1x1            [N, 192, 1, 1]

Flatten                            [N, 192]
Linear 192 -> 256, ReLU            [N, 256]
Linear 256 -> 27                   [N, 27]
Softmax for evaluation             [N, 27]
```

## Template H: Teacher-Huge-2D

Software-only teacher used for distillation. Not MAX78000 deployable.

```text
Input signed_motion12              [N, 12, 64, 64]

Stage 1:
  ConvBNReLU 3x3, 12 -> 96         [N, 96, 64, 64]
  ResidualBlock x3, 96 -> 96       [N, 96, 64, 64]
  MaxPool2d 2x2                    [N, 96, 32, 32]

Stage 2:
  ConvBNReLU 3x3, 96 -> 160        [N, 160, 32, 32]
  ResidualBlock x3, 160 -> 160     [N, 160, 32, 32]
  MaxPool2d 2x2                    [N, 160, 16, 16]

Stage 3:
  ConvBNReLU 3x3, 160 -> 224       [N, 224, 16, 16]
  ResidualBlock x3, 224 -> 224     [N, 224, 16, 16]
  MaxPool2d 2x2                    [N, 224, 8, 8]

Stage 4:
  ConvBNReLU 3x3, 224 -> 320       [N, 320, 8, 8]
  ResidualBlock x3, 320 -> 320     [N, 320, 8, 8]
  AdaptiveAvgPool2d 1x1            [N, 320, 1, 1]

Flatten                            [N, 320]
Linear 320 -> 512, ReLU            [N, 512]
Linear 512 -> 27                   [N, 27]
Softmax for evaluation             [N, 27]
```

## Template I: JesterLite3D Probe

True Conv3D software experiment. This is not directly supported by ai8x
synthesis for MAX78000 because the deployable operator set has Conv1d/Conv2d,
but no Conv3d hardware/YAML operation.

### 8-frame input

```text
Input raw video                    [N, 1, 8, 64, 64]
ConvBNReLU3D 3x3x3, 1 -> 16        [N, 16, 8, 64, 64]
ResidualBlock3D x1, 16 -> 16       [N, 16, 8, 64, 64]
MaxPool3d (1,2,2)                  [N, 16, 8, 32, 32]
ConvBNReLU3D 3x3x3, 16 -> 32       [N, 32, 8, 32, 32]
ResidualBlock3D x1, 32 -> 32       [N, 32, 8, 32, 32]
MaxPool3d (2,2,2)                  [N, 32, 4, 16, 16]
ConvBNReLU3D 3x3x3, 32 -> 64       [N, 64, 4, 16, 16]
ResidualBlock3D x1, 64 -> 64       [N, 64, 4, 16, 16]
MaxPool3d (2,2,2)                  [N, 64, 2, 8, 8]
ConvBNReLU3D 3x3x3, 64 -> 96       [N, 96, 2, 8, 8]
ResidualBlock3D x1, 96 -> 96       [N, 96, 2, 8, 8]
AdaptiveAvgPool3d 1x1x1            [N, 96, 1, 1, 1]
Flatten                            [N, 96]
Linear 96 -> 128, ReLU             [N, 128]
Linear 128 -> 27                   [N, 27]
```

### 12-frame input

Same architecture, but temporal sizes are larger:

```text
Input raw video                    [N, 1, 12, 64, 64]
ConvBNReLU3D                       [N, 16, 12, 64, 64]
ResidualBlock3D                    [N, 16, 12, 64, 64]
MaxPool3d (1,2,2)                  [N, 16, 12, 32, 32]
ConvBNReLU3D                       [N, 32, 12, 32, 32]
ResidualBlock3D                    [N, 32, 12, 32, 32]
MaxPool3d (2,2,2)                  [N, 32, 6, 16, 16]
ConvBNReLU3D                       [N, 64, 6, 16, 16]
ResidualBlock3D                    [N, 64, 6, 16, 16]
MaxPool3d (2,2,2)                  [N, 64, 3, 8, 8]
ConvBNReLU3D                       [N, 96, 3, 8, 8]
ResidualBlock3D                    [N, 96, 3, 8, 8]
AdaptiveAvgPool3d 1x1x1            [N, 96, 1, 1, 1]
Flatten                            [N, 96]
Linear 96 -> 128, ReLU             [N, 128]
Linear 128 -> 27                   [N, 27]
```

## TikZ Drawing Notes

- For ai8x models, draw `FusedMaxPoolConv2dReLU` as a pool block followed by a
  convolution block if you want the data-size transition to be explicit.
- Residual additions occur at identical tensor shapes:
  - residual block 1: `[N, 64, 16, 16]`
  - residual block 2: `[N, 64, 8, 8]`
- The `signed_motion12`, `motion12`, `mixed12`, and `raw12` variants can share
  the same architecture drawing. Only the input plane semantics differ.
- Distilled students use the same `ResidualXConv-2D` diagram as the signed
  residual extra-conv student. Distillation changes training, not inference
  topology.
- Teacher models and true 3D CNN probes should be drawn separately from
  deployable MAX78000 models.
