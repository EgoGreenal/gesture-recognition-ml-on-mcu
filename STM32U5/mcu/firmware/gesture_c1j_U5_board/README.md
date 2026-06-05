# STM32U5 CubeIDE firmware

This directory contains the STM32CubeIDE project used to deploy the C1j int8 gesture-recognition model on the STM32U5 board with X-CUBE-AI.

Included:

- STM32CubeIDE project metadata: `.project`, `.cproject`, `.mxproject`, and `gesture_c1j_U5_board.ioc`
- Application sources under `Core/`
- STM32 HAL/CMSIS dependencies under `Drivers/`
- X-CUBE-AI runtime headers and library under `Middlewares/ST/AI/`
- Generated model integration code under `X-CUBE-AI/`
- Linker scripts and host-side benchmark helpers

Excluded from version control:

- STM32CubeIDE build outputs under `Debug/` and `Release/`
- Local workspace metadata under `.metadata/`
- X-CUBE-AI local analysis cache under `.ai/`
- Binary and intermediate build artifacts such as `.elf`, `.map`, `.o`, `.d`, `.su`, and `.cyclo`
