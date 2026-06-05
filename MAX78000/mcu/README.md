# MCU Sources

This directory contains standalone MAX78000 firmware projects.

- `firmware/stable/`: default realtime gesture-recognition firmware.
- `firmware/early_stop/`: experimental early-exit firmware variant.

Both folders contain the project C sources, generated CNN files, generated weights header, Makefile, project configuration, and host demo helper scripts. They require an external MSDK installation at build time.

See `../docs/MCU_FIRMWARE.md`.
