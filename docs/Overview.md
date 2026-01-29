# Overview

USD Search Placer for Isaac Sim generates room layouts from floor plan images
or existing JSON and places matching USD assets on the stage via USD Search.

## What it does

- Analyze a floor plan image with OpenAI and generate a layout JSON
- Load an existing layout JSON and run the same placement pipeline
- Search Nucleus assets and reference them into the stage
- Optionally generate floors, walls, and doors procedurally

## High-level flow

1. UI collects inputs (image/dimensions/prompts or JSON).
2. `backend/ai_processing.py` creates layout JSON (Step 1/2).
3. `core/commands.py` resolves assets and applies transforms to the stage.

## Entrypoint

The extension entrypoint is defined in `config/extension.toml`:

- `my.research.asset_placer_isaac.core.extension_app`

## Directory layout

- `config/extension.toml`: Extension manifest and entrypoint
- `my/research/asset_placer_isaac/core/`: UI, handlers, commands, settings, state
- `my/research/asset_placer_isaac/backend/`: OpenAI + file utilities
- `my/research/asset_placer_isaac/procedural/`: Floor/wall/door helpers
- `my/research/asset_placer_isaac/tests/`: Extension tests
- `docs/`: Documentation

## Related docs

- [ARCHITECTURE.md](ARCHITECTURE.md)
- [API_REFERENCE.md](API_REFERENCE.md)
- [SETUP.md](SETUP.md)
- [TROUBLESHOOTING.md](TROUBLESHOOTING.md)
- [KNOWN_ISSUES.md](KNOWN_ISSUES.md)