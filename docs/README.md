# USD Search Placer for Isaac Sim [my.research.asset_placer_isaac]

Asset placement extension for Isaac Sim with AI-powered procedural generation capabilities.

## Setup

1. Copy the example settings files:
   ```bash
   cp my/research/asset_placer_isaac/extension_settings.json.example my/research/asset_placer_isaac/extension_settings.json
   cp .claude/settings.local.json.example .claude/settings.local.json
   ```

2. Edit `extension_settings.json` with your actual values:
   - `openai_api_key`: Your OpenAI API key
   - `search_root_url`: Your Omniverse asset server URL
   - Other paths as needed

**Important**: Never commit `extension_settings.json` or `.claude/settings.local.json` with actual API keys!

## Structure

- `config/extension.toml`: Extension manifest and python entrypoint.
- `my/research/asset_placer_isaac/core/`: Internal implementation modules.
- `my/research/asset_placer_isaac/backend/`: AI API helpers and IO utilities.
- `my/research/asset_placer_isaac/procedural/`: Floor/wall/door procedural generators.
- `my/research/asset_placer_isaac/tests/`: Tests.

## Entrypoint

The extension loads the module defined in `config/extension.toml`:

- `my.research.asset_placer_isaac.core.extension_app`

This module defines `MyExtension` (the main `omni.ext.IExt`) and exposes `some_public_function`.

## Development Notes

- Internal mixins live under `my/research/asset_placer_isaac/core/`.
- Avoid adding new logic to the package root; keep it in `core/` and import from there.

## Docs

- [Overview](Overview.md)
- [Architecture](ARCHITECTURE.md)
- [Setup](SETUP.md)
- [Troubleshooting](TROUBLESHOOTING.md)

## Compatibility

- `extension.py` remains as a thin wrapper for external imports; the entrypoint is `core.extension_app`.
