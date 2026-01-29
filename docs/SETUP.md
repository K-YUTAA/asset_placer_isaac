# Setup Guide

## Prerequisites

1. **Isaac Sim / Kit SDK repo** checked out (this extension lives under `source/extensions`)
2. **NVIDIA GPU** with driver version 537.70+
3. **Nucleus Server** running and reachable
4. **OpenAI API Key** available (env var or settings file)

## Step 1: Clone or open the repo

```bash
git clone <repository-url> isaacsim
cd isaacsim
```

If you already have the repo, skip this step.

## Step 2: Build / fetch Kit dependencies

```powershell
# Windows
.\repo.bat build
```

```bash
# Linux
./repo.sh build
```

## Step 3: Install Python dependencies (if needed)

The extension uses `pipapi` (see `config/extension.toml`) to auto-install
packages. For offline or restricted environments, install manually:

```powershell
# Windows
.\_build\windows-x86_64\release\kit\python.bat -m pip install openai numpy pillow opencv-python
```

```bash
# Linux
./_build/linux-x86_64/release/kit/python.sh -m pip install openai numpy pillow opencv-python
```

## Step 4: Configure extension settings

Copy the example file and edit as needed:

```bash
cp source/extensions/my.research.asset_placer_isaac/my/research/asset_placer_isaac/extension_settings.json.example \
   source/extensions/my.research.asset_placer_isaac/my/research/asset_placer_isaac/extension_settings.json
```

Set:
- `openai_api_key` (optional if using `OPENAI_API_KEY`)
- `search_root_url`
- other paths as needed

## Step 5: Launch Kit / Isaac Sim

```powershell
.\repo.bat launch
```

Enable the extension in **Window > Extensions** by searching for
"USD Search Placer".

## Step 6: Verify entrypoint

`config/extension.toml` should contain:

```
[[python.module]]
name = "my.research.asset_placer_isaac.core.extension_app"
```

## Common commands

```powershell
.\repo.bat clean
.\repo.bat build
.\repo.bat launch
```
