# Dependencies

## Kit Extension Dependencies

Declared in `config/extension.toml`:

- `omni.client`
- `omni.kit.search.service`
- `omni.usd`

## Python Packages (pipapi)

Configured in `config/extension.toml` under `[python.pipapi]` and
auto-installed when the extension loads:

- `openai`
- `numpy>=1.21.2`
- `pillow`
- `opencv-python==4.10.0.84`

`backend/__init__.py` also attempts a lightweight auto-install via
`omni.kit.pipapi` as a fallback.

## Manual Installation (offline or restricted environments)

This extension uses the **Kit SDK's embedded Python environment**, not your
system Python.

### Windows (PowerShell)

```powershell
.\_build\windows-x86_64\release\kit\python.bat -m pip install openai numpy pillow opencv-python
```

### Linux

```bash
./_build/linux-x86_64/release/kit/python.sh -m pip install openai numpy pillow opencv-python
```

## Verification

After installation, verify packages are available:

```python
# In Omniverse Console (Window > Console)
import openai
import numpy
import PIL
import cv2
print("All packages imported successfully!")
```

## OpenAI API Key

Set your OpenAI API key as an environment variable or store it in
`extension_settings.json` (`openai_api_key`).

```bash
# Windows
set OPENAI_API_KEY=your-api-key-here

# Linux
export OPENAI_API_KEY=your-api-key-here
```

## Nucleus Server

- **Required**: Nucleus Server for USD Search functionality
- **Default**: `192.168.11.65` (configurable in the UI or settings file)
- Ensure the server is accessible from your machine

## NVIDIA Driver

- **Minimum**: Version 537.70 or higher
- Required for proper GPU acceleration and rendering
