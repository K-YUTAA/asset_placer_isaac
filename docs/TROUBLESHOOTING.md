# Troubleshooting

## Installation Issues

### Packages not found: `ModuleNotFoundError: No module named 'openai'`

**Cause**: Packages installed in wrong Python environment.

**Solution**:
```powershell
# Use Kit's Python, not system Python
.\_build\windows-x86_64\release\kit\python.bat -m pip install openai numpy pillow opencv-python
```

Verify installation:
```python
# In Omniverse Console
import openai
print(openai.__version__)
```

---

### Build fails: `Extension not found`

**Cause**: Extension path not included in Kit's extension search path.

**Solution**:
1. Ensure the extension lives under `source/extensions/my.research.asset_placer_isaac`.
2. If you moved it, add its folder to the extension search path
   (e.g., repo config or `--ext-folder`).
3. Rebuild:
   ```bash
   .\repo.bat clean
   .\repo.bat build
   ```

---

## Runtime Issues

### Extension window doesn't open

**Solution**:
1. Open Extension Manager (Window > Extensions)
2. Search "USD Search Placer"
3. Check **Autoload** checkbox
4. Click **Enable**

If error appears, check Console (Window > Console) for details.

---

### OpenAI API errors: `AuthenticationError`

**Cause**: API key not set or invalid.

**Solution**:
```bash
# Windows (PowerShell)
$env:OPENAI_API_KEY="sk-..."

# Windows (Command Prompt)
set OPENAI_API_KEY=sk-...

# Linux
export OPENAI_API_KEY=sk-...
```

Verify:
```python
import os
print(os.getenv("OPENAI_API_KEY"))
```

---

### USD Search returns no results

**Cause**: Nucleus Server connection failed or asset not found.

**Solution**:
1. **Check Nucleus Server**:
   - Verify IP address (default: `192.168.11.65`)
   - Test connection: `ping 192.168.11.65`
   - Open Nucleus Navigator (Window > Content)

2. **Check Asset Names**:
   - Asset names in JSON must match Nucleus file names
   - Example: `"Sofa"` → `Sofa.usd` or `Sofa.usdz`
   - Use USD Search UI to test queries manually

3. **Update JSON**:
   - Use exact asset names from Nucleus
   - Check capitalization (case-sensitive)

---

### Collision warnings but objects look fine

**Cause**: Collision detection is very precise; minor overlaps are flagged.

**Solution**:
- Small overlaps (e.g., <0.001 m³) are usually acceptable
- Adjust object positions in JSON if needed
- Ignore "Floor" collisions (floors excluded by default)

---

### Generated JSON has wrong coordinates

**Cause**: GPT misinterpreted image or dimensions.

**Solution**:
1. **Improve dimensions.txt format** (meters):
   ```
   Room: 5.0m x 6.0m
   Sofa: 2.0m x 0.9m x 0.8m (L x W x H)
   Table: 1.2m x 0.6m x 0.45m
   ```

2. **Edit prompt1.txt** to emphasize:
   - Coordinate origin (center or corner)
   - Units (always meters)
   - Orientation (X/Z axes)

3. **Manual correction**:
   - Edit generated JSON directly
   - Reload via "Load from File" tab

---

### High GPU usage / slow performance

**Cause**: Omniverse rendering loop running continuously.

**Solution**:
1. Reduce viewport quality (Settings > Rendering)
2. Pause rendering when not needed
3. Close unused extensions

---

## Debug Strategies

### Enable Debug Logging

```python
# In core/extension_app.py or core/commands.py
import omni.log
omni.log.set_level(omni.log.LEVEL_DEBUG)
```

### Check Python Environment

```bash
# List installed packages
.\_build\windows-x86_64\release\kit\python.bat -m pip list
```

### Inspect Generated Files

- JSON outputs: `json/` under the project root (or extension root fallback)
- Rotation offsets: `asset_rotation_offsets.json` in the extension root
- Settings: `extension_settings.json` in the extension root
- Logs: `logs/` directory (if `--log_dir` specified)

### Test Backend Independently

```python
# In Omniverse Console
from my.research.asset_placer_isaac import backend
help(backend.step1_analyze_image)
```

---

## Common Error Messages

### `TypeError: 'NoneType' object is not subscriptable`

**Location**: `core/commands.py`, `_start_asset_search()`

**Cause**: `layout_data` is None or missing expected keys.

**Solution**:
- Validate JSON structure before passing
- Check `area_objects_list` exists and is non-empty

---

### `FileNotFoundError: [Errno 2] No such file or directory: 'prompt_1.txt'`

**Cause**: Prompt files not found in expected location.

**Solution**:
- Place prompt files in extension directory
- Use absolute paths in file picker
- Or implement default prompts in code

---

### `cv2.error: OpenCV(4.x.x) (...) error: (-215:Assertion failed)`

**Cause**: Invalid image format or corrupted file.

**Solution**:
- Verify image is valid JPEG/PNG
- Check image dimensions (>0 pixels)
- Try re-exporting image from source

---

## Getting Help

1. **Check logs**: `logs/{timestamp}.log`
2. **Console output**: Window > Console
3. **Extension Manager**: Check for dependency conflicts
4. **Documentation**: Re-read [ARCHITECTURE.md](ARCHITECTURE.md) and [API_REFERENCE.md](API_REFERENCE.md)
