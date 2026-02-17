# Known Issues

## Current Limitations

### 1. USD Search Matching Accuracy

**Status**: Partially implemented

**Description**:
- USD Search is implemented but relies on name matching
- Some assets may be missed due to naming differences
- Results depend on the Search Root URL and prompt object names

**Workaround**:
- Adjust Search Root URL in the UI
- Rename objects in the prompt or retry with synonyms

**Planned Fix**: Add name mapping rules / curated aliases

---

### 2. AI Processing Has No Cancel Button

**Status**: Known limitation

**Description**:
- AI calls are async but can take a long time
- There is no cancel button while processing

**Workaround**:
- Wait for completion or restart the extension if needed

**Planned Fix**: Add cancel support and progress indicators

---

### 3. No Undo Support

**Status**: By design (limitation of USD References)

**Description**:
- Placed assets cannot be undone via Ctrl+Z
- Must manually delete from Stage

**Workaround**:
- Save stage before running placement
- Use Layer system for non-destructive edits

---

### 4. OpenAI API Errors / Retries

**Status**: Intermittent

**Description**:
- OpenAI API may return 5xx errors and retry automatically
- End-to-end latency can increase when retries occur

**Workaround**:
- Retry the request or lower the model/verbosity settings
- Check console logs for retry behavior

**Planned Fix**: Add explicit retry/backoff settings and UI feedback

---

### 5. Collision Detection is 2D (ignores Y-axis gaps)

**Status**: Intentional simplification

**Description**:
- Objects at different heights may show false collisions
- Floor is excluded but other vertical stacking not handled

**Example**:
- Table lamp (on table) and table itself may "collide"

**Workaround**:
- Ignore small overlap volumes (e.g., <0.001 m³)
- Manually verify in 3D viewport

---

### 6. Limited Image Format Support

**Status**: OpenCV dependency

**Description**:
- Only JPEG, PNG, BMP tested
- SVG, PDF, TIFF may fail

**Workaround**:
- Convert to JPEG/PNG before processing
- Use standard image editors

---

### 7. Search Root URL Misconfiguration

**Status**: Configuration needed

**Description**:
- USD Search depends on the Search Root URL
- If the URL is wrong or lacks permissions, search returns no assets

**Workaround**:
- Update Search Root URL in the UI
- Or edit `extension_settings.json` in the extension root

**Planned Fix**: Add validation and connection test in the settings UI

---

### 8. No Multi-Room Support

**Status**: JSON schema limitation

**Description**:
- One JSON = one room
- Multi-room layouts require multiple JSON files

**Workaround**:
- Generate separate JSON per room
- Manually merge in USD stage

---

### 9. Rotation Offsets May Be Required

**Status**: Partial

**Description**:
- Some assets have different forward axes and need per-asset offsets
- Rotation offsets apply only when the asset reference or custom metadata is found

**Workaround**:
- Use Asset Orientation Offset and select the referenced prim
- Ensure assets are placed via this extension so metadata is stored
- Check `asset_rotation_offsets.json` in the extension root

**Planned Fix**: Improve reference discovery and batch offset application

---

### 10. Custom Prompt Files Are Optional

**Status**: Informational

**Description**:
- If prompt files are not provided, built-in defaults are used
- Custom prompt files override those defaults
- Defaults live in `core/constants.py`

**Workaround**:
- Provide `prompt_1.txt` / `prompt_2.txt` when you need custom behavior

**Planned Fix**: Add UI hints about prompt precedence

---

## Compatibility

### Windows Only (Partially)

**Status**: Linux untested

**Description**:
- File paths use Windows-style separators
- Bash scripts provided but not validated

**Workaround**:
- Test on Linux and report issues
- Use `pathlib.Path` for cross-platform paths

---

### Omniverse Kit 105.0+ Required

**Status**: Older versions untested

**Description**:
- API may differ in older Kit versions
- Python 3.10+ required

**Workaround**:
- Update to latest Kit SDK

---

## Security Considerations

### OpenAI API Key Exposure

**Risk**: Key logged in console output

**Mitigation**:
- Never commit logs to version control
- Use `.gitignore` for `logs/` directory

---

### Nucleus Server Access

**Risk**: No authentication check

**Mitigation**:
- Use Nucleus authentication features
- Restrict network access to trusted IPs

---

## Reporting Issues

If you encounter unlisted issues:

1. Check [TROUBLESHOOTING.md](TROUBLESHOOTING.md)
2. Review console output (Window > Console)
3. Check logs in `logs/` directory
4. Create issue with:
   - Error message
   - Steps to reproduce
   - JSON file (if applicable)
   - Omniverse version
