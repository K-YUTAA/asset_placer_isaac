# Architecture

## System Overview

```
┌──────────────────────────────────────────────────────────────┐
│ Extension UI (core/extension_app.py)                          │
│  - Settings / State / UI / Handlers / Commands (mixins)        │
└───────────────┬───────────────────────────────────────────────┘
                │
        ┌───────┴────────┐
        │                │
        ▼                ▼
Backend AI (backend/ai_processing.py)   Placement (core/commands.py)
  - step1/step2/step3                   - USD Search + stage placement
        │                │
        ▼                ▼
     Layout JSON     Nucleus / USD Search
        │                │
        └──────────┬─────┘
                   ▼
             USD Stage (references)
```

## Workflow

### Method 1: Generate from Image (Tab 1)

1. **User Input**:
   - Floor plan image file
   - Dimensions text file
   - (Optional) Custom prompts

2. **AI Processing** (`backend/ai_processing.py`):
   - `step1_analyze_image()`: Analyze floor plan with OpenAI
   - Optional approval loop
   - `step2_generate_json()`: Generate structured layout JSON
   - `step3_check_collisions()`: Validate object placements

3. **Output**: Layout JSON with furniture positions

### Method 2: Load from File (Tab 2)

1. **User Input**:
   - Pre-existing layout JSON file

2. **Validation**: Basic JSON structure check

3. **Output**: Same layout JSON format

### Common Path: Asset Placement

4. **USD Search**:
   - For each object in `area_objects_list`
   - Query Nucleus Server with `object_name`
   - Retrieve USD asset path

5. **Stage Placement**:
   - Add asset as **USD Reference** (not copy)
   - Apply transformations: position, scale, rotation
   - Use coordinates from JSON (X, Z, Height, etc.)

## Project Structure

```
source/extensions/my.research.asset_placer_isaac/
├── config/
│   └── extension.toml          # Extension metadata + entrypoint
├── docs/
│   └── (documentation)
└── my/research/asset_placer_isaac/
    ├── __init__.py             # Re-exports MyExtension
    ├── extension.py            # Compatibility wrapper (not entrypoint)
    ├── core/
    │   ├── extension_app.py    # IExt entrypoint + mixins
    │   ├── commands.py         # USD Search + placement
    │   ├── handlers.py         # UI callbacks
    │   ├── settings.py         # JSON settings + file IO helpers
    │   ├── state.py            # Rotation offsets + metadata
    │   ├── ui.py               # UI construction helpers
    │   └── constants.py        # Prompts + model lists
    ├── backend/
    │   ├── __init__.py         # Re-exports backend APIs
    │   ├── ai_processing.py    # OpenAI logic
    │   └── file_utils.py       # Base64, file utils
    ├── procedural/
    │   ├── door_detector.py
    │   ├── floor_generator.py
    │   └── wall_generator.py
    └── tests/
        └── (test modules)
```

## Key Design Patterns

### Lazy Imports

All heavy dependencies imported **inside functions**, not at module level:

```python
def step1_analyze_image(...):
    from openai import OpenAI  # Import here
    import base64
    # ... function logic
```

**Reason**: Prevents startup failures if packages missing.

### Separation of Concerns

- **core/extension_app.py**: Entry point and UI wiring (`omni.ui`)
- **core/ui.py**: UI construction helpers
- **core/handlers.py**: UI callbacks and file pickers
- **core/commands.py**: USD Search + asset placement
- **core/settings.py**: Settings persistence and file IO
- **core/state.py**: Rotation offsets + prim metadata
- **backend/ai_processing.py**: OpenAI requests and layout JSON generation
- **procedural/*.py**: Floor/wall/door generators

### Non-Interactive Mode

`backend/ai_processing.py` supports both:
- **Interactive** (CLI): User approval loops, retry prompts
- **Non-Interactive** (GUI): Automatic approval, no `input()` calls

Controlled via `non_interactive=True` parameter.

## Data Flow

### JSON Schema

```json
{
  "area_name": "LivingRoom",
  "area_size_X": 5.0,
  "area_size_Y": 6.0,
  "room_polygon": [
    { "X": -2.5, "Y": -3.0 },
    { "X": 2.5, "Y": -3.0 },
    { "X": 2.5, "Y": 3.0 },
    { "X": -2.5, "Y": 3.0 }
  ],
  "windows": [
    { "X": 1.2, "Y": 3.0, "Width": 1.2, "Height": 1.0, "SillHeight": 0.9 }
  ],
  "area_objects_list": [
    {
      "object_name": "Sofa",
      "X": 1.2,
      "Y": 0.8,
      "Length": 2.0,
      "Width": 0.9,
      "Height": 0.8
    },
    {
      "object_name": "Coffee_Table",
      "X": 1.2,
      "Y": 1.6,
      "Length": 1.2,
      "Width": 0.6,
      "Height": 0.45
    }
  ]
}
```

### Coordinate System

- **X**: Horizontal (left-right)
- **Y**: Depth (front-back)
- **Z**: Height (up)
- **Units**: Meters (m)

USD stage is configured as **Z-Up** with **1.0 meters per unit** in
`core/commands.py` when placement begins.

## Extension Lifecycle

1. **Startup** (`core/extension_app.py` → `on_startup()`):
   - Create UI window
   - Initialize file pickers
   - Set up tab structure

2. **Runtime**:
   - User interactions trigger callbacks
   - Backend functions called asynchronously
   - USD commands modify stage

3. **Shutdown** (`core/extension_app.py` → `on_shutdown()`):
   - Destroy UI elements
   - Clean up file pickers
   - Release resources

## Dependencies

- **UI**: `omni.ui`, `omni.kit.window.filepicker`
- **USD**: `pxr` (Usd, UsdGeom, Gf, Sdf, Tf, Vt)
- **AI**: `openai`
- **Processing**: `numpy`, `PIL`, `cv2`

See [DEPENDENCIES.md](DEPENDENCIES.md) for installation.
