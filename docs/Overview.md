# Overview

USD Search Placer for Isaac Sim - AI-Powered Furniture Placement Extension

## System Overview

USD Search Placer is an extension for NVIDIA Omniverse Isaac Sim that **generates room layouts from 2D floor plan images and dimension information using OpenAI GPT, and automatically searches and places assets from Nucleus servers using USD Search (vector search)**. This is a research tool designed for experimental workflows.

### Primary Purpose

The extension focuses on enabling a complete experimental workflow: **"AI Analysis → JSON Generation → 3D Placement → Replacement/Adjustment"** for research purposes.

## System Workflow

### Method 1: Generate from Image (AI-Driven)

```
[Input]
  ↓
1. Floor plan image + Dimensions text + Prompts
  ↓
2. Step1: Image Analysis with OpenAI GPT
  ├─ Understand room structure
  ├─ Identify furniture types and positions
  └─ Output analysis results as text (with approval workflow)
  ↓
3. Step2: Generate Layout JSON from Analysis
  ├─ Output structured JSON format
  ├─ Define furniture position, size, and rotation
  └─ Automatic coordinate system conversion (Y-Up → Z-Up, cm → m)
  ↓
4. Step3: Collision Detection (Optional)
  ├─ Detect collisions using AABB (Axis-Aligned Bounding Box)
  └─ Report problematic placements
  ↓
5. Asset Search with USD Search
  ├─ Generate search queries from object_name for each furniture
  ├─ Execute vector search on Nucleus server
  └─ Select optimal USD assets
  ↓
6. Placement on USD Stage
  ├─ Place as USD Reference (not copy)
  ├─ Automatically apply position, scale, and rotation
  ├─ Procedurally generate floor, walls, and windows
  └─ Save metadata
  ↓
[Output: 3D Scene]
```

### Method 2: Direct Placement from JSON File

```
[Input: Existing Layout JSON]
  ↓
1. Load JSON file
  ↓
2. Validation (structure check)
  ↓
3. Asset Search with USD Search
  ↓
4. Placement on USD Stage
  ↓
[Output: 3D Scene]
```

## Key Features

### 🤖 AI-Driven Layout Generation

- **Step1: Image Analysis**
  - Analyze floor plan images with OpenAI GPT
  - Understand room structure and furniture placement
  - Output analysis results as text (with approval workflow support)

- **Step2: JSON Generation**
  - Generate structured layout JSON from analysis results
  - Define furniture position (X, Y), size (Length, Width, Height), and rotation (rotationZ)
  - Automatic coordinate system conversion (Y-Up → Z-Up, cm → m)

- **Step3: Collision Detection**
  - Detect collisions using AABB (Axis-Aligned Bounding Box)
  - Report problematic placements

### 🔍 USD Search Integration

- **Vector Search**
  - Search USD assets on Nucleus server using natural language queries
  - Search API: `http://192.168.11.65:30080/search` (Basic authentication)

- **Automatic Placement**
  - Select optimal assets from search results
  - Place as USD Reference (not copy)
  - Automatically apply position, scale, and rotation

- **Blacklist Functionality**
  - Register unwanted assets to blacklist
  - Excluded during re-search
  - Dual management with URL and identity keys

- **Replacement Functionality**
  - Replace placed assets with different assets
  - "Blacklist & Replace" and "Replace Only" modes

### 🏗️ Procedural Generation

- **Floor Generation**
  - Automatically generate rectangular or polygon-shaped floors
  - Uses `UsdGeom.Cube` or `UsdGeom.Mesh`

- **Wall Generation**
  - Automatically generate walls along room perimeter
  - Split wall segments at openings (doors, windows)
  - Wall thickness: Default 0.10m

- **Window Glass Generation**
  - Automatically generate semi-transparent glass material for window openings
  - Uses `UsdPreviewSurface` with opacity=0.2, ior=1.5

### ⚙️ Advanced Settings

- **AI Model Selection**
  - Select different models for Step1/Step2
  - GPT-4o-mini, GPT-4o, GPT-4o-reasoning, etc.

- **Reasoning Effort Adjustment**
  - Select from `low`, `medium`, `high`, `xhigh`

- **Detail Level Adjustment**
  - Text verbosity (`low`, `medium`, `high`)
  - Image detail (`low`, `high`)

- **Token Limit**
  - Set maximum output token count (default: 16000)

### 🎯 Coordinate System and Conversion

- **Coordinate System Unification**
  - Unified coordinate system: Right=+X / Left=-X
  - X-axis inversion safety feature

- **Automatic Conversion**
  - Automatic conversion from Y-Up to Z-Up coordinate system
  - Automatic conversion from cm to m units

- **Rotation Offset**
  - Save rotation offset per asset
  - Batch update Prims of the same asset

### 💾 Layout Auto-Restore

- **Quick Layout Support**
  - Automatically load Quick Layout on startup
  - Supports Quick Save/Quick Load

- **Settings Persistence**
  - Automatically save API keys, search root URL, file paths, etc.
  - Saved to `extension_settings.json`

## UI Structure

### Main Window

**Tab1: Generate from Image**
- Select image/dimensions/prompt files
- AI Model selection + Advanced Settings
- OpenAI API Key input
- Search Root URL configuration
- Search Tester button
- Asset Orientation Offset area
- Approval checkbox
- Generate JSON / Preview JSON buttons
- AI analysis results display + Approve/Reject buttons

**Tab2: Load from File**
- JSON file selection
- Search Root URL / Search Tester / Orientation Offset
- Place Assets button

### Sub Windows

- **AI Advanced Settings**: Step1/Step2 models, reasoning effort, verbosity, image detail, max tokens
- **Selected File Preview**: Preview of image/dimensions/prompts
- **Generated JSON Preview**: Read-only display of generated JSON
- **Blacklisted Assets**: View, delete, and clear blacklist
- **USD Search Tester**: Query input, search result thumbnail display, blacklist registration

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
    {
      "X": 1.2,
      "Y": 3.0,
      "Width": 1.2,
      "Height": 1.0,
      "SillHeight": 0.9
    }
  ],
  "area_objects_list": [
    {
      "object_name": "Sofa",
      "category": "Furniture",
      "search_prompt": "modern sofa",
      "X": 1.2,
      "Y": 0.8,
      "Length": 2.0,
      "Width": 0.9,
      "Height": 0.8,
      "rotationZ": 0
    }
  ]
}
```

### Coordinate System

- **X**: Horizontal direction (Right=+X, Left=-X)
- **Y**: Depth direction (Image upward direction=+Y)
- **Z**: Height direction (Up=+Z)
- **Units**: Meters (m)

## System Architecture

### Directory Structure

```
my.research.asset_placer_isaac/
├── config/
│   └── extension.toml          # Extension manifest and entrypoint
├── docs/                        # Documentation
├── my/research/asset_placer_isaac/
│   ├── core/                   # Core functionality
│   │   ├── extension_app.py    # IExt entrypoint
│   │   ├── ui.py               # UI construction
│   │   ├── handlers.py         # UI event handlers
│   │   ├── commands.py         # Asset placement commands
│   │   ├── settings.py         # Settings management
│   │   ├── state.py            # State management (rotation offsets, blacklist)
│   │   └── constants.py        # Constants, prompts, model lists
│   ├── backend/                 # Backend processing
│   │   ├── ai_processing.py    # OpenAI processing (Step1/Step2/Step3)
│   │   └── file_utils.py       # File utilities
│   ├── procedural/             # Procedural generation
│   │   ├── floor_generator.py  # Floor generation
│   │   ├── wall_generator.py   # Wall generation
│   │   └── door_detector.py   # Door detection
│   └── tests/                  # Tests
└── data/                       # Resources
    ├── icon.png
    └── preview.png
```

### Entry Point

Defined in `config/extension.toml`:
- `my.research.asset_placer_isaac.core.extension_app`

### Key Modules

- **`core/extension_app.py`**: Extension entry point, UI initialization
- **`core/commands.py`**: USD Search, asset placement, transform application
- **`backend/ai_processing.py`**: Communication with OpenAI API, image analysis, JSON generation
- **`procedural/wall_generator.py`**: Wall and window glass generation
- **`procedural/floor_generator.py`**: Floor generation

## Technology Stack

### Dependencies

- **Omniverse**: `omni.ext`, `omni.ui`, `omni.usd`, `omni.client`, `omni.kit.app`
- **USD**: `pxr` (Usd, UsdGeom, Sdf, Gf, UsdShade)
- **AI**: `openai` (AsyncOpenAI)
- **Image Processing**: `opencv-python`, `PIL` (Pillow)
- **Numerical Computing**: `numpy`

### External Services

- **OpenAI API**: Image analysis and JSON generation
- **USD Search API**: Asset search on Nucleus server
- **Omniverse Nucleus**: USD asset storage

## Configuration Files

### `extension_settings.json`

Saved items:
- `openai_api_key`: OpenAI API key
- `search_root_url`: Nucleus server asset root URL
- `image_path`, `dimensions_path`, `prompt1_path`, `prompt2_path`: File paths
- `json_output_dir`: JSON output directory
- `model_index`: Default model
- `ai_step1_model_index`, `ai_step2_model_index`: Step1/Step2 models
- `ai_reasoning_effort_index`, `ai_text_verbosity_index`, `ai_image_detail_index`: AI settings
- `ai_max_output_tokens`: Token limit
- `asset_blacklist`, `asset_blacklist_keys`: Blacklist

### `asset_rotation_offsets.json`

Saves rotation offset (degrees) per asset URL.

## Latest Feature Additions (2026-01-20)

### ✅ Stability Improvements
- Fixed window glass material connection errors
- Enhanced error handling

### ✅ Coordinate System Alignment
- Unified coordinate system (Right=+X / Left=-X)
- X-axis inversion safety feature

### ✅ Feature Enhancements
- Suppressed automatic window generation
- Layout auto-restore functionality

## Related Documentation

- [ARCHITECTURE.md](ARCHITECTURE.md) - System design and workflow
- [API_REFERENCE.md](API_REFERENCE.md) - API specifications
- [SYSTEM_DESIGN.md](SYSTEM_DESIGN.md) - Detailed design document
- [SETUP.md](SETUP.md) - Setup guide
- [TROUBLESHOOTING.md](TROUBLESHOOTING.md) - Troubleshooting
- [KNOWN_ISSUES.md](KNOWN_ISSUES.md) - Known issues
