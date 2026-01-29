# USD Search Placer for Isaac Sim

AI-powered furniture placement extension - Generate 3D scenes from floor plan images

## Overview

USD Search Placer is an extension for NVIDIA Omniverse Isaac Sim that generates room layouts from 2D floor plan images and dimension information using OpenAI GPT, and automatically searches and places assets from Nucleus servers using USD Search (vector search). This is a research tool designed for experimental workflows.

## Key Features

### 🤖 AI-Driven Layout Generation
- **Image Analysis (Step1)**: Analyze floor plan images with OpenAI GPT to understand room structure and furniture placement
- **JSON Generation (Step2)**: Generate structured layout JSON from analysis results
- **Collision Detection (Step3)**: Detect collisions in generated layouts and report issues

### 🔍 USD Search Integration
- **Vector Search**: Search USD assets on Nucleus server using natural language queries
- **Automatic Placement**: Select optimal assets from search results and place as USD References
- **Blacklist Functionality**: Register unwanted assets to blacklist to prevent re-selection
- **Replacement Functionality**: Replace placed assets with different assets

### 🏗️ Procedural Generation
- **Floor Generation**: Automatically generate rectangular or polygon-shaped floors
- **Wall Generation**: Automatically generate walls along room perimeter (with opening support)
- **Window Glass Generation**: Automatically generate semi-transparent glass material for window openings

### ⚙️ Advanced Settings
- **AI Model Selection**: Select different models for Step1/Step2
- **Reasoning Effort Adjustment**: Choose from `low`, `medium`, `high`, `xhigh`
- **Detail Level Adjustment**: Set text verbosity and image detail individually
- **Token Limit**: Set maximum output token count

### 🎯 Coordinate System and Conversion
- **Coordinate System Unification**: Unified coordinate system (Right=+X / Left=-X)
- **Automatic Conversion**: Y-Up to Z-Up coordinate system conversion, cm to m unit conversion
- **Rotation Offset**: Save and batch apply rotation offsets per asset

### 💾 Layout Auto-Restore
- **Quick Layout Support**: Automatically load Quick Layout on startup
- **Settings Persistence**: Automatically save API keys, search root URL, file paths, etc.

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

## Setup

### 1. Copy Example Settings Files

```bash
cp my/research/asset_placer_isaac/extension_settings.json.example my/research/asset_placer_isaac/extension_settings.json
```

### 2. Edit Settings

Edit `extension_settings.json` with your actual values:
- `openai_api_key`: Your OpenAI API key (or use `OPENAI_API_KEY` environment variable)
- `search_root_url`: Your Omniverse Nucleus server asset root URL (must start with `omniverse://`)

**Important**: Never commit `extension_settings.json` with actual API keys!

### 3. Install Dependencies

The extension uses the following Python packages (automatically installed via `config/extension.toml`):
- `opencv-python==4.10.0.84`
- `openai`
- `pillow`
- `numpy>=1.21.2`

## Usage

### Method 1: Generate from Image (Tab 1)

1. **Select Input Files**:
   - Floor plan image (JPEG/PNG/BMP)
   - Dimensions text file
   - (Optional) Custom prompt files

2. **Adjust AI Settings**:
   - Select AI model
   - Adjust reasoning effort, detail levels, etc. in Advanced Settings

3. **Generate**:
   - Click "Generate JSON" button
   - Step1: Image analysis (with approval workflow)
   - Step2: JSON generation
   - Step3: Collision detection

4. **Place Assets**:
   - Preview generated JSON
   - Click "Place Assets" button to start placement

### Method 2: Direct Placement from JSON File (Tab 2)

1. **Select JSON File**:
   - Select existing layout JSON file

2. **Place Assets**:
   - Click "Place Assets" button to start immediate placement

## Project Structure

```
my.research.asset_placer_isaac/
├── config/
│   └── extension.toml          # Extension manifest
├── docs/                        # Documentation
│   ├── README.md               # This file
│   ├── Overview.md             # Overview
│   ├── ARCHITECTURE.md         # Architecture
│   ├── API_REFERENCE.md        # API Reference
│   ├── SYSTEM_DESIGN.md        # Detailed design document
│   ├── SETUP.md                # Setup guide
│   ├── TROUBLESHOOTING.md      # Troubleshooting
│   └── CHANGELOG.md            # Changelog
├── my/research/asset_placer_isaac/
│   ├── core/                   # Core functionality
│   │   ├── extension_app.py    # Entry point
│   │   ├── ui.py               # UI construction
│   │   ├── handlers.py         # UI event handlers
│   │   ├── commands.py         # Asset placement commands
│   │   ├── settings.py         # Settings management
│   │   ├── state.py            # State management
│   │   └── constants.py        # Constants and prompts
│   ├── backend/                 # Backend processing
│   │   ├── ai_processing.py    # OpenAI processing
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

## JSON Schema

Layout JSON structure:

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

## Key Feature Details

### USD Search Integration
- **Search API**: POST request to `http://192.168.11.65:30080/search`
- **Authentication**: Basic authentication (`omniverse:tsukuverse`)
- **Vector Search**: Search assets using natural language queries
- **Duplicate Prevention**: Prevent duplicates using blacklist and identity keys

### Asset Placement
- **Reference Placement**: Place as USD Reference (not copy)
- **Automatic Scaling**: Automatically scale based on JSON's `Length`, `Width`, `Height`
- **Rotation Processing**: Combine `rotationZ` with asset-specific rotation offsets
- **Coordinate Conversion**: Automatic conversion from Y-Up to Z-Up, cm to m

### Procedural Generation
- **Floor**: Supports rectangular or polygon shapes
- **Walls**: Automatically generate along perimeter edges, split at openings
- **Window Glass**: Semi-transparent material (opacity=0.2, ior=1.5)

## Troubleshooting

See [`TROUBLESHOOTING.md`](TROUBLESHOOTING.md) for detailed solutions.

### Common Issues

1. **USD Search returns no results**
   - Verify Search Root URL is correctly configured
   - Must start with `omniverse://`

2. **AI processing fails**
   - Verify OpenAI API key is correctly configured
   - Check network connection

3. **Assets are not placed correctly**
   - Check and adjust rotation offsets
   - Verify coordinate system is correct (Right=+X)

## Developer Information

### Entry Point

Defined in `config/extension.toml`:
- `my.research.asset_placer_isaac.core.extension_app`

This module defines `MyExtension` (the main `omni.ext.IExt`) and exposes `some_public_function`.

### Dependencies

- **Omniverse**: `omni.ext`, `omni.ui`, `omni.usd`, `omni.client`, `omni.kit.app`
- **USD**: `pxr` (Usd, UsdGeom, Sdf, Gf, UsdShade)
- **AI**: `openai` (AsyncOpenAI)
- **Other**: `requests`, `numpy`, `PIL`, `cv2`

### Tests

```bash
# Run tests
python -m pytest my/research/asset_placer_isaac/tests/
```

## Development Notes

- Internal mixins live under `my/research/asset_placer_isaac/core/`.
- Avoid adding new logic to the package root; keep it in `core/` and import from there.
- `extension.py` remains as a thin wrapper for external imports; the entrypoint is `core.extension_app`.

## License

NVIDIA Proprietary License

## Repository

[GitHub Repository](https://github.com/K-YUTAA/asset_placer_isaac.git)

## Related Documentation

- [Overview](Overview.md) - Feature overview
- [Architecture](ARCHITECTURE.md) - System design
- [API Reference](API_REFERENCE.md) - API specifications
- [System Design](SYSTEM_DESIGN.md) - Detailed design document
- [Setup Guide](SETUP.md) - Setup guide
- [Troubleshooting](TROUBLESHOOTING.md) - Troubleshooting
- [Changelog](CHANGELOG.md) - Changelog
