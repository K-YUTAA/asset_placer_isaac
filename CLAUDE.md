# USD Search Placer Extension - Main Documentation

## Overview

AI-powered furniture placement extension for NVIDIA Omniverse Kit.

- **Input**: Floor plan image + dimensions via OpenAI GPT
- **Process**: Layout JSON generation + collision detection
- **Output**: USD asset search & automatic placement in Stage

## Quick Start

```bash
.\repo.bat build    # Build extension
.\repo.bat launch   # Start Omniverse
```

Extension Manager > USD Search Placer

## File Structure

```
my.research.asset_placer_isaac/
├── extension.py     # UI layer (tabs, file pickers, search integration)
├── backend/
│   ├── ai_processing.py  # AI logic (OpenAI, image analysis, collision detection)
│   └── file_utils.py
└── config/extension.toml
```

## Key Concepts

- **Lazy Imports**: All external packages imported inside functions
- **Omniverse Python**: Separate embedded environment (not system Python)
- **USD References**: Assets added as references, not duplicates
- **Two Input Methods**: AI generation (tab 1) or JSON file loading (tab 2)

## Critical Dependencies

See [@docs/DEPENDENCIES.md](docs/DEPENDENCIES.md) for package installation.

## Common Issues

See [@docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) for solutions.

## Architecture & API Details

- [@docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) - System design and workflow
- [@docs/API_REFERENCE.md](docs/API_REFERENCE.md) - Function specifications
- [@docs/SETUP.md](docs/SETUP.md) - Detailed setup instructions
- [@docs/KNOWN_ISSUES.md](docs/KNOWN_ISSUES.md) - Current limitations

---

# Omniverse Extension Project: Next Tasks and Branching Strategy

## 1. Role

You are a Senior Python Developer specializing in NVIDIA Omniverse Kit SDK.
You understand the current project architecture (`my.research.asset_placer_isaac`): `extension.py` handles UI, `backend/ai_processing.py` handles AI logic.
You are proficient with `omni.ui`, `omni.kit.app`, `asyncio`, and `UsdGeom`.

## 2. Current Status Summary

The current `main` branch is considered **stable (v1.0)** with the following features implemented:

* **UI Stability:** `Container::destroy` crashes are **completely fixed** using `_deferred_destroy_picker` and `next_update_async` (with fallback).
* **Async Processing:** "Generate" button uses `asyncio.ensure_future` to call AI logic in `backend/ai_processing.py` asynchronously, so **UI does not freeze**.
* **UI Features:**
    * Japanese font (`msgothic.ttc`) is registered, **mojibake is fixed**.
    * "Approval workflow" (approve/reject/resubmit popup) is **implemented**.
    * API key and Search Root URL are **persisted** via `carb.settings`.
* **Asset Placement (Incomplete):**
    * `_apply_transform` function only supports **Translate** (using JSON's `X`, `Z` coordinates) and **RotateY**.
    * `USD Search` only finds assets with **exact match** to JSON's `object_name`.

## 3. Git Branching Strategy (Required)

**Do NOT work on the `main` branch.**
For the two features to be implemented, create new branches from `main`:

1. **Search Logic:** `feature/usd-search-mapping`
2. **Scaling:** `feature/absolute-scaling`

## 4. Next Tasks

### Task 1: Improve Search Logic (Mapping)

* **Branch:** Create `feature/usd-search-mapping` from `main`.
* **Purpose:** Map AI-generated descriptive names (e.g., `Left_Side_Access_Bed`) to actual file names on Nucleus server (e.g., `Bed_01.usd`) to improve search hit rate.
* **Implementation Instructions (`extension.py` or `backend/ai_processing.py`):**

    1. **Implement `_normalize_furniture_name` function:**
        * Create a dictionary mapping AI-generated names to Nucleus asset names (or search queries).
        ```python
        # Example:
        NAME_MAPPING = {
            "left_side_access_bed": "bed",
            "accessible_master_bed": "bed",
            "tall_refrigerator": "refrigerator",
            "office_chair": "chair",
            "main_entrance_door": "door"
        }
        ```

    2. **Implement fallback logic:**
        * If no match in mapping dictionary, extract the last word from `object_name` (e.g., `Left_Side_Access_Bed` -> `bed`) as the search query.

    3. **Modify `_search_and_place_assets`:**
        * After getting `object_name` from JSON, call `_normalize_furniture_name` to generate `search_query`.
        * Pass this `search_query` to `client.async_search`, not the original `object_name`.

### Task 2: "Absolute Dimension" Scaling

* **Branch:** Create `feature/absolute-scaling` from `main`.
* **Purpose:** Apply JSON's `Length`, `Width`, `Height` as "absolute dimensions (in meters)" instead of "scale factors".
* **Implementation Instructions (`_apply_transform` function in `extension.py`):**

    1. **Import required:** `import pxr.Usd` (for `UsdGeom.BBoxCache`).

    2. Immediately after adding the asset as a **Reference**, get the Prim's **"original bounding box (BBox)"**:
        ```python
        stage = omni.usd.get_context().get_stage()
        bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
        prim_bbox = bbox_cache.ComputeWorldBound(prim).GetBox()
        original_size_vec = prim_bbox.GetRange().GetSize() # Gf.Vec3d(orig L, H, W)
        ```

    3. Get "target size" from JSON (`Length`, `Height`, `Width`) and **convert cm to m** (`* 0.01`).

    4. Calculate "scale factor to apply". **Watch for division by zero**:
        ```python
        # Map JSON axes (L=X, H=Y, W=Z) to USD axes (X, Y, Z)
        target_size_x = self._extract_float(data, "Length", 100.0) * 0.01 # L -> X
        target_size_y = self._extract_float(data, "Height", 100.0) * 0.01 # H -> Y
        target_size_z = self._extract_float(data, "Width", 100.0) * 0.01  # W -> Z

        scale_x = target_size_x / original_size_vec[0] if original_size_vec[0] != 0 else 1.0
        scale_y = target_size_y / original_size_vec[1] if original_size_vec[1] != 0 else 1.0
        scale_z = target_size_z / original_size_vec[2] if original_size_vec[2] != 0 else 1.0
        final_scale = Gf.Vec3f(float(scale_x), float(scale_y), float(scale_z))
        ```

    5. Modify `_apply_transform` to apply this `final_scale` using `xformable.AddScaleOp()`.
        * Keep existing `Translate` and `RotateY` logic, but also convert `Translate` values to meters (`* 0.01`).

## 5. Initial Instructions

First, create a new branch named `feature/usd-search-mapping` from `main`.
Then, open `extension.py` and start implementing **Task 1 (Improve Search Logic)**.
