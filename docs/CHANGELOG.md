# Changelog

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## Extension Change History (Chronological Order)

## [Unreleased]

- Refactored extension internals into `core/` mixins for maintainability
- Entry point now loads `core.extension_app` via `config/extension.toml`
- Documentation refreshed to match the new structure
- JSON workflow UI reorganized (inputs, AI generation, JSON files, placement)
- Added debug BBox placement button driven by loaded/selected JSON
- Split JSON loading from scene generation (separate "Select JSON" and "Generate Scene" actions)
- Unified wording from "Loaded" to "Selected" JSON across UI
- Removed tabbed UI; consolidated metadata/offset controls into the main view

## [2026-01-29]

### Added
- **Restoration of Generation Logs**: Re-implemented timestamped log files (`.log`) in the project root's `log/` folder.
  - Logs now include AI settings, token usage statistics, full step 1/2 texts, and AABB collision results.
- **Rotation Accuracy Improvements**: Enhanced `prompt_1.txt` and `prompt_2.txt` with unified rotation definitions and 'functional front' rules for various furniture categories.

### Fixed
- **UI Button State**: Fixed a bug where the "Generate JSON (AI)" button remained disabled after a single use or cancellation.
- **Token Usage Display**: Improved real-time token tracking in streaming mode to ensure accurate usage stats are displayed in the UI.
- **Import Error**: Fixed a `NameError` for `Any` and missing `traceback` import in `commands.py` that caused extension load failure.

## [2026-01-24]

### Added
- **4eb2ace** - Add outer polygon wall support
  - Added support for outer polygon walls
  - Updated `core/commands.py` and `core/constants.py`

## [2026-01-20]

### Fixed
- Fixed window glass material connection errors (`UsdShade` connection with correct types)
- Enhanced error handling (wall generation continues even if glass material generation fails)

### Changed
- Unified coordinate system (Right=+X / Left=-X)
- Fixed `rotationZ` definitions to match coordinate system
- Added X-axis inversion safety feature (automatic inversion only when "Left=+X" is detected)

### Added
- Suppressed automatic window generation rule (windows not explicitly shown in image are not generated)
- Layout auto-restore functionality (Quick Save/Quick Load support)
- Created detailed design document (`SYSTEM_DESIGN.md`)

**Work Report**: REPORT_20260120.md

## [2026-01-19]

### Changed
- **01c3de7** - Refactor extension and add search tools
  - Major refactoring of extension (split `extension.py` into `core/` modules)
  - Added search tool functionality
  - Updated documentation

## [2026-01-18]

### Added
- AI Advanced Settings (model selection, reasoning effort, verbosity, image detail, max tokens)
- Asset Orientation Offset improvements
- Preview functionality (image, dimensions, prompts, generated JSON)
- Asset blacklist functionality

### Fixed
- UI overflow issues
- Documentation consistency fixes

**Work Report**: REPORT_20260118.md

## [2026-01-12]

### Added
- **843f560** - Initial import of Isaac Sim extension
  - Initial project creation

## [2025-12-09]

### Changed
- **584062c** - Add up-axis handling and vector search tweaks
  - Added up-axis processing and vector search adjustments

## [2025-12-08]

### Added
- **91b6e06** - Add category hints and path filters for asset placement
  - Added category hints and path filters for asset placement

## [2025-11-23]

### Fixed
- **4ede0ae** - Fix vector search: increase timeout and filter USD files
  - Increased timeout for vector search and added USD file filtering

### Changed
- **31fdeee** - Replace keyword search with CLIP vector search (TDD)
  - Replaced keyword search with CLIP vector search (TDD)

### Added
- **ae708c6** - Implement recursive directory search for USD files (TDD)
  - Implemented recursive directory search for USD files (TDD)
- **f847414** - Add debug logging for directory search
  - Added debug logging for directory search
- **7ffd677** - Add automatic USD file search in directories
  - Added automatic USD file search in directories

### Fixed
- **396e823** - Fix: Restore normalized_root variable for semantic search
  - Restored normalized_root variable for semantic search

### Added
- **e94caa0** - Implement semantic search API integration (TDD)
  - Implemented semantic search API integration (TDD)

## [2025-11-08]

### Changed
- **9c17794** - Refactor procedural generation into separate modules
  - Refactored procedural generation into separate modules

### Added
- **021b98e** - Implement automatic floor and wall generation
  - Implemented automatic floor and wall generation functionality

## [2025-11-06]

### Changed
- **53ebdd6** - Revert cm to m unit conversion - keep JSON values as-is in meters
  - Reverted cm→m conversion, keeping JSON values in meters
- **e313290** - Ignore JSON Y coordinate and place all models with bottom at Y=0
  - Ignored JSON Y coordinate and placed all models with bottom at Y=0
- **99dcb58** - Fix unit conversion and ground alignment
  - Fixed unit conversion and ground alignment
- **7e19bc0** - Fix Y-coordinate handling: JSON Y represents asset center, not bottom surface
  - Fixed Y-coordinate handling: JSON Y represents asset center, not bottom surface
- **87ad25a** - Revert cm to m conversion - user feedback that scale became incorrect
  - Reverted cm→m conversion (scale became incorrect)
- **88dca3c** - Add centimeter to meter conversion for JSON coordinates
  - Added centimeter to meter conversion for JSON coordinates

### Fixed
- **96f93bb** - Fix test assertion for Y-offset ground alignment
  - Fixed test assertion for Y-offset ground alignment
- **55898bf** - Fix asset loading race condition with async BBox calculation
  - Fixed asset loading race condition with async BBox calculation
- **959852c** - Fix floating models by aligning all model bases to JSON Y coordinate
  - Fixed floating models by aligning all model bases to JSON Y coordinate

### Added
- **f55845b** - Implement rotation-aware axis mapping for absolute scaling
  - Implemented rotation-aware axis mapping for absolute scaling

## [2025-11-05]

### Fixed
- **3793628** - Fix USD Search to use single-word queries and add debug logging
  - Fixed USD Search to use single-word queries and added debug logging
- **18117ae** - Fix asset selection to use normalized search query
  - Fixed asset selection to use normalized search query

### Changed
- **70458a7** - Expand furniture name mapping for better USD Search results
  - Expanded furniture name mapping for better USD Search results
- **5073ce5** - Merge feature/absolute-scaling branch into main
  - Merged feature/absolute-scaling branch into main
- **5d8dddb** - Remove cm to meter conversion - JSON already in meters
  - Removed cm→m conversion (JSON already in meters)
- **9eaf4ba** - Fix xformOp order for correct transform application
  - Fixed xformOp order for correct transform application

### Added
- **99b3fd8** - Implement absolute scaling for asset placement
  - Implemented absolute scaling for asset placement
- **700109a** - Add tests for absolute scaling functionality
  - Added tests for absolute scaling functionality
- **4dd103d** - Merge feature/usd-search-mapping branch into main
  - Merged feature/usd-search-mapping branch into main

## [0.1.0] - 2025-11-02

- Initial version of basic python extension template
