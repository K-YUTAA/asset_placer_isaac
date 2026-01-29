# API Reference

## Backend Functions (`backend/ai_processing.py`)

### `step1_analyze_image()` (async)

Analyzes a floor plan image and returns descriptive text plus stats.

**Signature**:
```python
async def step1_analyze_image(
    image_base64: str,
    prompt1_text: str,
    dimensions_text: str,
    model_name: str,
    api_key: Optional[str] = None,
    reasoning_effort: Optional[str] = None,
    max_output_tokens: Optional[int] = None,
    text_verbosity: Optional[str] = None,
    image_detail: Optional[str] = None,
    non_interactive: bool = True,
) -> Tuple[str, Dict[str, Any]]
```

**Parameters**:
- `image_base64` (str): Base64-encoded floor plan image
- `prompt1_text` (str): Analysis prompt template
- `dimensions_text` (str): Room and furniture dimensions
- `model_name` (str): OpenAI model name (e.g., `gpt-4o-mini`)
- `api_key` (str, optional): OpenAI API key (falls back to env if omitted)
- `reasoning_effort` (str, optional): `low|medium|high|xhigh`
- `max_output_tokens` (int, optional): Output token cap
- `text_verbosity` (str, optional): `low|medium|high`
- `image_detail` (str, optional): `low|high`
- `non_interactive` (bool): Reserved for CLI workflows

**Returns**:
- `(analysis_text, stats)` where `stats` includes timing and token usage.

**Example**:
```python
analysis, stats = await step1_analyze_image(
    img_b64, prompt1, dimensions, "gpt-4o-mini", api_key="sk-..."
)
```

---

### `step2_generate_json()` (async)

Generates structured layout JSON from analysis text and image input.

**Signature**:
```python
async def step2_generate_json(
    analysis_text: str,
    dimensions_text: str,
    image_base64: str,
    prompt2_base_text: str,
    model_name: str,
    api_key: Optional[str] = None,
    reasoning_effort: Optional[str] = None,
    max_output_tokens: Optional[int] = None,
    text_verbosity: Optional[str] = None,
    image_detail: Optional[str] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]
```

**Returns**:
- `(layout_json, stats)`

**Layout JSON (extended)**:
- `area_name` (str)
- `area_size_X` / `area_size_Y` (number, meters)
- `room_polygon` (list, optional): ordered XY points for the room outline, e.g.
  `[{ "X": 0.0, "Y": 0.0 }, { "X": 4.0, "Y": 0.0 }, ...]`
- `windows` (list, optional): window openings placed on walls, e.g.
  `[{ "X": 1.2, "Y": 2.0, "Width": 1.2, "Height": 1.0, "SillHeight": 0.9 }]`
- `area_objects_list` (list): furniture/door/etc objects for USD Search placement

**Example**:
```python
layout_json, stats = await step2_generate_json(
    analysis, dimensions, img_b64, prompt2, "gpt-4o-mini", api_key="sk-..."
)
```

---

### `step3_check_collisions()`

Detects collisions between objects using AABB.

**Signature**:
```python
def step3_check_collisions(layout: Dict[str, Any]) -> List[Tuple[str, str, float]]
```

---

## Backend Utilities (`backend/file_utils.py`)

### `encode_image_to_base64(image_path: str) -> str`

Reads an image file and returns a base64 string.

### `read_text_from_file(file_path: str) -> str`

Reads a UTF-8 text file and returns its contents.

### `build_timestamped_path(...) -> pathlib.Path`

Builds a timestamped output path, honoring directory or file inputs.

---

## Extension Entry Point (`core/extension_app.py`)

### `MyExtension` Class

Main extension class inheriting from `omni.ext.IExt`.

- `on_startup(ext_id: str)`: Initializes settings, UI, and state.
- `on_shutdown()`: Cancels tasks and disposes UI resources.

This class composes the following mixins:

- `SettingsMixin` (`core/settings.py`)
- `StateMixin` (`core/state.py`)
- `UIMixin` (`core/ui.py`)
- `HandlersMixin` (`core/handlers.py`)
- `CommandsMixin` (`core/commands.py`)

### `some_public_function(x: int) -> int`

Public helper function re-exported for compatibility.

---

## Key Methods by Mixin

### `core/commands.py`

- `_do_ai_generation()`
- `_start_asset_search(layout_data)`
- `_search_and_place_assets(layout_data)`
- `_apply_transform(prim, position, rotation, scale, ...)`

### `core/handlers.py`

- `_on_generate_json_click()`
- `_on_load_json_click()`
- `_on_select_image_click()`
- `_on_select_dims_click()`
- `_on_approve_click()` / `_on_reject_click()`

### `core/settings.py`

- `_load_settings_from_json()` / `_save_settings_to_json()`
- `_resolve_prompt_text(...)`
- `_get_json_output_dir()`

---

## Constants (`core/constants.py`)

- `DEFAULT_PROMPT1_TEXT`
- `DEFAULT_PROMPT2_TEXT`
- `MODEL_CHOICES`, `ADV_MODEL_CHOICES`
- `VECTOR_SEARCH_LIMIT`
