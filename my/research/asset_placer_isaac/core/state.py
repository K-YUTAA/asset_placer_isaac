# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import asyncio
import json
import math
import os
from typing import Dict, List, Optional, Tuple

import omni.log
from pxr import Gf, Usd, UsdGeom


class StateMixin:
    @staticmethod
    def _normalize_size_mode(value: object, default: str = "world") -> str:
        mode = str(value or default).strip().lower()
        if mode not in ("world", "local"):
            return default
        return mode

    @staticmethod
    def _safe_float(value: object, default: float = 0.0) -> float:
        try:
            if value is None:
                return float(default)
            return float(value)
        except (TypeError, ValueError):
            return float(default)

    def _load_rotation_offsets(self) -> Dict[str, int]:
        if os.path.exists(self._rotation_offsets_file):
            try:
                with open(self._rotation_offsets_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    return {str(k): int(v) for k, v in data.items()}
            except Exception as exc:
                omni.log.warn(f"Failed to load rotation offsets: {exc}")
        return {}

    def _save_rotation_offsets(self) -> None:
        try:
            with open(self._rotation_offsets_file, "w", encoding="utf-8") as f:
                json.dump(self._asset_rotation_offsets, f, indent=4)
        except Exception as exc:
            omni.log.warn(f"Failed to save rotation offsets: {exc}")

    def _get_asset_rotation_offset(self, asset_url: Optional[str]) -> int:
        if not asset_url:
            return 0
        value = self._asset_rotation_offsets.get(asset_url, 0)
        try:
            return int(value) % 360
        except (ValueError, TypeError):
            return 0

    def _parse_rotation_offset(self, value: str) -> int:
        try:
            offset = int(float(value))
        except (ValueError, TypeError):
            return 0
        offset %= 360
        if offset < 0:
            offset += 360
        return offset

    def _normalize_asset_url(self, asset_url: str) -> str:
        if not asset_url:
            return ""
        normalized = asset_url.strip()
        if normalized.startswith("http://"):
            normalized = normalized.replace("http://", "omniverse://")
        if normalized.startswith("/"):
            normalized = f"omniverse://192.168.11.65{normalized}"
        return normalized

    @staticmethod
    def _split_asset_filename(asset_url: str) -> Tuple[str, str]:
        if not asset_url:
            return "", ""
        basename = asset_url.split("?")[0].split("/")[-1]
        if "." in basename:
            name, ext = basename.rsplit(".", 1)
        else:
            name, ext = basename, ""
        return name.lower(), ext.lower()

    def _build_asset_identity_key(self, asset_url: str, size_bytes: Optional[int]) -> Optional[str]:
        if not asset_url:
            return None
        name, ext = self._split_asset_filename(asset_url)
        if not name and not ext:
            return None
        size_token = str(size_bytes) if isinstance(size_bytes, int) else "unknown"
        if ext:
            return f"{name}.{ext}|{size_token}"
        return f"{name}|{size_token}"

    def _is_asset_blacklisted(self, asset_url: str) -> bool:
        if not asset_url or not hasattr(self, "_asset_blacklist"):
            return False
        normalized = self._normalize_asset_url(asset_url)
        return normalized in self._asset_blacklist

    def _add_asset_to_blacklist(self, asset_url: str) -> bool:
        if not asset_url or not hasattr(self, "_asset_blacklist"):
            return False
        normalized = self._normalize_asset_url(asset_url)
        if not normalized:
            return False
        if normalized in self._asset_blacklist:
            return False
        self._asset_blacklist.add(normalized)
        return True

    def _add_asset_key_to_blacklist(self, identity_key: Optional[str]) -> bool:
        if not identity_key or not hasattr(self, "_asset_blacklist_keys"):
            return False
        if identity_key in self._asset_blacklist_keys:
            return False
        self._asset_blacklist_keys.add(identity_key)
        return True

    def _remove_asset_from_blacklist(self, asset_url: str) -> bool:
        if not asset_url or not hasattr(self, "_asset_blacklist"):
            return False
        normalized = self._normalize_asset_url(asset_url)
        if normalized in self._asset_blacklist:
            self._asset_blacklist.remove(normalized)
            return True
        return False

    def _remove_asset_key_from_blacklist(self, identity_key: Optional[str]) -> bool:
        if not identity_key or not hasattr(self, "_asset_blacklist_keys"):
            return False
        if identity_key in self._asset_blacklist_keys:
            self._asset_blacklist_keys.remove(identity_key)
            return True
        return False

    def _clear_asset_blacklist(self) -> bool:
        if not hasattr(self, "_asset_blacklist"):
            return False
        if not self._asset_blacklist:
            return False
        self._asset_blacklist.clear()
        return True

    def _clear_asset_blacklist_keys(self) -> bool:
        if not hasattr(self, "_asset_blacklist_keys"):
            return False
        if not self._asset_blacklist_keys:
            return False
        self._asset_blacklist_keys.clear()
        return True

    def _get_reference_items_from_prim(self, prim) -> List[object]:
        # Prim metadata (preferred)
        try:
            refs = prim.GetMetadata("references")
            if refs:
                if hasattr(refs, "GetAddedOrExplicitItems"):
                    items = refs.GetAddedOrExplicitItems()
                    if items:
                        return list(items)
                elif isinstance(refs, (list, tuple)):
                    return list(refs)
        except Exception as exc:
            omni.log.warn(f"Failed to read references metadata from prim '{prim.GetPath()}': {exc}")

        # Prim stack fallback
        try:
            for prim_spec in prim.GetPrimStack():
                ref_list = prim_spec.referenceList
                items = ref_list.GetAddedOrExplicitItems()
                if items:
                    return list(items)
        except Exception as exc:
            omni.log.warn(f"Failed to read references from prim stack '{prim.GetPath()}': {exc}")

        return []

    def _get_reference_prim(self, prim):
        current = prim
        while current and current.IsValid():
            try:
                custom = current.GetCustomData()
                if custom.get("asset_placer.asset_url"):
                    return current
            except Exception:
                pass

            ref_items = self._get_reference_items_from_prim(current)
            for item in ref_items:
                asset_path = getattr(item, "assetPath", None)
                if asset_path:
                    return current

            current = current.GetParent()
        return prim

    def _get_asset_url_from_prim(self, prim) -> Optional[str]:
        # Walk up the prim hierarchy until we find a reference or custom data.
        current = prim
        while current and current.IsValid():
            # Custom data (if placed by this extension)
            try:
                custom = current.GetCustomData()
                asset_url = custom.get("asset_placer.asset_url")
                if asset_url:
                    return str(asset_url)
            except Exception as exc:
                omni.log.warn(f"Failed to read custom data from prim '{current.GetPath()}': {exc}")

            # References metadata
            ref_items = self._get_reference_items_from_prim(current)
            for item in ref_items:
                asset_path = getattr(item, "assetPath", None)
                if asset_path:
                    return str(asset_path)

            current = current.GetParent()

        return None

    def _get_search_metadata_from_prim(self, prim) -> Dict[str, str]:
        try:
            custom = prim.GetCustomData()
        except Exception:
            custom = {}

        object_name = custom.get("asset_placer.object_name") or prim.GetName()
        category = custom.get("asset_placer.category") or ""
        search_prompt = custom.get("asset_placer.search_prompt") or ""
        search_query = custom.get("asset_placer.search_query") or ""

        return {
            "object_name": str(object_name) if object_name else "",
            "category": str(category) if category else "",
            "search_prompt": str(search_prompt) if search_prompt else "",
            "search_query": str(search_query) if search_query else "",
        }

    def _build_data_from_prim_metadata(self, prim) -> Optional[Dict[str, object]]:
        try:
            custom = prim.GetCustomData()
        except Exception as exc:
            omni.log.warn(f"Failed to read custom data from '{prim.GetPath()}': {exc}")
            return None

        length = custom.get("asset_placer.length")
        width = custom.get("asset_placer.width")
        height = custom.get("asset_placer.height")
        rotation = custom.get("asset_placer.rotationZ", 0.0)
        x = custom.get("asset_placer.x", 0.0)
        y = custom.get("asset_placer.y", 0.0)
        object_name = custom.get("asset_placer.object_name") or prim.GetName()
        category = custom.get("asset_placer.category") or ""
        search_prompt = custom.get("asset_placer.search_prompt") or ""
        search_query = custom.get("asset_placer.search_query") or ""
        custom_size_mode = custom.get("asset_placer.size_mode")

        if length is None or width is None or height is None:
            omni.log.warn(f"Missing placement metadata for '{prim.GetPath()}'.")
            return None

        xform_data = self._extract_pose_from_xform_ops(
            prim,
            preferred_size_mode=str(custom_size_mode) if custom_size_mode else None,
            fallback_x=self._safe_float(x, 0.0),
            fallback_y=self._safe_float(y, 0.0),
            fallback_rotation=self._safe_float(rotation, 0.0),
        )
        resolved_size_mode = xform_data["size_mode"]

        return {
            "Length": float(length),
            "Width": float(width),
            "Height": float(height),
            "rotationZ": float(xform_data["rotationZ"]),
            "X": float(xform_data["X"]),
            "Y": float(xform_data["Y"]),
            "size_mode": resolved_size_mode,
            "object_name": str(object_name) if object_name else "",
            "category": str(category) if category else "",
            "search_prompt": str(search_prompt) if search_prompt else "",
            "search_query": str(search_query) if search_query else "",
        }

    def _store_placement_metadata(
        self,
        prim,
        asset_url: Optional[str],
        base_rotation: float,
        length: float,
        width: float,
        height: float,
        x: float,
        y: float,
        object_name: Optional[str] = None,
        category: Optional[str] = None,
        search_prompt: Optional[str] = None,
        search_query: Optional[str] = None,
        size_mode: Optional[str] = None,
    ) -> None:
        try:
            prim.SetCustomDataByKey("asset_placer.asset_url", asset_url or "")
            prim.SetCustomDataByKey("asset_placer.rotationZ", float(base_rotation))
            prim.SetCustomDataByKey("asset_placer.length", float(length))
            prim.SetCustomDataByKey("asset_placer.width", float(width))
            prim.SetCustomDataByKey("asset_placer.height", float(height))
            prim.SetCustomDataByKey("asset_placer.x", float(x))
            prim.SetCustomDataByKey("asset_placer.y", float(y))
            if object_name is not None:
                prim.SetCustomDataByKey("asset_placer.object_name", str(object_name))
            if category is not None:
                prim.SetCustomDataByKey("asset_placer.category", str(category))
            if search_prompt is not None:
                prim.SetCustomDataByKey("asset_placer.search_prompt", str(search_prompt))
            if search_query is not None:
                prim.SetCustomDataByKey("asset_placer.search_query", str(search_query))
            if size_mode is not None:
                prim.SetCustomDataByKey("asset_placer.size_mode", self._normalize_size_mode(size_mode))
        except Exception as exc:
            omni.log.warn(f"Failed to store placement metadata for '{prim.GetPath()}': {exc}")

    def _collect_xform_op_values(self, prim) -> Dict[str, object]:
        values: Dict[str, object] = {}
        try:
            xformable = UsdGeom.Xformable(prim)
            for op in xformable.GetOrderedXformOps():
                try:
                    name = str(op.GetOpName())
                    values[name] = op.Get()
                except Exception:
                    continue
        except Exception as exc:
            omni.log.warn(f"Failed to collect xform ops from '{prim.GetPath()}': {exc}")
        return values

    def _infer_size_mode_from_xform_ops(self, op_values: Dict[str, object]) -> str:
        if any(
            name in op_values
            for name in (
                "xformOp:translate:world",
                "xformOp:rotateZ:world",
                "xformOp:scale:world",
                "xformOp:rotateZ:offset",
            )
        ):
            return "local"
        # Local-mode affine-collapsed stack (translate:world + transform:affine)
        if "xformOp:translate:world" in op_values and "xformOp:transform:affine" in op_values:
            return "local"
        # Local-mode affine-collapsed stack (unsuffixed ops)
        if "xformOp:translate" in op_values and "xformOp:transform" in op_values:
            return "local"
        return "world"

    @staticmethod
    def _extract_xy_from_translate_value(value: object) -> Optional[Tuple[float, float]]:
        if value is None:
            return None
        try:
            return float(value[0]), float(value[1])
        except Exception:
            return None

    @staticmethod
    def _extract_rot_from_value(value: object) -> Optional[float]:
        if value is None:
            return None
        try:
            return float(value)
        except Exception:
            return None

    @staticmethod
    def _extract_yaw_deg_from_transform_value(value: object) -> Optional[float]:
        if value is None:
            return None
        try:
            matrix = Gf.Matrix4d(value)
            m = matrix.GetOrthonormalized()
            yaw_rad = math.atan2(m[1][0], m[0][0])
            return math.degrees(yaw_rad)
        except Exception:
            return None

    def _extract_pose_from_xform_ops(
        self,
        prim,
        preferred_size_mode: Optional[str],
        fallback_x: float,
        fallback_y: float,
        fallback_rotation: float,
    ) -> Dict[str, object]:
        op_values = self._collect_xform_op_values(prim)
        mode = self._normalize_size_mode(preferred_size_mode, default=self._infer_size_mode_from_xform_ops(op_values))

        if mode == "local":
            translate_candidates = ("xformOp:translate:world", "xformOp:translate")
            rotate_candidates = ("xformOp:rotateZ:world", "xformOp:rotateZ")
        else:
            translate_candidates = ("xformOp:translate", "xformOp:translate:world")
            rotate_candidates = ("xformOp:rotateZ", "xformOp:rotateZ:world")

        x = float(fallback_x)
        y = float(fallback_y)
        rotation = float(fallback_rotation)

        for name in translate_candidates:
            xy = self._extract_xy_from_translate_value(op_values.get(name))
            if xy is not None:
                x, y = xy
                break

        for name in rotate_candidates:
            rot = self._extract_rot_from_value(op_values.get(name))
            if rot is not None:
                rotation = rot
                break
        else:
            # Affine-collapsed local stack has no explicit rotate op.
            for name in ("xformOp:transform:affine", "xformOp:transform"):
                rot = self._extract_yaw_deg_from_transform_value(op_values.get(name))
                if rot is not None:
                    rotation = rot
                    break

        return {"X": x, "Y": y, "rotationZ": rotation, "size_mode": mode}

    def _extract_yaw_deg_from_matrix(self, matrix: Gf.Matrix4d) -> float:
        try:
            m = matrix.GetOrthonormalized()
            yaw_rad = math.atan2(m[1][0], m[0][0])
            return math.degrees(yaw_rad)
        except Exception as exc:
            omni.log.warn(f"Failed to extract yaw rotation: {exc}")
            return 0.0

    def _attach_placement_metadata_from_prim(self, prim, asset_url: Optional[str] = None) -> bool:
        if not prim or not prim.IsValid():
            omni.log.warn("Invalid prim for metadata attachment.")
            return False

        if asset_url is None:
            asset_url = self._get_asset_url_from_prim(prim)

        try:
            try:
                custom = prim.GetCustomData()
            except Exception:
                custom = {}

            custom_length = custom.get("asset_placer.length")
            custom_width = custom.get("asset_placer.width")
            custom_height = custom.get("asset_placer.height")
            custom_rotation = custom.get("asset_placer.rotationZ", 0.0)
            custom_x = custom.get("asset_placer.x", 0.0)
            custom_y = custom.get("asset_placer.y", 0.0)
            custom_size_mode = custom.get("asset_placer.size_mode")

            time_code = Usd.TimeCode.Default()
            bbox_cache = UsdGeom.BBoxCache(time_code, ["default"])
            bbox = bbox_cache.ComputeWorldBound(prim)
            bbox_range = bbox.ComputeAlignedRange()
            size_vec = bbox_range.GetMax() - bbox_range.GetMin()

            xformable = UsdGeom.Xformable(prim)
            matrix = xformable.ComputeLocalToWorldTransform(time_code)
            translation = matrix.ExtractTranslation()
            rotation = self._extract_yaw_deg_from_matrix(matrix)

            xform_data = self._extract_pose_from_xform_ops(
                prim,
                preferred_size_mode=str(custom_size_mode) if custom_size_mode else None,
                fallback_x=self._safe_float(custom_x, float(translation[0])),
                fallback_y=self._safe_float(custom_y, float(translation[1])),
                fallback_rotation=self._safe_float(custom_rotation, float(rotation)),
            )
            resolved_size_mode = xform_data["size_mode"]

            # Local mode keeps metadata dimensions as object-local semantics.
            # Recomputing from world AABB would corrupt Length/Width after rotation.
            if (
                resolved_size_mode == "local"
                and custom_length is not None
                and custom_width is not None
                and custom_height is not None
            ):
                length = float(custom_length)
                width = float(custom_width)
                height = float(custom_height)
            else:
                length = float(size_vec[0])
                width = float(size_vec[1])
                height = float(size_vec[2])

            x = float(xform_data["X"])
            y = float(xform_data["Y"])
            rotation = float(xform_data["rotationZ"])

            if length <= 0.0 or width <= 0.0 or height <= 0.0:
                omni.log.warn(f"Invalid bbox size for '{prim.GetPath()}': {size_vec}")
                return False

            self._store_placement_metadata(
                prim,
                asset_url,
                rotation,
                length,
                width,
                height,
                x,
                y,
                object_name=custom.get("asset_placer.object_name"),
                category=custom.get("asset_placer.category"),
                search_prompt=custom.get("asset_placer.search_prompt"),
                search_query=custom.get("asset_placer.search_query"),
                size_mode=resolved_size_mode,
            )
            omni.log.info(
                f"Attached placement metadata for '{prim.GetPath()}' (size_mode={resolved_size_mode})"
            )
            return True
        except Exception as exc:
            omni.log.warn(f"Failed to attach placement metadata for '{prim.GetPath()}': {exc}")
            return False

    def _iter_matching_asset_prims(self, stage, asset_url: str):
        """Iterate prims that directly reference the given asset URL or store it in custom data."""
        for prim in stage.Traverse():
            if not prim or not prim.IsValid():
                continue

            try:
                custom = prim.GetCustomData()
            except Exception:
                custom = {}

            if custom.get("asset_placer.asset_url") == asset_url:
                yield prim
                continue

            ref_items = self._get_reference_items_from_prim(prim)
            for item in ref_items:
                asset_path = getattr(item, "assetPath", None)
                if asset_path and str(asset_path) == asset_url:
                    yield prim
                    break

    async def _apply_rotation_offset_to_matching_assets(self, stage, asset_url: str) -> None:
        matched = 0
        updated = 0
        skipped = 0

        for prim in self._iter_matching_asset_prims(stage, asset_url):
            matched += 1
            data = self._build_data_from_prim_metadata(prim)
            if not data:
                skipped += 1
                continue
            updated += 1
            await self._apply_transform(prim, data, asset_url)
            await asyncio.sleep(0)

        omni.log.info(
            f"Applied rotation offset for asset '{asset_url}': updated={updated}, skipped={skipped}, matched={matched}"
        )
