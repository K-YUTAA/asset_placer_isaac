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

        if length is None or width is None or height is None:
            omni.log.warn(f"Missing placement metadata for '{prim.GetPath()}'.")
            return None

        return {
            "Length": float(length),
            "Width": float(width),
            "Height": float(height),
            "rotationZ": float(rotation),
            "X": float(x),
            "Y": float(y),
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
        except Exception as exc:
            omni.log.warn(f"Failed to store placement metadata for '{prim.GetPath()}': {exc}")

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
            time_code = Usd.TimeCode.Default()
            bbox_cache = UsdGeom.BBoxCache(time_code, ["default"])
            bbox = bbox_cache.ComputeWorldBound(prim)
            bbox_range = bbox.ComputeAlignedRange()
            size_vec = bbox_range.GetMax() - bbox_range.GetMin()

            xformable = UsdGeom.Xformable(prim)
            matrix = xformable.ComputeLocalToWorldTransform(time_code)
            translation = matrix.ExtractTranslation()
            rotation = self._extract_yaw_deg_from_matrix(matrix)

            length = float(size_vec[0])
            width = float(size_vec[1])
            height = float(size_vec[2])
            x = float(translation[0])
            y = float(translation[1])

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
            )
            omni.log.info(f"Attached placement metadata for '{prim.GetPath()}'")
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
