"""Separate skills-about-skills (meta) from domain skills during retrieval."""
from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Literal, TypeVar

from skill_inject_mcp.schemas import SkillMeta, ValidationErrorItem

SkillLayer = Literal["meta", "domain"]


def load_layer_overrides(
    path: Path | None,
    errors: list[ValidationErrorItem] | None = None,
    known_skill_ids: Iterable[str] | None = None,
) -> dict[str, SkillLayer]:
    if path is None:
        return {}
    file_path = str(path)
    try:
        raw_bytes = Path(path).read_bytes()
    except OSError:
        if errors is not None:
            errors.append(
                ValidationErrorItem(
                    code="unreadable_layer_overrides",
                    message=f"Layer overrides file could not be read: {file_path}",
                    path=file_path,
                )
            )
        return {}
    try:
        raw_text = raw_bytes.decode("utf-8-sig")
        data = json.loads(raw_text)
    except (UnicodeDecodeError, json.JSONDecodeError):
        if errors is not None:
            errors.append(
                ValidationErrorItem(
                    code="invalid_layer_overrides",
                    message=f"Layer overrides file is not valid JSON: {file_path}",
                    path=file_path,
                )
            )
        return {}
    if not isinstance(data, dict):
        if errors is not None:
            errors.append(
                ValidationErrorItem(
                    code="invalid_layer_overrides",
                    message="Layer overrides root must be a JSON object",
                    path=file_path,
                )
            )
        return {}
    if "schema_version" not in data or type(data.get("schema_version")) is not int or data.get("schema_version") != 1:
        if errors is not None:
            errors.append(
                ValidationErrorItem(
                    code="invalid_layer_overrides_version",
                    message=f"Invalid layer overrides schema_version {data.get('schema_version')!r}; expected 1",
                    path=file_path,
                )
            )
        return {}
    layers = data.get("layers")
    if not isinstance(layers, dict) or set(data.keys()) != {"schema_version", "layers"}:
        if errors is not None:
            errors.append(
                ValidationErrorItem(
                    code="invalid_layer_overrides",
                    message="Layer overrides must contain 'schema_version' and a 'layers' object",
                    path=file_path,
                )
            )
        return {}
    parsed: dict[str, SkillLayer] = {}
    file_errors: list[ValidationErrorItem] = []
    for skill_id, raw_layer in layers.items():
        if not isinstance(skill_id, str) or not skill_id.strip():
            file_errors.append(
                ValidationErrorItem(
                    code="invalid_layer_override",
                    message=f"Invalid skill_id {skill_id!r} in layer overrides",
                    path=file_path,
                )
            )
            continue
        if not isinstance(raw_layer, str) or raw_layer not in ("meta", "domain"):
            file_errors.append(
                ValidationErrorItem(
                    code="invalid_layer_override",
                    message=f"Invalid layer {raw_layer!r} for skill '{skill_id}' in layer overrides; expected 'meta' or 'domain'",
                    path=file_path,
                )
            )
            continue
        parsed[skill_id] = raw_layer  # type: ignore[assignment]
    if file_errors:
        if errors is not None:
            errors.extend(file_errors)
        return {}
    if known_skill_ids is not None and errors is not None:
        known = set(known_skill_ids)
        for skill_id in parsed:
            if skill_id not in known:
                errors.append(
                    ValidationErrorItem(
                        code="unknown_layer_override",
                        message=f"Layer override references unknown skill_id '{skill_id}'",
                        path=file_path,
                    )
                )
    return parsed


def classify_skill_layer(
    skill: SkillMeta,
    errors: list[ValidationErrorItem] | None = None,
    layer_overrides: Mapping[str, SkillLayer] | None = None,
) -> SkillLayer:
    if layer_overrides and skill.skill_id in layer_overrides:
        return layer_overrides[skill.skill_id]
    if "layer" not in skill.frontmatter or skill.frontmatter.get("layer") is None:
        return "domain"
    raw = skill.frontmatter.get("layer")
    if isinstance(raw, str):
        normalized = raw.strip().lower()
        if normalized in ("meta", "domain"):
            return normalized  # type: ignore[return-value]
    if errors is not None:
        errors.append(
            ValidationErrorItem(
                code="invalid_skill_layer",
                message=f"Invalid skill layer {raw!r} in frontmatter; expected 'meta' or 'domain'",
                path=skill.path,
            )
        )
    return "domain"


Hit = TypeVar("Hit")


def partition_hits(
    hits: Iterable[Hit],
    skills: dict[str, SkillMeta],
    wanted: SkillLayer,
) -> tuple[list[Hit], list[Hit]]:
    primary: list[Hit] = []
    secondary: list[Hit] = []
    for hit in hits:
        skill_id = getattr(hit, "skill_id", None)
        skill = skills.get(skill_id) if isinstance(skill_id, str) else None
        layer = skill.layer if skill is not None else "domain"
        (primary if layer == wanted else secondary).append(hit)
    return primary, secondary
