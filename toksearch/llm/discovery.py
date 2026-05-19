# Copyright 2024 General Atomics
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Entry-point discovery for toksearch.llm contributors.

Three entry-point groups are scanned at Session construction time:

- ``toksearch.llm.namespace`` -- value resolves to a module/object; bound
  under the entry-point name in the run_python namespace.  A module-level
  ``__llm_description__`` attribute, if present, populates the system-prompt
  catalog line.
- ``toksearch.llm.skills`` -- value resolves to a ``Path``, a string path, or
  a no-argument callable returning one.  Pointed-to directory is scanned
  for ``SKILL.md`` files.
- ``toksearch.llm.presets`` -- value resolves to a ``Preset`` instance and
  is added to the preset registry.

Results are cached on first call; ``clear_discovery_cache()`` resets them
(intended for tests).
"""

import logging
from importlib.metadata import entry_points
from pathlib import Path
from typing import Callable

from .presets import Preset

_log = logging.getLogger("toksearch.llm.discovery")

_NAMESPACE_CACHE: list | None = None
_SKILLS_CACHE: list | None = None
_PRESETS_CACHE: dict | None = None


def _entry_points(group: str):
    """Indirection for monkeypatching in tests; returns iterable of EntryPoints."""
    return entry_points(group=group)


def clear_discovery_cache() -> None:
    global _NAMESPACE_CACHE, _SKILLS_CACHE, _PRESETS_CACHE
    _NAMESPACE_CACHE = None
    _SKILLS_CACHE = None
    _PRESETS_CACHE = None


def discover_namespace_contributors() -> list[tuple[str, object, str]]:
    """Return ``[(name, value, description), ...]``."""
    global _NAMESPACE_CACHE
    if _NAMESPACE_CACHE is not None:
        return _NAMESPACE_CACHE
    out = []
    for ep in _entry_points("toksearch.llm.namespace"):
        try:
            value = ep.load()
        except Exception as e:
            _log.warning("Failed to load namespace entry point %r: %s",
                          ep.name, e)
            continue
        desc = getattr(value, "__llm_description__", "")
        out.append((ep.name, value, desc))
    _NAMESPACE_CACHE = out
    return out


def discover_skill_dirs() -> list[tuple[str, Path]]:
    """Return ``[(name, dir), ...]`` of contributor skill directories."""
    global _SKILLS_CACHE
    if _SKILLS_CACHE is not None:
        return _SKILLS_CACHE
    out = []
    for ep in _entry_points("toksearch.llm.skills"):
        try:
            value = ep.load()
        except Exception as e:
            _log.warning("Failed to load skills entry point %r: %s",
                          ep.name, e)
            continue
        if callable(value):
            value = value()
        if isinstance(value, str):
            value = Path(value)
        if not isinstance(value, Path):
            _log.warning("Skills entry point %r did not resolve to a Path: %r",
                          ep.name, value)
            continue
        out.append((ep.name, value))
    _SKILLS_CACHE = out
    return out


def discover_presets() -> dict[str, Preset]:
    """Return ``{name: Preset}`` of contributor-supplied presets."""
    global _PRESETS_CACHE
    if _PRESETS_CACHE is not None:
        return _PRESETS_CACHE
    out: dict[str, Preset] = {}
    for ep in _entry_points("toksearch.llm.presets"):
        try:
            value = ep.load()
        except Exception as e:
            _log.warning("Failed to load presets entry point %r: %s",
                          ep.name, e)
            continue
        if not isinstance(value, Preset):
            _log.warning("Presets entry point %r did not resolve to a Preset: %r",
                          ep.name, value)
            continue
        out[ep.name] = value
    _PRESETS_CACHE = out
    return out
