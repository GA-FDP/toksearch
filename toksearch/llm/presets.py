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
"""Backend presets.

A ``Preset`` is a named configuration that maps a user-facing backend name
(e.g. ``"anthropic"``, ``"amsc"``, ``"openai"``) to a concrete ``Backend``
class plus the kwargs needed to construct it.  Presets exist so that
site-specific endpoints (AmSC's Anthropic-compatible URL, etc.) can be added
without subclassing ``AnthropicBackend`` just to change a ``base_url``.

Built-in presets ship with core toksearch.  User presets come from
``~/.fdp/config.toml``'s ``[llm.presets.<name>]`` tables and are merged in via
``Config.user_presets``.  In PR 2, package-contributed presets will also be
discovered via the ``toksearch.llm.presets`` entry point.
"""

from dataclasses import dataclass, field, replace

from .config import Config
from .errors import LLMConfigError


@dataclass(frozen=True)
class Preset:
    backend: str                # "anthropic" | "openai" | "claude-max"
    model: str | None = None
    base_url: str | None = None
    api_key_file: str | None = None
    api_key_env: str | None = None
    extra: dict = field(default_factory=dict)


BUILTIN_PRESETS: dict[str, Preset] = {
    "anthropic": Preset(
        backend="anthropic",
        model="claude-sonnet-4-6",
        api_key_env="ANTHROPIC_API_KEY",
    ),
    "openai": Preset(
        backend="openai",
        model="gpt-4o",
        api_key_env="OPENAI_API_KEY",
    ),
    "claude-max": Preset(
        backend="claude-max",
        model=None,  # SDK picks the default; users may override via CLI.
        api_key_env=None,
        api_key_file=None,
    ),
}


def resolve_preset(name: str, config: Config) -> Preset:
    """Resolve a preset name to a fully-populated ``Preset``.

    Lookup order (highest precedence first):
      1. User preset (``Config.user_presets[name]``) merged onto whatever
         base (built-in, discovered, or new) applies.
      2. Built-in preset (``BUILTIN_PRESETS[name]``).
      3. Discovered preset (``toksearch.llm.presets`` entry point).
      4. A user-only preset (no base; ``Preset(**user_preset)``).

    Built-ins shadow discovered presets of the same name -- this prevents
    a misconfigured package from silently hijacking a stable user-facing
    name like ``"anthropic"``.
    """
    from .discovery import discover_presets

    user = config.user_presets.get(name, {})
    if name in BUILTIN_PRESETS:
        base = BUILTIN_PRESETS[name]
        if user:
            return replace(base, **{k: v for k, v in user.items()
                                    if k in {"backend", "model", "base_url",
                                             "api_key_file", "api_key_env"}})
        return base
    discovered = discover_presets()
    if name in discovered:
        base = discovered[name]
        if user:
            return replace(base, **{k: v for k, v in user.items()
                                    if k in {"backend", "model", "base_url",
                                             "api_key_file", "api_key_env"}})
        return base
    if user:
        try:
            return Preset(**{k: v for k, v in user.items()
                             if k in {"backend", "model", "base_url",
                                      "api_key_file", "api_key_env", "extra"}})
        except TypeError as e:
            raise LLMConfigError(f"Invalid user preset {name!r}: {e}") from e
    raise LLMConfigError(
        f"Unknown backend / preset: {name!r}. Built-in presets: "
        f"{sorted(BUILTIN_PRESETS)}. Discovered presets: {sorted(discovered)}. "
        f"Define a user preset in ~/.fdp/config.toml under "
        f"[llm.presets.{name}].")
