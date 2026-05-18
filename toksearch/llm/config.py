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
"""Configuration object and loader for toksearch.llm.

Precedence (highest first):
  1. CLI flags (applied by callers after ``load_config()`` returns)
  2. Environment variables: ``FDP_LLM_BACKEND``, ``ANTHROPIC_API_KEY``,
     ``OPENAI_API_KEY``
  3. ``~/.fdp/config.toml`` (``[llm]`` table)
  4. Built-in defaults
"""

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from .errors import LLMConfigError


@dataclass
class Config:
    backend: str | None = None          # preset name; None → caller's default
    model: str | None = None            # overrides preset's default model
    max_iterations: int = 20
    anthropic_api_key: str | None = None
    openai_api_key: str | None = None
    user_presets: dict = field(default_factory=dict)


_DEFAULT_CONFIG_PATH = Path.home() / ".fdp" / "config.toml"


def load_config(config_path: Path | None = None) -> Config:
    cfg = Config()
    path = config_path if config_path is not None else _DEFAULT_CONFIG_PATH
    if path.exists():
        try:
            with path.open("rb") as f:
                data = tomllib.load(f)
        except tomllib.TOMLDecodeError as e:
            raise LLMConfigError(f"Malformed TOML in {path}: {e}") from e
        llm = data.get("llm", {})
        if "backend" in llm:
            cfg.backend = llm["backend"]
        if "model" in llm:
            cfg.model = llm["model"]
        if "max_iterations" in llm:
            cfg.max_iterations = int(llm["max_iterations"])
        if "anthropic_api_key" in llm:
            cfg.anthropic_api_key = llm["anthropic_api_key"]
        if "openai_api_key" in llm:
            cfg.openai_api_key = llm["openai_api_key"]
        cfg.user_presets = llm.get("presets", {})
    # Env overlay
    if "FDP_LLM_BACKEND" in os.environ:
        cfg.backend = os.environ["FDP_LLM_BACKEND"]
    if "ANTHROPIC_API_KEY" in os.environ:
        cfg.anthropic_api_key = os.environ["ANTHROPIC_API_KEY"]
    if "OPENAI_API_KEY" in os.environ:
        cfg.openai_api_key = os.environ["OPENAI_API_KEY"]
    return cfg
