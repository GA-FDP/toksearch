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
"""Session — the user-facing object for toksearch.llm.

Owns: history, run_python namespace, tool registry, skill registry, backend.
"""

from pathlib import Path
from typing import Callable

from .backends.base import Backend, Callbacks
from .events import TurnComplete
from .messages import Message, TextBlock, ToolResultBlock, ToolUseBlock
from .prompts import build_system_prompt
from .tools import LOOKUP_DOCS, RUN_PYTHON, ToolOutput, discover_skills


def _default_namespace() -> dict:
    """Build the standard run_python namespace.

    Imports happen here (not at module load) so that simply importing
    ``toksearch.llm`` doesn't drag in matplotlib / pandas / numpy until the
    user actually constructs a Session.  toksearch and numpy are required;
    pandas and matplotlib are optional (matplotlib is in the ``[llm]`` extra
    but tests run without it).
    """
    import toksearch
    import numpy as np
    ns = {"toksearch": toksearch, "np": np}
    try:
        import pandas as pd
        ns["pd"] = pd
    except ImportError:
        pass
    try:
        import matplotlib
        matplotlib.use("Agg", force=False)  # idempotent; safe in non-GUI envs
        import matplotlib.pyplot as plt
        ns["plt"] = plt
    except ImportError:
        pass
    try:
        import plotly.express as _px
        import plotly.graph_objects as _go
        ns["px"] = _px
        ns["go"] = _go
    except ImportError:
        pass
    return ns


class Session:
    """A conversational LLM session over the run_python persistent namespace."""

    def __init__(
        self,
        backend: Backend,
        model: str | None = None,
        max_iterations: int = 20,
        extra_namespace: dict | None = None,
        packages: list[str] | None = None,
        extra_skill_dirs: list[Path] | None = None,
    ):
        from .discovery import (
            discover_namespace_contributors,
            discover_skill_dirs,
        )
        self.backend = backend
        self.model = model or backend.default_model
        self.max_iterations = max_iterations
        # ---- Namespace: defaults + discovered + extras ----
        self.namespace = _default_namespace()
        namespace_entries: list[tuple[str, str]] = []
        for name, value, desc in discover_namespace_contributors():
            if packages is not None and name not in packages:
                continue
            self.namespace[name] = value
            namespace_entries.append((name, desc))
        if extra_namespace:
            self.namespace.update(extra_namespace)
        # ---- Skills: discovered + extras ----
        skill_dirs: list[Path] = []
        for name, d in discover_skill_dirs():
            if packages is not None and name not in packages:
                continue
            skill_dirs.append(d)
        if extra_skill_dirs:
            skill_dirs.extend(extra_skill_dirs)
        self.skills = discover_skills(skill_dirs)
        # ---- Tools (fixed in PR 1) ----
        self.tool_specs = [RUN_PYTHON, LOOKUP_DOCS]
        self._tools_by_name = {t.name: t for t in self.tool_specs}
        # ---- System prompt ----
        self.system_prompt = build_system_prompt(self.skills, namespace_entries)
        self.history: list[Message] = []

    # ---- Public API ----

    def send(
        self,
        prompt: str,
        *,
        on_text: Callable[[str], None] | None = None,
        on_tool_call=None,
        on_tool_result=None,
        on_event=None,
        confirm=None,
    ) -> TurnComplete:
        cbs = Callbacks(
            on_text=on_text,
            on_tool_call=on_tool_call,
            on_tool_result=on_tool_result,
            on_event=on_event,
            confirm=confirm,
        )
        return self.backend.run_conversation(
            self, prompt, cbs, max_iterations=self.max_iterations,
        )

    def reset(self) -> None:
        """Clear history and reset the namespace to its initial pre-populated state."""
        self.namespace = _default_namespace()
        self.history = []

    # ---- Hooks called by backends (semi-private) ----

    def _append_user(self, text: str) -> None:
        self.history.append(Message(role="user",
                                     content=[TextBlock(text=text)]))

    def _append_assistant(self, blocks) -> None:
        self.history.append(Message(role="assistant",
                                     content=list(blocks)))

    def _append_tool_result(self, tool_use_id: str, output: str,
                             is_error: bool) -> None:
        self.history.append(Message(role="user", content=[
            ToolResultBlock(tool_use_id=tool_use_id,
                            output=output, is_error=is_error)
        ]))

    def _execute_tool(self, block: ToolUseBlock) -> ToolOutput:
        spec = self._tools_by_name.get(block.name)
        if spec is None:
            return ToolOutput(text=f"Unknown tool: {block.name}",
                              is_error=True)
        return spec.handler(block.args, self)
