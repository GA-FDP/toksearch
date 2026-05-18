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
"""Tools registered with a Session.

Two tools ship in PR 1:

- ``run_python`` — executes Python code in the Session's persistent namespace.
- ``lookup_docs`` — returns the SKILL.md body for a named skill.

Tool handlers take ``(args: dict, session)`` and return a ``ToolOutput``.
"""

import io
import traceback
from contextlib import redirect_stdout, redirect_stderr
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


@dataclass(frozen=True)
class ToolOutput:
    """Result of a tool invocation."""

    text: str
    is_error: bool = False
    interrupted: bool = False    # True only when handler caught KeyboardInterrupt


@dataclass(frozen=True)
class ToolSpec:
    """Backend-agnostic tool descriptor.

    Backends translate ``input_schema`` to their native shape on request build.
    """

    name: str
    description: str
    input_schema: dict
    handler: Callable[[dict, Any], ToolOutput]


# ----------------------------------------------------------------------
# run_python
# ----------------------------------------------------------------------

_RUN_PYTHON_DESCRIPTION = (
    "Execute a Python code string. The execution namespace persists across "
    "calls within this session, so variables set in earlier calls are "
    "available later. The namespace is pre-populated with: toksearch, plt "
    "(matplotlib.pyplot), pd (pandas), np (numpy). Returns captured stdout "
    "and stderr. Populate 'thought' with a one-sentence description of what "
    "this code does and why."
)


def _run_python_handler(args: dict, session) -> ToolOutput:
    code = args["code"]
    stdout_buf = io.StringIO()
    stderr_buf = io.StringIO()
    is_error = False
    interrupted = False
    try:
        with redirect_stdout(stdout_buf), redirect_stderr(stderr_buf):
            exec(code, session.namespace)  # noqa: S102 -- intentional
    except KeyboardInterrupt:
        return ToolOutput(text="(interrupted)", is_error=True, interrupted=True)
    except Exception:
        stderr_buf.write(traceback.format_exc())
        is_error = True
    out = stdout_buf.getvalue()
    err = stderr_buf.getvalue()
    parts = []
    if out:
        parts.append(f"stdout:\n{out}")
    if err:
        parts.append(f"stderr:\n{err}")
    text = "\n".join(parts) if parts else "(no output)"
    return ToolOutput(text=text, is_error=is_error, interrupted=interrupted)


RUN_PYTHON = ToolSpec(
    name="run_python",
    description=_RUN_PYTHON_DESCRIPTION,
    input_schema={
        "type": "object",
        "properties": {
            "code":    {"type": "string",
                        "description": "Python code to execute."},
            "thought": {"type": "string",
                        "description": "One-sentence description of what this "
                                       "code does and why."},
        },
        "required": ["code", "thought"],
    },
    handler=_run_python_handler,
)
