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
    "(matplotlib.pyplot), pd (pandas), np (numpy), px (plotly.express), "
    "go (plotly.graph_objects). Returns captured stdout "
    "and stderr. Populate 'thought' with a one-sentence description of what "
    "this code does and why."
    "\n\nPrefer plotly (`px`, `go`) for visualizations — the GUI "
    "renders them as interactive figures with zoom/pan/hover. Use "
    "matplotlib (`plt`) only for cases plotly handles poorly "
    "(animations, specialized scientific plots). Either way, call "
    "`fig.show()` or `plt.show()` to send the figure to the side "
    "pane in the GUI; in CLI mode the call is captured as text "
    "output."
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

# ----------------------------------------------------------------------
# lookup_docs + skill discovery
# ----------------------------------------------------------------------

@dataclass(frozen=True)
class Skill:
    name: str
    description: str
    body: str


def parse_skill_md(path: Path) -> tuple[dict, str]:
    """Return ``(frontmatter, body)`` for a SKILL.md file.

    Frontmatter is the YAML-ish block between two ``---`` lines at the top of
    the file (we only need ``name`` and ``description`` so we do a tiny
    line-by-line parser instead of pulling in PyYAML).  If no frontmatter is
    present, returns ``({}, full_text)``.
    """
    text = path.read_text()
    if text.startswith("---"):
        _, fm, body = text.split("---", 2)
        fm_dict = {}
        for line in fm.strip().splitlines():
            if ":" in line:
                k, _, v = line.partition(":")
                fm_dict[k.strip()] = v.strip()
        return fm_dict, body.lstrip("\n")
    return {}, text


def discover_skills(skill_dirs: list[Path]) -> dict[str, Skill]:
    """Return ``{name: Skill}`` for every SKILL.md found under the given dirs.

    Each entry in ``skill_dirs`` is searched one level deep: each subdirectory
    containing a ``SKILL.md`` becomes a skill named after the subdirectory.
    Dirs that don't exist are silently skipped.
    """
    skills: dict[str, Skill] = {}
    for d in skill_dirs:
        if not d.exists():
            continue
        for sub in sorted(d.iterdir()):
            if not sub.is_dir():
                continue
            skill_md = sub / "SKILL.md"
            if not skill_md.exists():
                continue
            fm, body = parse_skill_md(skill_md)
            skills[sub.name] = Skill(
                name=sub.name,
                description=fm.get("description", ""),
                body=body,
            )
    return skills


_LOOKUP_DOCS_DESCRIPTION = (
    "Read a documentation skill. Returns the SKILL.md body for one of the "
    "registered skills. Call this when you need detail on a specific "
    "toksearch feature beyond what's in the system prompt."
)


def _lookup_docs_handler(args: dict, session) -> ToolOutput:
    name = args["skill_name"]
    skill = session.skills.get(name)
    if skill is None:
        available = sorted(session.skills)
        return ToolOutput(
            text=f"Unknown skill: {name!r}. Available: {available}",
            is_error=True,
        )
    return ToolOutput(text=skill.body, is_error=False)


LOOKUP_DOCS = ToolSpec(
    name="lookup_docs",
    description=_LOOKUP_DOCS_DESCRIPTION,
    input_schema={
        "type": "object",
        "properties": {
            "skill_name": {
                "type": "string",
                "description": "Name of a skill registered with the Session.",
            },
        },
        "required": ["skill_name"],
    },
    handler=_lookup_docs_handler,
)
