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
"""Standalone skills MCP server.

Discovers ``SKILL.md`` documentation skills from the ``toksearch.llm.skills``
entry-point group (plus extra dirs) and serves each as a ``skill://<name>``
resource, with a ``read_skill`` tool for clients that don't read resources.
"""

from pathlib import Path

from mcp.server.fastmcp import FastMCP

from ..discovery import discover_skill_dirs
from ..tools import Skill, discover_skills


def _make_reader(body: str):
    def _read() -> str:
        return body
    return _read


def discover_filtered_skills(
    extra_dirs: list[Path] | None = None,
    packages: list[str] | None = None,
) -> dict[str, Skill]:
    """Return ``{name: Skill}`` from entry-point dirs + extra dirs.

    ``packages`` (entry-point names) filters the entry-point dirs only;
    ``extra_dirs`` are always included.  ``packages=None`` means no filter;
    ``packages=[]`` excludes all entry-point dirs.

    On a skill-name collision, ``extra_dirs`` take precedence over
    entry-point dirs because they are appended last and ``discover_skills``
    uses last-wins dict assignment.
    """
    pairs = discover_skill_dirs()  # [(entry_point_name, Path), ...]
    if packages is not None:
        pairs = [(n, d) for n, d in pairs if n in packages]
    dirs = [d for _, d in pairs]
    if extra_dirs:
        dirs.extend(extra_dirs)
    return discover_skills(dirs)


def build_server(
    extra_dirs: list[Path] | None = None,
    packages: list[str] | None = None,
) -> FastMCP:
    """Build a FastMCP server exposing discovered skills as resources + tool."""
    mcp = FastMCP("toksearch-skills")
    skills = discover_filtered_skills(extra_dirs=extra_dirs, packages=packages)

    for name, skill in skills.items():
        mcp.resource(
            f"skill://{name}",
            name=name,
            description=skill.description,
            mime_type="text/markdown",
        )(_make_reader(skill.body))

    @mcp.tool(
        description="Read a documentation skill's SKILL.md body by name. "
                    "Call this for detail on a specific toksearch feature.")
    def read_skill(skill_name: str) -> str:
        s = skills.get(skill_name)
        if s is None:
            raise ValueError(
                f"Unknown skill: {skill_name!r}. Available: {sorted(skills)}")
        return s.body

    return mcp
