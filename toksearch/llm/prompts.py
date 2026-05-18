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
"""System-prompt assembly.

The kernel prompt is small (~20 lines) and constant.  The dynamic parts —
the list of contributed packages and the catalog of available skills — are
built from the Session's installed contributors at construction time.
"""

from .tools import Skill


_KERNEL = """\
You are a TokSearch expert. Use the run_python tool to execute code that fetches
and analyzes fusion data. The namespace persists across tool calls in this
session - variables from earlier calls are available in later ones.

{namespace_section}{catalog_section}Rules:
- Do not include import statements; common modules are pre-imported.
- If code raises an error, read the traceback, fix it, and try again.
- When you have a result, store it in a variable named `result` or describe it
  in plain text. Do not call any tool to "finish" - just stop emitting tool calls.
"""


def build_system_prompt(
    skills: dict[str, Skill],
    namespace_entries: list[tuple[str, str]],
) -> str:
    """Build the system prompt from the registered contributors.

    Parameters
    ----------
    skills:
        Mapping of skill name to ``Skill`` (from ``tools.discover_skills``).
    namespace_entries:
        List of ``(name, description)`` for each package contributed to the
        run_python namespace (core toksearch + any extras).
    """
    if namespace_entries:
        lines = ["You have access to fusion data via the following installed packages:"]
        for name, desc in namespace_entries:
            lines.append(f"- {name}: {desc}" if desc else f"- {name}")
        namespace_section = "\n".join(lines) + "\n\n"
    else:
        namespace_section = ""

    if skills:
        lines = ["Available documentation skills (call lookup_docs(skill_name=...) to read):"]
        for name in sorted(skills):
            desc = skills[name].description
            lines.append(f"- {name}: {desc}" if desc else f"- {name}")
        catalog_section = "\n".join(lines) + "\n\n"
    else:
        catalog_section = ""

    return _KERNEL.format(namespace_section=namespace_section,
                          catalog_section=catalog_section)
