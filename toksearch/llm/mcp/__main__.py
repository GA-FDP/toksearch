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
"""Run the standalone skills MCP server over stdio.

Usage: ``python -m toksearch.llm.mcp``

Environment:
- ``TOKSEARCH_SKILL_DIRS``: os.pathsep-delimited extra skill directories.
- ``TOKSEARCH_SKILL_PACKAGES``: os.pathsep-delimited entry-point names to keep.
  Unset => no filter (all entry-point skills).  Set-but-empty => exclude all
  entry-point skills (extra dirs still load).
"""

import os
from pathlib import Path

from .server import build_server


def main() -> None:
    raw_dirs = os.environ.get("TOKSEARCH_SKILL_DIRS", "")
    extra_dirs = [Path(p) for p in raw_dirs.split(os.pathsep) if p]

    if "TOKSEARCH_SKILL_PACKAGES" in os.environ:
        raw_pkgs = os.environ["TOKSEARCH_SKILL_PACKAGES"]
        packages = [p for p in raw_pkgs.split(os.pathsep) if p]
    else:
        packages = None

    build_server(extra_dirs=extra_dirs, packages=packages).run("stdio")


if __name__ == "__main__":
    main()
