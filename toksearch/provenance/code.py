# Copyright 2026 General Atomics
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
"""Capture of the code version behind a pipeline run."""

import os
import subprocess
import sys
from dataclasses import dataclass, asdict
from typing import Optional, Tuple


@dataclass(frozen=True)
class CodeSpec:
    """Which code produced a run."""

    commit: Optional[str]
    dirty: Optional[bool]
    repo_root: Optional[str]
    script: Optional[str]
    argv: Tuple[str, ...]

    def to_dict(self) -> dict:
        return asdict(self)


def _git(args, cwd):
    try:
        out = subprocess.run(
            ["git"] + args,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    return out.stdout.strip()


def capture_code(cwd: Optional[str] = None) -> CodeSpec:
    """Describe the code version of the running script.

    Running outside a git repository is not an error here — it yields a
    CodeSpec with ``commit=None``. Backends that *require* git (cmflib does)
    are responsible for rejecting that themselves, early and with a clear
    message.
    """
    cwd = cwd or os.getcwd()

    commit = _git(["rev-parse", "HEAD"], cwd)
    repo_root = _git(["rev-parse", "--show-toplevel"], cwd)

    dirty = None
    if commit is not None:
        status = _git(["status", "--porcelain"], cwd)
        dirty = bool(status) if status is not None else None

    script = sys.argv[0] if sys.argv else None

    return CodeSpec(
        commit=commit,
        dirty=dirty,
        repo_root=repo_root,
        script=script,
        argv=tuple(sys.argv),
    )
