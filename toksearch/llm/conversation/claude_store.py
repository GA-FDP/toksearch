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

"""FolderSessionStore: a claude-agent-sdk SessionStore kept in one folder.

The SDK mirrors every transcript line here and, on resume, loads it back
into a temporary CLAUDE_CONFIG_DIR, so a conversation folder alone is enough
to resume and ~/.claude is never consulted. Each conversation has its own
folder, so project_key is ignored. Duck-typed to the SDK's SessionStore
protocol (only append and load are required), so importing this module does
not need the SDK.
"""

import json
import uuid
from pathlib import Path


def _check_session_id(session_id: str) -> str:
    try:
        uuid.UUID(session_id)
    except (ValueError, TypeError, AttributeError):
        raise ValueError(f"not a session UUID: {session_id!r}") from None
    return session_id


class FolderSessionStore:
    def __init__(self, directory):
        self.dir = Path(directory)

    def _path(self, key) -> Path:
        sid = _check_session_id(key["session_id"])
        sub = key.get("subpath")
        if not sub:
            return self.dir / f"{sid}.jsonl"
        parts = sub.split("/")
        if any(p in ("", ".", "..") for p in parts):
            raise ValueError(f"bad subpath: {sub!r}")
        return self.dir.joinpath(sid, *parts[:-1], f"{parts[-1]}.jsonl")

    def has_transcript(self, session_id: str) -> bool:
        p = self.dir / f"{_check_session_id(session_id)}.jsonl"
        return p.is_file() and p.stat().st_size > 0

    async def append(self, key, entries) -> None:
        p = self._path(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "a") as f:
            for e in entries:
                f.write(json.dumps(e, separators=(",", ":")) + "\n")
            f.flush()

    async def load(self, key):
        p = self._path(key)
        if not p.is_file():
            return None
        out, seen = [], set()
        for line in p.read_text().splitlines():
            if not line.strip():
                continue
            try:
                e = json.loads(line)
            except json.JSONDecodeError:  # truncated by a crash mid-write
                continue
            u = e.get("uuid")
            if u is not None:
                if u in seen:
                    continue
                seen.add(u)
            out.append(e)
        return out

    async def list_sessions(self, project_key):
        if not self.dir.is_dir():
            return []
        return [{"session_id": p.stem, "mtime": int(p.stat().st_mtime * 1000)}
                for p in self.dir.glob("*.jsonl")]

    async def list_subkeys(self, key):
        base = self.dir / _check_session_id(key["session_id"])
        if not base.is_dir():
            return []
        return sorted(p.relative_to(base).with_suffix("").as_posix()
                      for p in base.rglob("*.jsonl"))
