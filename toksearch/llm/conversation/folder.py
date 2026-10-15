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

"""One self-contained folder per conversation.

Layout: conversation.json (metadata), events.jsonl (everything shown),
history.jsonl (Session.history), claude/ (SDK transcript mirror),
namespace/ (variable checkpoint), figures/.
"""

import json
import os
import uuid
from datetime import datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


def default_root() -> Path:
    env = os.environ.get("TOKSEARCH_SESSIONS_DIR")
    return Path(env) if env else Path.home() / "FDP" / "sessions"


def _toksearch_version() -> str | None:
    try:
        return version("toksearch")
    except PackageNotFoundError:
        return None


class ConversationFolder:
    def __init__(self, path):
        self.path = Path(path)

    meta_path = property(lambda self: self.path / "conversation.json")
    events_path = property(lambda self: self.path / "events.jsonl")
    history_path = property(lambda self: self.path / "history.jsonl")
    claude_dir = property(lambda self: self.path / "claude")
    namespace_dir = property(lambda self: self.path / "namespace")
    figures_dir = property(lambda self: self.path / "figures")

    @classmethod
    def create(cls, root, *, now: datetime | None = None) -> "ConversationFolder":
        now = now or datetime.now()
        base = Path(root) / now.strftime("%Y-%m-%d-%H%M%S")
        path, n = base, 1
        while path.exists():
            n += 1
            path = base.with_name(f"{base.name}-{n}")
        path.mkdir(parents=True)
        stamp = now.isoformat(timespec="seconds")
        folder = cls(path)
        folder.write_meta({
            "id": str(uuid.uuid4()), "title": "", "created": stamp,
            "last_active": stamp, "backend": None, "model": None,
            "claude_session_id": None,
            "fdp_core_version": os.environ.get("FDP_CORE_VERSION"),
            "toksearch_version": _toksearch_version(),
        })
        return folder

    def read_meta(self) -> dict:
        return json.loads(self.meta_path.read_text())

    def write_meta(self, meta: dict) -> None:
        tmp = self.meta_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(meta, indent=1))
        os.replace(tmp, self.meta_path)

    def update_meta(self, **changes) -> dict:
        meta = self.read_meta()
        meta.update(changes)
        self.write_meta(meta)
        return meta

    def size_bytes(self) -> int:
        return sum(p.stat().st_size for p in self.path.rglob("*")
                   if p.is_file())


def list_conversations(root=None) -> list[dict]:
    """Metadata of every conversation under ``root``, newest activity first."""
    root = Path(root) if root is not None else default_root()
    if not root.is_dir():
        return []
    out = []
    for d in root.iterdir():
        folder = ConversationFolder(d)
        if not folder.meta_path.is_file():
            continue
        meta = folder.read_meta()
        meta["path"] = str(d)
        meta["size_bytes"] = folder.size_bytes()
        out.append(meta)
    out.sort(key=lambda m: m.get("last_active") or "", reverse=True)
    return out
