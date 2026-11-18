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
namespace/ (variable checkpoint), figures/, and .lock while it is open.
"""

import json
import os
import socket
import subprocess
import sys
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


class ConversationInUse(RuntimeError):
    """The conversation is open in another process.

    ``host``, ``pid`` and ``since`` describe the holder, for a message such
    as "open on <host> since <since>"; any may be None if unknown.
    """

    def __init__(self, message, *, host=None, pid=None, since=None):
        super().__init__(message)
        self.host, self.pid, self.since = host, pid, since


LOCK_NAME = ".lock"


def _machine_id() -> str | None:
    """This machine's hardware address as hex, or None if it has none.

    uuid.getnode() falls back to a random number (multicast bit set) that
    changes per process, which would make every lock look foreign.
    """
    node = uuid.getnode()
    if node >> 40 & 1:
        return None
    return format(node, "012x")


def _process_start(pid) -> str | None:
    """An opaque start-time stamp for ``pid``, or None if it cannot be read.

    Linux: field 22 of /proc/<pid>/stat (clock ticks since boot). macOS:
    ``ps -o lstart=``. Distinguishes a reused pid from the original holder.
    """
    try:
        pid = int(pid)
    except (TypeError, ValueError):
        return None
    try:
        stat = open(f"/proc/{pid}/stat").read()
        return stat.rsplit(")", 1)[1].split()[19]
    except (OSError, IndexError):
        pass
    if sys.platform == "darwin":
        try:
            out = subprocess.run(["ps", "-o", "lstart=", "-p", str(pid)],
                                 capture_output=True, text=True, timeout=2)
            return out.stdout.strip() or None
        except (OSError, subprocess.SubprocessError):
            return None
    return None


def _pid_is_dead(pid) -> bool:
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return True
    except (PermissionError, ValueError, TypeError, OverflowError):
        return False
    return False


def _lock_is_stale(held: dict) -> bool:
    """True only if the holder is provably gone: same machine, process dead.

    Machine identity is the hardware address when both sides have one, else
    the hostname. A live pid whose start time differs from the recorded one
    is a reused pid, so the holder is dead.
    """
    mine = _machine_id()
    if held.get("machine") and mine:
        same_machine = held["machine"] == mine
    else:
        same_machine = held.get("host") == socket.gethostname()
    if not same_machine or "pid" not in held:
        return False
    if _pid_is_dead(held["pid"]):
        return True
    recorded = held.get("start")
    current = _process_start(held["pid"])
    return bool(recorded and current and recorded != current)


class ConversationFolder:
    def __init__(self, path):
        self.path = Path(path)

    meta_path = property(lambda self: self.path / "conversation.json")
    events_path = property(lambda self: self.path / "events.jsonl")
    history_path = property(lambda self: self.path / "history.jsonl")
    claude_dir = property(lambda self: self.path / "claude")
    namespace_dir = property(lambda self: self.path / "namespace")
    figures_dir = property(lambda self: self.path / "figures")
    lock_path = property(lambda self: self.path / LOCK_NAME)

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
                   if p.is_file() and p != self.lock_path)

    # ---- open-by-one-process lock ----
    #
    # An O_EXCL-created file, not flock: flock on BeeGFS does not reach
    # other clients. A lock left by a dead process on this host is taken
    # over; one from another host is never presumed stale.

    def _read_lock_raw(self) -> str | None:
        try:
            return self.lock_path.read_text()
        except OSError:
            return None

    @staticmethod
    def _parse_lock(raw) -> dict:
        try:
            held = json.loads(raw) if raw else {}
        except ValueError:
            return {}
        return held if isinstance(held, dict) else {}

    def _read_lock(self) -> dict:
        return self._parse_lock(self._read_lock_raw())

    def _lock_info(self) -> dict:
        info = {"pid": os.getpid(), "host": socket.gethostname(),
                "machine": _machine_id(),
                "start": _process_start(os.getpid()),
                "since": datetime.now().isoformat(timespec="seconds")}
        return {k: v for k, v in info.items() if v is not None}

    def _create_lock(self) -> bool:
        try:
            fd = os.open(self.lock_path,
                         os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        except FileExistsError:
            return False
        with os.fdopen(fd, "w") as f:
            json.dump(self._lock_info(), f)
        return True

    def _in_use(self, held: dict) -> ConversationInUse:
        return ConversationInUse(
            f"Conversation {self.path} is open in another process "
            f"(host {held.get('host', '?')}, pid {held.get('pid', '?')}, "
            f"since {held.get('since', '?')}).",
            host=held.get("host"), pid=held.get("pid"),
            since=held.get("since"))

    def _take_over(self, stale_raw: str) -> bool:
        """Replace the stale lock whose text is ``stale_raw``; False if beaten.

        The new lock is written to a unique temp file and hard-linked into
        place, which fails if a competitor created the lock first.
        """
        tmp = self.path / f"{LOCK_NAME}.{uuid.uuid4().hex}.tmp"
        try:
            with open(tmp, "x") as f:
                json.dump(self._lock_info(), f)
            # Remove only the lock judged stale, not one that replaced it.
            if self._read_lock_raw() == stale_raw:
                try:
                    os.unlink(self.lock_path)
                except FileNotFoundError:
                    pass
            try:
                os.link(tmp, self.lock_path)
            except FileExistsError:
                return False
            return True
        finally:
            try:
                os.unlink(tmp)
            except FileNotFoundError:
                pass

    def acquire_lock(self) -> None:
        """Mark the conversation open here; ConversationInUse if it is elsewhere."""
        if self._create_lock():
            return
        for _ in range(2):      # one retry if another taker wins the race
            raw = self._read_lock_raw()
            if raw is None:     # released meanwhile
                if self._create_lock():
                    return
                continue
            held = self._parse_lock(raw)
            if not _lock_is_stale(held):
                raise self._in_use(held)
            if self._take_over(raw):
                return
        raise self._in_use(self._read_lock())

    def break_lock(self) -> None:
        """Remove the lock whoever holds it ("open anyway"); the caller confirms."""
        try:
            os.unlink(self.lock_path)
        except FileNotFoundError:
            pass

    def release_lock(self) -> None:
        """Remove the lock if this process holds it; safe to call twice."""
        held = self._read_lock()
        mine = (held.get("pid") == os.getpid()
                and held.get("host") == socket.gethostname())
        if mine:
            try:
                os.unlink(self.lock_path)
            except FileNotFoundError:
                pass


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
