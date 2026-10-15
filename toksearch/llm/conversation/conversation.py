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

"""Conversation: a Session whose chat, model context and variables survive a stop.

Every event is appended to events.jsonl as it happens. After each turn the
history is saved and the worker's namespace is checkpointed. open() restores
the model's context (SDK resume from the folder's transcript for claude-max,
history.jsonl otherwise), falling back to a recap built from events.jsonl,
and prefixes the next prompt with a note saying what was and was not
restored. Variables are loaded by the worker when Python next runs.
"""

import json
import os
import uuid
from datetime import datetime
from pathlib import Path

from ..errors import LLMResumeError
from ..kernel.checkpoint import read_index
from ..kernel.client import KernelClient, KernelError
from ..kernel.executor import KernelExecutor
from ..messages import message_to_dict
from .claude_store import FolderSessionStore
from .events import EventLog
from .folder import ConversationFolder, default_root
from .history import load_history, save_history
from .recap import build_recap
from .restore_note import restore_messages

HISTORY_CHAR_BUDGET = 400_000
TITLE_LENGTH = 80


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _chain(first, second):
    if second is None:
        return first
    def both(x):
        first(x)
        second(x)
    return both


class Conversation:
    def __init__(self, session, folder: ConversationFolder, *, packages=None,
                 client_factory=KernelClient):
        self.session = session
        self.folder = folder
        self.events = EventLog(folder.events_path)
        self.restore_line: str | None = None
        self._pending_note: str | None = None
        self._restore_parts: tuple | None = None
        self.client = client_factory(packages,
                                     checkpoint_dir=folder.namespace_dir)
        next_cell = 1 + max((e["cell"] for e in self.events.read()
                             if e.get("type") == "cell"), default=0)
        self.executor = KernelExecutor(self.client, folder.figures_dir,
                                       first_cell=next_cell,
                                       on_cell=self._on_cell)
        session.executor = self.executor

    # ---- construction ----

    @classmethod
    def create(cls, session, root=None, *, now=None, packages=None,
               client_factory=KernelClient) -> "Conversation":
        folder = ConversationFolder.create(
            root if root is not None else default_root(), now=now)
        folder.update_meta(backend=session.backend.name, model=session.model)
        conv = cls(session, folder, packages=packages,
                   client_factory=client_factory)
        conv._start_backend_fresh()
        return conv

    @classmethod
    def open(cls, session, path, *, packages=None,
             client_factory=KernelClient) -> "Conversation":
        folder = ConversationFolder(Path(path))
        meta = folder.read_meta()
        conv = cls(session, folder, packages=packages,
                   client_factory=client_factory)
        events = conv.events.read()
        if events and events[-1].get("type") not in (
                "turn_end", "checkpoint", "restore"):
            conv.events.append({"type": "turn_end", "status": "interrupted"})
        context_restored = conv._restore_model_context(meta)
        recap = (build_recap(events)
                 if events and not context_restored else None)
        index = read_index(folder.namespace_dir)
        saved_version = (index or {}).get("fdp_core_version",
                                          meta.get("fdp_core_version"))
        conv._restore_parts = (index, saved_version,
                               os.environ.get("FDP_CORE_VERSION"))
        note, line = restore_messages(*conv._restore_parts, recap=recap)
        conv._pending_note, conv.restore_line = note, line
        conv.events.append({"type": "restore", "line": line,
                            "recap_used": recap is not None})
        folder.update_meta(backend=session.backend.name, model=session.model,
                           last_active=_now())
        return conv

    # ---- backend context ----

    def _is_persistent_backend(self) -> bool:
        return hasattr(self.session.backend, "configure_persistence")

    def _start_backend_fresh(self) -> None:
        if not self._is_persistent_backend():
            return
        sid = str(uuid.uuid4())
        self.session.backend.configure_persistence(
            store=FolderSessionStore(self.folder.claude_dir),
            cwd=self.folder.path, session_id=sid)
        self.folder.update_meta(claude_session_id=sid)

    def _restore_model_context(self, meta: dict) -> bool:
        """Restore what the model remembers; False means a recap is needed."""
        same_backend = meta.get("backend") == self.session.backend.name
        history = load_history(self.folder.history_path) if same_backend else None
        if self._is_persistent_backend():
            if history is not None:
                self.session.history = history
            store = FolderSessionStore(self.folder.claude_dir)
            sid = meta.get("claude_session_id")
            if same_backend and sid and store.has_transcript(sid):
                self.session.backend.configure_persistence(
                    store=store, cwd=self.folder.path, resume=sid)
                return True
            self._start_backend_fresh()
            return False
        if history is None:
            return False
        size = sum(len(json.dumps(message_to_dict(m))) for m in history)
        if size > HISTORY_CHAR_BUDGET:
            return False
        self.session.history = history
        return True

    # ---- event logging ----

    def _on_cell(self, record: dict) -> None:
        self.events.append({"type": "cell", **record})

    def _log_text(self, text: str) -> None:
        self._text_logged = True
        self.events.append({"type": "assistant_text", "text": text})

    def _log_tool_call(self, call) -> None:
        self.events.append({"type": "tool_call", "id": call.id,
                            "name": call.name, "args": call.args})

    def _log_tool_result(self, result) -> None:
        self.events.append({"type": "tool_result", "id": result.id,
                            "output": result.output,
                            "is_error": result.is_error})

    # ---- turns ----

    def send(self, prompt: str, *, on_text=None, on_tool_call=None,
             on_tool_result=None, on_event=None, confirm=None):
        if not self.folder.read_meta().get("title"):
            self.folder.update_meta(title=prompt[:TITLE_LENGTH])
        self.events.append({"type": "user", "text": prompt})
        callbacks = dict(
            on_text=_chain(self._log_text, on_text),
            on_tool_call=_chain(self._log_tool_call, on_tool_call),
            on_tool_result=_chain(self._log_tool_result, on_tool_result),
            on_event=on_event, confirm=confirm)
        model_prompt = (prompt if self._pending_note is None
                        else f"{self._pending_note}\n\n{prompt}")
        mark = len(self.session.history)
        self._text_logged = False
        try:
            try:
                result = self.session.send(model_prompt, **callbacks)
            except LLMResumeError:
                del self.session.history[mark:]
                model_prompt = self._fall_back_to_recap(prompt)
                result = self.session.send(model_prompt, **callbacks)
        except Exception as e:
            self.events.append({"type": "error",
                                "text": f"{type(e).__name__}: {e}"})
            self.events.append({"type": "turn_end", "status": "error"})
            raise
        self._pending_note = None
        if not self._text_logged and result.final_text:
            # Backends that do not stream (FakeBackend) never fire on_text.
            self._log_text(result.final_text)
        self.events.append({"type": "turn_end", "status": result.stop_reason})
        self._after_turn()
        return result

    def _fall_back_to_recap(self, prompt: str) -> str:
        self.session.backend.close()
        self._start_backend_fresh()
        earlier = self.events.read()[:-1]   # without the user line just logged
        index, saved, current = self._restore_parts or (None, None, None)
        note, line = restore_messages(index, saved, current,
                                      recap=build_recap(earlier))
        self.restore_line = line
        self.events.append({"type": "restore", "line": line,
                            "recap_used": True})
        return f"{note}\n\n{prompt}"

    def _after_turn(self) -> None:
        self.events.sync()
        save_history(self.folder.history_path, self.session.history)
        changes = {"last_active": _now()}
        sid = getattr(self.session.backend, "session_id", None)
        if sid:
            changes["claude_session_id"] = sid
        self.folder.update_meta(**changes)
        self.checkpoint()

    # ---- checkpoints and shutdown ----

    def checkpoint(self, wait: float | None = None) -> dict | None:
        """Checkpoint the variables; logs and returns the index (None if no worker).

        ``wait`` bounds how long to wait for a running cell; a cell still
        running then is logged as a failed checkpoint.
        """
        try:
            index = self.client.checkpoint(wait=wait)
        except KernelError as e:
            self.events.append({"type": "error",
                                "text": f"Checkpoint failed: {e}"})
            return None
        if index is not None:
            self.events.append({
                "type": "checkpoint",
                "saved": [v["name"] for v in index["variables"]
                          if v["status"] == "saved"],
                "not_saved": [v["name"] for v in index["variables"]
                              if v["status"] != "saved"],
                "bytes": index["total_bytes"]})
        return index

    def close(self) -> None:
        """Stop any running cell, checkpoint, stop the worker, disconnect.

        The checkpoint waits at most 30 s for the interrupted cell, so close()
        never hangs on a stuck cell. The Session stays usable for a new
        Conversation (its skills server is owned by the caller).
        """
        self.executor.interrupt()
        self.checkpoint(wait=30)
        self.events.sync()
        self.executor.close()
        if self.session.executor is self.executor:
            self.session.executor = None
        backend_close = getattr(self.session.backend, "close", None)
        if backend_close is not None:
            backend_close()
