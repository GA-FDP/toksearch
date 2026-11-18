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

"""events.jsonl: an append-only log of everything shown in a conversation.

Each event is one JSON line, written as it happens with a single write()
on an O_APPEND descriptor, so lines from concurrent threads never
interleave and a crash loses at most the line being written; read() skips
a truncated last line.
Event types: user, assistant_text, tool_call, tool_result, cell, turn_end,
checkpoint, restore, error.
"""

import json
import os
import threading
import time
from pathlib import Path


class EventLog:
    def __init__(self, path):
        self.path = Path(path)
        self._lock = threading.Lock()

    def append(self, event: dict) -> dict:
        event = {"t": time.time(), **event}
        data = (json.dumps(event, default=str) + "\n").encode()
        with self._lock:
            fd = os.open(self.path, os.O_WRONLY | os.O_APPEND | os.O_CREAT,
                         0o644)
            try:
                view = memoryview(data)
                while view:     # one write unless the kernel takes less
                    view = view[os.write(fd, view):]
            finally:
                os.close(fd)
        return event

    def sync(self) -> None:
        if self.path.exists():
            with open(self.path, "a") as f:
                os.fsync(f.fileno())

    def read(self) -> list[dict]:
        if not self.path.exists():
            return []
        out = []
        for line in self.path.read_text().splitlines():
            if not line.strip():
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return out
