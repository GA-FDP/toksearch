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
"""FakeBackend — testing seam for Session-level tests.

Subclasses ``_ToolLoopBackend`` so the standard loop drives it.  ``_send_request``
pops one scripted ``AssistantTurn`` per call.  Optionally records each
``_send_request`` invocation for inspection.
"""

from .base import _ToolLoopBackend, AssistantTurn


class FakeBackend(_ToolLoopBackend):
    name = "fake"
    default_model = "fake-1"

    def __init__(self, scripted_turns=None, record=None):
        self._turns = list(scripted_turns or [])
        self._record = record  # list to append call dicts to, or None
        self._user_message_pending = None

    def run_conversation(self, session, new_user_message, callbacks,
                          max_iterations):
        # Capture the user message so _send_request can record it for the
        # first call of this conversation.
        self._user_message_pending = new_user_message
        return super().run_conversation(session, new_user_message, callbacks,
                                          max_iterations)

    def _send_request(self, *, system_prompt, history, tools, model,
                      on_text=None) -> AssistantTurn:
        if not self._turns:
            raise RuntimeError("FakeBackend script exhausted")
        if self._record is not None:
            self._record.append({
                "system_prompt": system_prompt,
                "history": list(history),
                "tools": list(tools),
                "model": model,
                "user_message": self._user_message_pending,
            })
            # Only attribute the user_message to the first call per turn
            self._user_message_pending = None
        return self._turns.pop(0)
