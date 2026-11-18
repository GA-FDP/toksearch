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

"""A text recap of a conversation, rebuilt from events.jsonl.

Used when the model's own context cannot be resumed: the new model session
gets the user's messages, the assistant's replies and the code of each cell,
newest kept when over budget.
"""

RECAP_BUDGET = 30_000


def build_recap(events: list[dict], budget: int = RECAP_BUDGET) -> str:
    pieces = []
    for e in events:
        kind = e.get("type")
        if kind == "user":
            pieces.append(f"User: {e['text']}")
        elif kind == "assistant_text":
            pieces.append(f"Assistant: {e['text']}")
        elif kind == "cell":
            pieces.append(f"Cell {e['cell']}:\n```python\n{e['code']}\n```")
    kept, used = [], 0
    for p in reversed(pieces):
        if used + len(p) > budget:
            break
        kept.append(p)
        used += len(p) + 2
    kept.reverse()
    omitted = len(pieces) - len(kept)
    head = ("Summary of the earlier conversation, rebuilt from its log after "
            "a restart (oldest first")
    head += (f"; the {omitted} earliest items are omitted)." if omitted
             else ").")
    return head + "\n\n" + "\n\n".join(kept)
