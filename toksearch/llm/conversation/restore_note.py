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

"""What a restored conversation tells the model (hidden) and the user (one line)."""

from ..kernel.checkpoint import describe_not_restored, not_saved


def restore_messages(index: dict | None, saved_version: str | None,
                     current_version: str | None,
                     recap: str | None = None) -> tuple[str, str]:
    """Return (note prefixed to the next prompt, line shown in the chat)."""
    note = ["[Note from the app, not the user: this conversation was "
            "restored after a restart."]
    line = ["Conversation restored."]
    if index is None:
        note.append("No Python variables had been saved.")
        line.append("No variables saved.")
    else:
        saved = [e["name"] for e in index["variables"]
                 if e["status"] == "saved"]
        note.append("Python variables available: "
                    f"{', '.join(saved) if saved else 'none'}.")
        line.append(f"{len(saved)} variable{'' if len(saved) == 1 else 's'} "
                    "restored.")
        missing = not_saved(index)
        if missing:
            desc = describe_not_restored(missing)
            note.append(f"Not restored: {desc}. Re-run those cells if they "
                        "are needed.")
            line.append(f"Not restored: {desc}.")
    if saved_version and current_version and saved_version != current_version:
        changed = (f"Environment changed: fdp-core {saved_version} → "
                   f"{current_version}.")
        note.append(changed)
        line.append(changed)
    if recap:
        note.append("Your own record of the conversation could not be "
                    "restored; a summary follows.")
        line.append("The model's memory was rebuilt from a summary.")
    text = " ".join(note) + "]"
    if recap:
        text += "\n\n" + recap
    return text, " ".join(line)
