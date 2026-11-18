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

"""history.jsonl: Session.history, one message per line, rewritten each turn."""

import json
import os
from pathlib import Path

from ..messages import dict_to_message, message_to_dict


def save_history(path, history) -> None:
    path = Path(path)
    tmp = path.with_suffix(".jsonl.tmp")
    with open(tmp, "w") as f:
        for m in history:
            f.write(json.dumps(message_to_dict(m)) + "\n")
    os.replace(tmp, path)


def load_history(path):
    path = Path(path)
    if not path.exists():
        return None
    return [dict_to_message(json.loads(line))
            for line in path.read_text().splitlines() if line.strip()]
