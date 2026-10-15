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

"""Live: a claude-max conversation survives close() and open().

Runs only with TOKSEARCH_LIVE_CLAUDE=1 and a logged-in `claude` CLI.
"""

import os
import shutil
import unittest
from tempfile import TemporaryDirectory

LIVE = os.environ.get("TOKSEARCH_LIVE_CLAUDE") == "1"


@unittest.skipUnless(LIVE, "set TOKSEARCH_LIVE_CLAUDE=1 to run")
class TestLiveResume(unittest.TestCase):
    def _session(self):
        from toksearch.llm.backends.claude_sdk import ClaudeSDKBackend
        from toksearch.llm.session import Session
        return Session(backend=ClaudeSDKBackend(), packages=[])

    def test_context_and_variables_survive_a_restart(self):
        from toksearch.llm.conversation import Conversation
        self.assertIsNotNone(shutil.which("claude"), "claude CLI not on PATH")
        with TemporaryDirectory() as tmp:
            s1 = self._session()
            c1 = Conversation.create(s1, tmp, packages=[])
            c1.send("Use run_python to run: secret_number = 7351. "
                    "Then remember that my favourite colour is teal. "
                    "Reply with just OK.")
            path = c1.folder.path
            c1.close()
            s1.close()
            self.assertTrue(any(path.joinpath("claude").glob("*.jsonl")))

            s2 = self._session()
            c2 = Conversation.open(s2, path, packages=[])
            self.addCleanup(s2.close)
            self.addCleanup(c2.close)
            self.assertNotIn("summary", c2.restore_line)
            r = c2.send("Without guessing: what is my favourite colour, and "
                        "what does run_python report for secret_number + 1? "
                        "Answer as '<colour> <number>'.")
            self.assertIn("teal", r.final_text.lower())
            self.assertIn("7352", r.final_text)
            restores = [e for e in c2.events.read()
                        if e["type"] == "restore"]
            self.assertFalse(any(e["recap_used"] for e in restores))


if __name__ == "__main__":
    unittest.main()
