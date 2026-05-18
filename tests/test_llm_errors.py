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
"""Tests for the LLMError hierarchy."""

import unittest

from toksearch.llm.errors import (
    LLMError,
    LLMConfigError,
    LLMAuthError,
    LLMBackendError,
    LLMRateLimitError,
    LLMUserAbort,
)


class TestErrorHierarchy(unittest.TestCase):
    def test_all_inherit_from_llm_error(self):
        for cls in (LLMConfigError, LLMAuthError, LLMBackendError,
                    LLMRateLimitError, LLMUserAbort):
            self.assertTrue(issubclass(cls, LLMError),
                            f"{cls.__name__} must inherit LLMError")

    def test_llm_error_inherits_from_exception(self):
        self.assertTrue(issubclass(LLMError, Exception))

    def test_user_abort_inherits_from_keyboard_interrupt(self):
        # LLMUserAbort is raised in response to user ctrl-C, so it should
        # also pass `except KeyboardInterrupt` for shells that explicitly
        # filter on that type.
        self.assertTrue(issubclass(LLMUserAbort, KeyboardInterrupt))

    def test_raise_and_catch_via_base(self):
        with self.assertRaises(LLMError):
            raise LLMAuthError("bad key")

    def test_message_is_preserved(self):
        try:
            raise LLMBackendError("503 service unavailable")
        except LLMBackendError as e:
            self.assertEqual(str(e), "503 service unavailable")


if __name__ == "__main__":
    unittest.main()
