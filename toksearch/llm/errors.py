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
"""Exception hierarchy for toksearch.llm.

All exceptions raised by toksearch.llm inherit from ``LLMError``.  ``LLMUserAbort``
also inherits from ``KeyboardInterrupt`` so REPL frames can use a single
``except KeyboardInterrupt`` to handle ctrl-C.
"""


class LLMError(Exception):
    """Base class for all toksearch.llm exceptions."""


class LLMConfigError(LLMError):
    """Bad configuration: unknown backend, unknown model, malformed preset."""


class LLMAuthError(LLMError):
    """Authentication failure: missing API key, invalid token, 401/403."""


class LLMBackendError(LLMError):
    """Backend returned an unrecoverable error: 5xx, network failure after retries."""


class LLMRateLimitError(LLMError):
    """Rate-limited by the provider; ``Retry-After`` exhausted."""


class LLMUserAbort(LLMError, KeyboardInterrupt):
    """User interrupted the conversation (e.g. ctrl-C between tool calls)."""
