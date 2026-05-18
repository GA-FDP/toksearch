# `toksearch.llm` PR 1 — Library Core Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship the `toksearch.llm` library: a sync, callback-driven `Session` API over Anthropic and OpenAI backends, with `run_python` (persistent namespace) and `lookup_docs` tools, a minimal `toksearch chat` / `toksearch query` CLI, and end-to-end test coverage via a `FakeBackend`. This is PR 1 of the migration described in the design doc.

**Architecture:** New `toksearch/llm/` subpackage. State is owned by `Session`; backends are stateless adapters that implement `run_conversation(session, prompt, callbacks, max_iterations) -> TurnComplete`. `_ToolLoopBackend` holds the shared tool-use loop for Anthropic and OpenAI; subclasses translate to/from native shapes. `FakeBackend` returns scripted assistant turns and is the testing seam for Session-level tests. Tools are backend-agnostic `ToolSpec`s registered with the Session at construction; `run_python` execs against `session.namespace`; `lookup_docs` reads `SKILL.md` files from `extra_skill_dirs`. CLI is a thin argparse wrapper installed as the `toksearch` console script.

**Tech Stack:** Python 3.11, `anthropic`, `openai`, `unittest`, `unittest.mock`, `argparse`, `tomllib` (stdlib), pixi-managed env. Apache 2.0 license header on every new file.

**Reference spec:** `docs/superpowers/specs/2026-05-18-toksearch-llm-design.md`

## Scope notes — what's deferred from the spec

To keep PR 1 reviewable, the following spec items are explicitly deferred to follow-up PRs (call them out in the PR description):

- **Streaming text deltas.** `_ToolLoopBackend` calls the blocking, non-streaming API; `on_text` fires once per assistant turn with the complete text. OpenAI's tool-arg streaming reassembly is not needed yet.
- **`prompt_toolkit`-based REPL.** `toksearch chat` in PR 1 uses stdlib `input()` with a flat prompt. Multi-line input, history, and key bindings come later.
- **Slash commands `/save`, `/history`, `/namespace`, `/backend`, `/model`.** Only `/help`, `/reset`, `/quit` ship in PR 1.
- **`--confirm` flag.** Auto-approve only in PR 1; `confirm` callable on `Session.send()` is wired through and tested, but no CLI flag yet.
- **`toksearch llm test` connectivity check.** Deferred.
- **`cache_control` on Anthropic system prompt.** Deferred (small optimization).
- **Entry-point discovery (`toksearch.llm.namespace` / `.skills` / `.presets`).** Deferred to PR 2. In PR 1, `Session` takes explicit `packages=` and `extra_skill_dirs=` keyword arguments; `packages` is currently unused (defaults to `["toksearch"]` which is hardcoded into the namespace).
- **`ClaudeSDKBackend`.** Deferred to PR 3.

All deferred items have placeholder seams in PR 1 so adding them later is additive, not a refactor.

---

## File Structure

| File | Action | Purpose |
|---|---|---|
| `toksearch/llm/__init__.py` | Create | Public re-exports: `Session`, event types, `LLMError`, `Config`, `Preset`. |
| `toksearch/llm/errors.py` | Create | `LLMError` and subclasses. |
| `toksearch/llm/events.py` | Create | `TextDelta`, `ToolCall`, `ToolResult`, `TurnComplete` frozen dataclasses. |
| `toksearch/llm/messages.py` | Create | Provider-neutral history: `Message`, `TextBlock`, `ToolUseBlock`, `ToolResultBlock`. |
| `toksearch/llm/config.py` | Create | `Config` dataclass + `load_config()` (TOML + env overlay). |
| `toksearch/llm/presets.py` | Create | `Preset` dataclass + built-in `anthropic` / `openai` presets + `resolve_preset()`. |
| `toksearch/llm/tools.py` | Create | `ToolSpec`, `ToolOutput`, `run_python` handler, `lookup_docs` handler, skill discovery helpers. |
| `toksearch/llm/prompts.py` | Create | Kernel system-prompt template + `build_system_prompt(skills, namespace_extras)`. |
| `toksearch/llm/backends/__init__.py` | Create | Backend registry + `get_backend(name, config)`. |
| `toksearch/llm/backends/base.py` | Create | `Backend` ABC, `_ToolLoopBackend`, `AssistantTurn`, `Callbacks` dataclasses. |
| `toksearch/llm/backends/anthropic.py` | Create | `AnthropicBackend` (subclass of `_ToolLoopBackend`). |
| `toksearch/llm/backends/openai.py` | Create | `OpenAIBackend` (subclass of `_ToolLoopBackend`). |
| `toksearch/llm/backends/fake.py` | Create | `FakeBackend` — scripted turn returner; testing seam. |
| `toksearch/llm/session.py` | Create | `Session` class — orchestrates state, dispatches to backend, manages namespace. |
| `toksearch/llm/cli.py` | Create | `toksearch chat` + `toksearch query` argparse entrypoint. |
| `pyproject.toml` | Modify | Add `[project.optional-dependencies] llm = [...]` and `[project.scripts] toksearch = ...`. |
| `setup.py` | Modify | Add `llm` subpackages to the package list; add `entry_points` for the console script. |
| `tests/test_llm_errors.py` | Create | Tests for `LLMError` hierarchy. |
| `tests/test_llm_events.py` | Create | Tests for event dataclasses. |
| `tests/test_llm_messages.py` | Create | Tests for `Message` / `ContentBlock` round-trip. |
| `tests/test_llm_config.py` | Create | Tests for `load_config()` precedence. |
| `tests/test_llm_presets.py` | Create | Tests for preset resolution. |
| `tests/test_llm_tools.py` | Create | Tests for `run_python` (namespace persistence, error capture, KeyboardInterrupt) and `lookup_docs`. |
| `tests/test_llm_prompts.py` | Create | Tests for system-prompt assembly. |
| `tests/test_llm_backends_base.py` | Create | Tests for `_ToolLoopBackend` loop behavior via a minimal subclass. |
| `tests/test_llm_backends_fake.py` | Create | Tests for `FakeBackend`. |
| `tests/test_llm_backends_anthropic.py` | Create | Tests for `AnthropicBackend` with `anthropic.Anthropic` mocked. |
| `tests/test_llm_backends_openai.py` | Create | Tests for `OpenAIBackend` with `openai.OpenAI` mocked. |
| `tests/test_llm_session.py` | Create | Session-level tests with `FakeBackend`. |
| `tests/test_llm_cli.py` | Create | CLI subcommand wiring + slash commands; Session mocked. |
| `tests/test_llm_package.py` | Create | Confirms `from toksearch.llm import Session, ...` works and console-script entry resolves. |

Test runner: existing `tests/testit.py` uses `unittest.TestLoader().discover(".")`; any file named `test_*.py` is auto-discovered. Tests must not require network. All LLM SDK calls are mocked.

---

## Task 1: Create the `toksearch.llm` package skeleton

**Files:**
- Create: `toksearch/llm/__init__.py`
- Create: `toksearch/llm/backends/__init__.py`
- Create: `tests/test_llm_package.py`

This task establishes the package directory structure so subsequent tasks have a place to put their files. The `__init__.py` files are empty placeholders that will get real re-exports in Task 17.

- [ ] **Step 1: Create `toksearch/llm/__init__.py`**

```python
# Copyright 2024 General Atomics
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""toksearch.llm — Conversational LLM interface for TokSearch.

Public re-exports are wired up at the end of PR 1; see
``docs/superpowers/specs/2026-05-18-toksearch-llm-design.md``.
"""
```

- [ ] **Step 2: Create `toksearch/llm/backends/__init__.py`**

```python
# Copyright 2024 General Atomics
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""Backend implementations for toksearch.llm."""
```

- [ ] **Step 3: Write a smoke test that the package is importable**

Create `tests/test_llm_package.py`:

```python
# Copyright 2024 General Atomics
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Smoke test: the toksearch.llm package and its backends subpackage import."""

import unittest


class TestPackageImports(unittest.TestCase):
    def test_import_toksearch_llm(self):
        import toksearch.llm  # noqa: F401

    def test_import_toksearch_llm_backends(self):
        import toksearch.llm.backends  # noqa: F401


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 4: Run the smoke test**

```bash
pixi run bash -c 'cd tests && python -m unittest test_llm_package -v'
```

Expected: 2 tests pass. If `import toksearch.llm` fails with `ModuleNotFoundError`, the editable install hasn't picked up the new subpackage — re-run `pixi run build` to refresh.

- [ ] **Step 5: Commit**

```bash
git add toksearch/llm/__init__.py toksearch/llm/backends/__init__.py tests/test_llm_package.py
git commit -m "Add toksearch.llm package skeleton"
```

---

## Task 2: Add the `LLMError` hierarchy

**Files:**
- Create: `toksearch/llm/errors.py`
- Create: `tests/test_llm_errors.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_llm_errors.py`:

```python
# Copyright 2024 General Atomics
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

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
```

- [ ] **Step 2: Run the tests and verify they fail**

```bash
pixi run bash -c 'cd tests && python -m unittest test_llm_errors -v'
```

Expected: `ModuleNotFoundError: No module named 'toksearch.llm.errors'`.

- [ ] **Step 3: Implement `errors.py`**

Create `toksearch/llm/errors.py`:

```python
# Copyright 2024 General Atomics
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

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
```

- [ ] **Step 4: Run the tests and verify they pass**

```bash
pixi run bash -c 'cd tests && python -m unittest test_llm_errors -v'
```

Expected: 5 tests pass.

- [ ] **Step 5: Commit**

```bash
git add toksearch/llm/errors.py tests/test_llm_errors.py
git commit -m "Add LLMError exception hierarchy"
```

---

## Task 3: Add event dataclasses

**Files:**
- Create: `toksearch/llm/events.py`
- Create: `tests/test_llm_events.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_llm_events.py`:

```python
# Copyright 2024 General Atomics
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Tests for the event dataclasses dispatched by Session.send()."""

import unittest

from toksearch.llm.events import (
    TextDelta,
    ToolCall,
    ToolResult,
    TurnComplete,
)


class TestEvents(unittest.TestCase):
    def test_text_delta_holds_text(self):
        e = TextDelta(text="hello")
        self.assertEqual(e.text, "hello")

    def test_tool_call_fields(self):
        e = ToolCall(id="abc", name="run_python",
                     args={"code": "x = 1", "thought": "set x"},
                     thought="set x")
        self.assertEqual(e.id, "abc")
        self.assertEqual(e.name, "run_python")
        self.assertEqual(e.args, {"code": "x = 1", "thought": "set x"})
        self.assertEqual(e.thought, "set x")

    def test_tool_call_thought_optional(self):
        e = ToolCall(id="abc", name="lookup_docs",
                     args={"skill_name": "toksearch-pipeline"},
                     thought=None)
        self.assertIsNone(e.thought)

    def test_tool_result_fields(self):
        e = ToolResult(id="abc", output="42", is_error=False)
        self.assertFalse(e.is_error)
        self.assertEqual(e.output, "42")

    def test_turn_complete_fields(self):
        e = TurnComplete(stop_reason="end_turn", final_text="done")
        self.assertEqual(e.stop_reason, "end_turn")
        self.assertEqual(e.final_text, "done")

    def test_events_are_frozen(self):
        e = TextDelta(text="hi")
        with self.assertRaises(Exception):  # FrozenInstanceError or AttributeError
            e.text = "bye"

    def test_events_compare_by_value(self):
        a = TextDelta(text="hi")
        b = TextDelta(text="hi")
        c = TextDelta(text="bye")
        self.assertEqual(a, b)
        self.assertNotEqual(a, c)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run tests and verify they fail**

```bash
pixi run bash -c 'cd tests && python -m unittest test_llm_events -v'
```

Expected: `ModuleNotFoundError: No module named 'toksearch.llm.events'`.

- [ ] **Step 3: Implement `events.py`**

Create `toksearch/llm/events.py`:

```python
# Copyright 2024 General Atomics
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Event dataclasses dispatched to Session.send() callbacks.

Events are frozen and compare by value.  They are emitted by the active
``Backend`` while the conversation advances; ``Session.send()`` routes them
to the appropriate ``on_<kind>`` callback (and to ``on_event`` if provided).
"""

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class TextDelta:
    """An incremental (or whole, in non-streaming backends) chunk of assistant text."""

    text: str


@dataclass(frozen=True)
class ToolCall:
    """The assistant is requesting a tool invocation.

    ``thought`` is populated for ``run_python`` (its schema requires a
    ``thought`` field) and is ``None`` for tools without one.
    """

    id: str
    name: str
    args: dict
    thought: str | None


@dataclass(frozen=True)
class ToolResult:
    """The output of a tool invocation, about to be sent back to the model."""

    id: str
    output: str
    is_error: bool


@dataclass(frozen=True)
class TurnComplete:
    """The assistant has finished this turn.

    ``stop_reason``:
    - ``"end_turn"``: assistant stopped emitting tool calls.
    - ``"max_iterations"``: hit ``Session.max_iterations`` before end_turn.
    - ``"interrupted"``: user aborted (ctrl-C) or ``confirm()`` returned False.
    """

    stop_reason: Literal["end_turn", "max_iterations", "interrupted"]
    final_text: str


Event = TextDelta | ToolCall | ToolResult | TurnComplete
```

- [ ] **Step 4: Run tests and verify they pass**

```bash
pixi run bash -c 'cd tests && python -m unittest test_llm_events -v'
```

Expected: 7 tests pass.

- [ ] **Step 5: Commit**

```bash
git add toksearch/llm/events.py tests/test_llm_events.py
git commit -m "Add event dataclasses for Session callbacks"
```

---

## Task 4: Add `Message` and `ContentBlock` types

**Files:**
- Create: `toksearch/llm/messages.py`
- Create: `tests/test_llm_messages.py`

Provider-neutral history representation.  Backends translate to/from their native
shapes on the boundary; the Session and tests see only this taxonomy.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_llm_messages.py`:

```python
# Copyright 2024 General Atomics
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Tests for the provider-neutral Message / ContentBlock types."""

import unittest

from toksearch.llm.messages import (
    Message,
    TextBlock,
    ToolUseBlock,
    ToolResultBlock,
    block_to_dict,
    dict_to_block,
    message_to_dict,
    dict_to_message,
)


class TestContentBlocks(unittest.TestCase):
    def test_text_block_kind(self):
        b = TextBlock(text="hi")
        self.assertEqual(b.kind, "text")
        self.assertEqual(b.text, "hi")

    def test_tool_use_block_kind(self):
        b = ToolUseBlock(id="t1", name="run_python", args={"code": "1"})
        self.assertEqual(b.kind, "tool_use")

    def test_tool_result_block_kind(self):
        b = ToolResultBlock(tool_use_id="t1", output="1", is_error=False)
        self.assertEqual(b.kind, "tool_result")


class TestMessage(unittest.TestCase):
    def test_user_text_message(self):
        m = Message(role="user", content=[TextBlock(text="hello")])
        self.assertEqual(m.role, "user")
        self.assertEqual(len(m.content), 1)

    def test_assistant_mixed_content(self):
        m = Message(role="assistant", content=[
            TextBlock(text="let me run code"),
            ToolUseBlock(id="t1", name="run_python",
                         args={"code": "x = 1", "thought": "set x"}),
        ])
        self.assertEqual(len(m.content), 2)


class TestSerialization(unittest.TestCase):
    """Round-trip support for future /save persistence."""

    def test_text_block_round_trip(self):
        b = TextBlock(text="hi")
        d = block_to_dict(b)
        self.assertEqual(d, {"kind": "text", "text": "hi"})
        self.assertEqual(dict_to_block(d), b)

    def test_tool_use_block_round_trip(self):
        b = ToolUseBlock(id="t1", name="run_python", args={"code": "x"})
        d = block_to_dict(b)
        self.assertEqual(d, {"kind": "tool_use", "id": "t1",
                             "name": "run_python", "args": {"code": "x"}})
        self.assertEqual(dict_to_block(d), b)

    def test_tool_result_block_round_trip(self):
        b = ToolResultBlock(tool_use_id="t1", output="ok", is_error=False)
        d = block_to_dict(b)
        self.assertEqual(d, {"kind": "tool_result", "tool_use_id": "t1",
                             "output": "ok", "is_error": False})
        self.assertEqual(dict_to_block(d), b)

    def test_message_round_trip(self):
        m = Message(role="assistant", content=[
            TextBlock(text="a"),
            ToolUseBlock(id="t1", name="run_python", args={"code": "1"}),
        ])
        d = message_to_dict(m)
        self.assertEqual(d["role"], "assistant")
        self.assertEqual(len(d["content"]), 2)
        self.assertEqual(dict_to_message(d), m)

    def test_dict_to_block_rejects_unknown_kind(self):
        with self.assertRaises(ValueError):
            dict_to_block({"kind": "bogus"})


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run tests and verify they fail**

```bash
pixi run bash -c 'cd tests && python -m unittest test_llm_messages -v'
```

Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Implement `messages.py`**

Create `toksearch/llm/messages.py`:

```python
# Copyright 2024 General Atomics
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Provider-neutral conversation history representation.

``Message`` is one turn in the history.  Its ``content`` is a list of
``ContentBlock``s — a tagged union (``kind`` field) of text, tool-use, and
tool-result blocks.  Backends translate to and from their native shapes on
their request/response boundary; the Session and tests see only this taxonomy.

``*_to_dict`` and ``dict_to_*`` helpers provide JSON-safe round-trip so
``/save`` (future) can serialize the history without provider-specific objects.
"""

from dataclasses import dataclass, field
from typing import Literal, Union


@dataclass
class TextBlock:
    text: str
    kind: Literal["text"] = "text"


@dataclass
class ToolUseBlock:
    id: str
    name: str
    args: dict
    kind: Literal["tool_use"] = "tool_use"


@dataclass
class ToolResultBlock:
    tool_use_id: str
    output: str
    is_error: bool
    kind: Literal["tool_result"] = "tool_result"


ContentBlock = Union[TextBlock, ToolUseBlock, ToolResultBlock]


@dataclass
class Message:
    role: Literal["user", "assistant"]
    content: list[ContentBlock] = field(default_factory=list)


def block_to_dict(b: ContentBlock) -> dict:
    if isinstance(b, TextBlock):
        return {"kind": "text", "text": b.text}
    if isinstance(b, ToolUseBlock):
        return {"kind": "tool_use", "id": b.id, "name": b.name, "args": b.args}
    if isinstance(b, ToolResultBlock):
        return {"kind": "tool_result", "tool_use_id": b.tool_use_id,
                "output": b.output, "is_error": b.is_error}
    raise ValueError(f"Unknown block type: {type(b).__name__}")


def dict_to_block(d: dict) -> ContentBlock:
    kind = d.get("kind")
    if kind == "text":
        return TextBlock(text=d["text"])
    if kind == "tool_use":
        return ToolUseBlock(id=d["id"], name=d["name"], args=d["args"])
    if kind == "tool_result":
        return ToolResultBlock(tool_use_id=d["tool_use_id"],
                               output=d["output"], is_error=d["is_error"])
    raise ValueError(f"Unknown block kind: {kind!r}")


def message_to_dict(m: Message) -> dict:
    return {"role": m.role,
            "content": [block_to_dict(b) for b in m.content]}


def dict_to_message(d: dict) -> Message:
    return Message(role=d["role"],
                   content=[dict_to_block(b) for b in d["content"]])
```

- [ ] **Step 4: Run tests and verify they pass**

```bash
pixi run bash -c 'cd tests && python -m unittest test_llm_messages -v'
```

Expected: 10 tests pass.

- [ ] **Step 5: Commit**

```bash
git add toksearch/llm/messages.py tests/test_llm_messages.py
git commit -m "Add provider-neutral Message and ContentBlock types"
```

---

## Task 5: Add `Config` and `load_config()`

**Files:**
- Create: `toksearch/llm/config.py`
- Create: `tests/test_llm_config.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_llm_config.py`:

```python
# Copyright 2024 General Atomics
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Tests for Config and load_config()."""

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from toksearch.llm.config import Config, load_config


class TestConfigDefaults(unittest.TestCase):
    def test_default_backend_is_none(self):
        cfg = Config()
        self.assertIsNone(cfg.backend)

    def test_default_max_iterations(self):
        cfg = Config()
        self.assertEqual(cfg.max_iterations, 20)

    def test_anthropic_key_default(self):
        cfg = Config()
        self.assertIsNone(cfg.anthropic_api_key)


class TestLoadConfig(unittest.TestCase):
    def setUp(self):
        # Isolate env so test order doesn't matter
        self._saved_env = {k: os.environ.pop(k, None)
                           for k in ("FDP_LLM_BACKEND",
                                     "ANTHROPIC_API_KEY",
                                     "OPENAI_API_KEY")}

    def tearDown(self):
        for k, v in self._saved_env.items():
            if v is not None:
                os.environ[k] = v

    def test_no_file_no_env_returns_defaults(self):
        cfg = load_config(config_path=Path("/does/not/exist"))
        self.assertIsNone(cfg.backend)
        self.assertEqual(cfg.max_iterations, 20)

    def test_loads_from_toml(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write('[llm]\nbackend = "openai"\nmax_iterations = 5\n'
                    'anthropic_api_key = "sk-file"\n')
            path = Path(f.name)
        try:
            cfg = load_config(config_path=path)
            self.assertEqual(cfg.backend, "openai")
            self.assertEqual(cfg.max_iterations, 5)
            self.assertEqual(cfg.anthropic_api_key, "sk-file")
        finally:
            path.unlink()

    def test_env_overrides_file(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write('[llm]\nbackend = "openai"\nanthropic_api_key = "sk-file"\n')
            path = Path(f.name)
        try:
            with mock.patch.dict(os.environ, {"FDP_LLM_BACKEND": "anthropic",
                                              "ANTHROPIC_API_KEY": "sk-env"}):
                cfg = load_config(config_path=path)
            self.assertEqual(cfg.backend, "anthropic")
            self.assertEqual(cfg.anthropic_api_key, "sk-env")
        finally:
            path.unlink()

    def test_malformed_toml_raises_config_error(self):
        from toksearch.llm.errors import LLMConfigError
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write("this is not [valid toml")
            path = Path(f.name)
        try:
            with self.assertRaises(LLMConfigError):
                load_config(config_path=path)
        finally:
            path.unlink()


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run tests and verify they fail**

```bash
pixi run bash -c 'cd tests && python -m unittest test_llm_config -v'
```

Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Implement `config.py`**

Create `toksearch/llm/config.py`:

```python
# Copyright 2024 General Atomics
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Configuration object and loader for toksearch.llm.

Precedence (highest first):
  1. CLI flags (applied by callers after ``load_config()`` returns)
  2. Environment variables: ``FDP_LLM_BACKEND``, ``ANTHROPIC_API_KEY``,
     ``OPENAI_API_KEY``
  3. ``~/.fdp/config.toml`` (``[llm]`` table)
  4. Built-in defaults
"""

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from .errors import LLMConfigError


@dataclass
class Config:
    backend: str | None = None          # preset name; None → caller's default
    model: str | None = None            # overrides preset's default model
    max_iterations: int = 20
    anthropic_api_key: str | None = None
    openai_api_key: str | None = None
    user_presets: dict = field(default_factory=dict)


_DEFAULT_CONFIG_PATH = Path.home() / ".fdp" / "config.toml"


def load_config(config_path: Path | None = None) -> Config:
    cfg = Config()
    path = config_path if config_path is not None else _DEFAULT_CONFIG_PATH
    if path.exists():
        try:
            with path.open("rb") as f:
                data = tomllib.load(f)
        except tomllib.TOMLDecodeError as e:
            raise LLMConfigError(f"Malformed TOML in {path}: {e}") from e
        llm = data.get("llm", {})
        if "backend" in llm:
            cfg.backend = llm["backend"]
        if "model" in llm:
            cfg.model = llm["model"]
        if "max_iterations" in llm:
            cfg.max_iterations = int(llm["max_iterations"])
        if "anthropic_api_key" in llm:
            cfg.anthropic_api_key = llm["anthropic_api_key"]
        if "openai_api_key" in llm:
            cfg.openai_api_key = llm["openai_api_key"]
        cfg.user_presets = llm.get("presets", {})
    # Env overlay
    if "FDP_LLM_BACKEND" in os.environ:
        cfg.backend = os.environ["FDP_LLM_BACKEND"]
    if "ANTHROPIC_API_KEY" in os.environ:
        cfg.anthropic_api_key = os.environ["ANTHROPIC_API_KEY"]
    if "OPENAI_API_KEY" in os.environ:
        cfg.openai_api_key = os.environ["OPENAI_API_KEY"]
    return cfg
```

- [ ] **Step 4: Run tests and verify they pass**

```bash
pixi run bash -c 'cd tests && python -m unittest test_llm_config -v'
```

Expected: 6 tests pass.

- [ ] **Step 5: Commit**

```bash
git add toksearch/llm/config.py tests/test_llm_config.py
git commit -m "Add Config dataclass and load_config() with env overlay"
```

---

## Task 6: Add `Preset` and preset resolution

**Files:**
- Create: `toksearch/llm/presets.py`
- Create: `tests/test_llm_presets.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_llm_presets.py`:

```python
# Copyright 2024 General Atomics
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Tests for Preset resolution."""

import unittest

from toksearch.llm.config import Config
from toksearch.llm.errors import LLMConfigError
from toksearch.llm.presets import Preset, resolve_preset, BUILTIN_PRESETS


class TestPreset(unittest.TestCase):
    def test_builtin_anthropic_exists(self):
        self.assertIn("anthropic", BUILTIN_PRESETS)
        p = BUILTIN_PRESETS["anthropic"]
        self.assertEqual(p.backend, "anthropic")
        self.assertEqual(p.api_key_env, "ANTHROPIC_API_KEY")

    def test_builtin_openai_exists(self):
        self.assertIn("openai", BUILTIN_PRESETS)
        p = BUILTIN_PRESETS["openai"]
        self.assertEqual(p.backend, "openai")
        self.assertEqual(p.api_key_env, "OPENAI_API_KEY")


class TestResolvePreset(unittest.TestCase):
    def test_resolves_builtin(self):
        p = resolve_preset("anthropic", Config())
        self.assertEqual(p.backend, "anthropic")

    def test_unknown_preset_raises(self):
        with self.assertRaises(LLMConfigError):
            resolve_preset("nonexistent", Config())

    def test_user_preset_overrides_builtin(self):
        cfg = Config(user_presets={"anthropic": {"model": "custom-model"}})
        p = resolve_preset("anthropic", cfg)
        # User preset is shallow-merged onto built-in; model is overridden.
        self.assertEqual(p.model, "custom-model")
        # Other fields fall through from built-in.
        self.assertEqual(p.backend, "anthropic")

    def test_user_only_preset(self):
        cfg = Config(user_presets={"mysite": {"backend": "anthropic",
                                              "base_url": "https://x.example",
                                              "model": "claude-sonnet-4-6",
                                              "api_key_env": "MY_KEY"}})
        p = resolve_preset("mysite", cfg)
        self.assertEqual(p.backend, "anthropic")
        self.assertEqual(p.base_url, "https://x.example")
        self.assertEqual(p.api_key_env, "MY_KEY")


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run tests and verify they fail**

```bash
pixi run bash -c 'cd tests && python -m unittest test_llm_presets -v'
```

Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Implement `presets.py`**

Create `toksearch/llm/presets.py`:

```python
# Copyright 2024 General Atomics
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Backend presets.

A ``Preset`` is a named configuration that maps a user-facing backend name
(e.g. ``"anthropic"``, ``"amsc"``, ``"openai"``) to a concrete ``Backend``
class plus the kwargs needed to construct it.  Presets exist so that
site-specific endpoints (AmSC's Anthropic-compatible URL, etc.) can be added
without subclassing ``AnthropicBackend`` just to change a ``base_url``.

Built-in presets ship with core toksearch.  User presets come from
``~/.fdp/config.toml``'s ``[llm.presets.<name>]`` tables and are merged in via
``Config.user_presets``.  In PR 2, package-contributed presets will also be
discovered via the ``toksearch.llm.presets`` entry point.
"""

from dataclasses import dataclass, field, replace

from .config import Config
from .errors import LLMConfigError


@dataclass(frozen=True)
class Preset:
    backend: str                # "anthropic" | "openai" | "claude-max"
    model: str | None = None
    base_url: str | None = None
    api_key_file: str | None = None
    api_key_env: str | None = None
    extra: dict = field(default_factory=dict)


BUILTIN_PRESETS: dict[str, Preset] = {
    "anthropic": Preset(
        backend="anthropic",
        model="claude-sonnet-4-6",
        api_key_env="ANTHROPIC_API_KEY",
    ),
    "openai": Preset(
        backend="openai",
        model="gpt-4o",
        api_key_env="OPENAI_API_KEY",
    ),
}


def resolve_preset(name: str, config: Config) -> Preset:
    """Resolve a preset name to a fully-populated ``Preset``.

    Lookup order: user presets > built-in presets.  When a preset name exists
    in both, the user preset's fields are merged on top of the built-in.
    """
    user = config.user_presets.get(name, {})
    if name in BUILTIN_PRESETS:
        base = BUILTIN_PRESETS[name]
        if user:
            return replace(base, **{k: v for k, v in user.items()
                                    if k in {"backend", "model", "base_url",
                                             "api_key_file", "api_key_env"}})
        return base
    if user:
        try:
            return Preset(**{k: v for k, v in user.items()
                             if k in {"backend", "model", "base_url",
                                      "api_key_file", "api_key_env", "extra"}})
        except TypeError as e:
            raise LLMConfigError(f"Invalid user preset {name!r}: {e}") from e
    raise LLMConfigError(
        f"Unknown backend / preset: {name!r}. Built-in presets: "
        f"{sorted(BUILTIN_PRESETS)}. Define a user preset in "
        f"~/.fdp/config.toml under [llm.presets.{name}].")
```

- [ ] **Step 4: Run tests and verify they pass**

```bash
pixi run bash -c 'cd tests && python -m unittest test_llm_presets -v'
```

Expected: 6 tests pass.

- [ ] **Step 5: Commit**

```bash
git add toksearch/llm/presets.py tests/test_llm_presets.py
git commit -m "Add Preset dataclass and resolution with built-in presets"
```

---

## Task 7: Add `ToolSpec`, `ToolOutput`, and the `run_python` handler

**Files:**
- Create: `toksearch/llm/tools.py`
- Create: `tests/test_llm_tools.py` (only the `run_python` sections — `lookup_docs` is added in Task 8)

This task adds the tool type scaffolding plus the first tool handler.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_llm_tools.py`:

```python
# Copyright 2024 General Atomics
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Tests for tools.py — ToolSpec, ToolOutput, and tool handlers.

Tool handlers take ``(args: dict, session)``.  These tests pass a duck-typed
stub with a ``.namespace`` attribute; the real ``Session`` class is tested in
``test_llm_session.py``.
"""

import unittest
from types import SimpleNamespace

from toksearch.llm.tools import (
    ToolSpec,
    ToolOutput,
    RUN_PYTHON,
)


def _stub_session(namespace: dict | None = None):
    return SimpleNamespace(namespace=namespace if namespace is not None else {})


class TestToolSpec(unittest.TestCase):
    def test_run_python_spec_shape(self):
        self.assertEqual(RUN_PYTHON.name, "run_python")
        self.assertIn("code", RUN_PYTHON.input_schema["properties"])
        self.assertIn("thought", RUN_PYTHON.input_schema["properties"])
        self.assertEqual(set(RUN_PYTHON.input_schema["required"]),
                         {"code", "thought"})


class TestRunPython(unittest.TestCase):
    def test_simple_expression_no_output(self):
        s = _stub_session()
        out = RUN_PYTHON.handler({"code": "x = 1 + 1", "thought": "test"}, s)
        self.assertFalse(out.is_error)
        self.assertEqual(s.namespace["x"], 2)

    def test_stdout_captured(self):
        s = _stub_session()
        out = RUN_PYTHON.handler({"code": "print('hi')", "thought": "x"}, s)
        self.assertFalse(out.is_error)
        self.assertIn("hi", out.text)

    def test_stderr_captured(self):
        s = _stub_session()
        out = RUN_PYTHON.handler({"code": "import sys; sys.stderr.write('warn\\n')",
                                  "thought": "x"}, s)
        self.assertFalse(out.is_error)
        self.assertIn("warn", out.text)

    def test_exception_becomes_error(self):
        s = _stub_session()
        out = RUN_PYTHON.handler({"code": "1/0", "thought": "boom"}, s)
        self.assertTrue(out.is_error)
        self.assertIn("ZeroDivisionError", out.text)
        # Traceback contains the offending line:
        self.assertIn("Traceback", out.text)

    def test_namespace_persists_across_calls(self):
        s = _stub_session()
        RUN_PYTHON.handler({"code": "x = 42", "thought": "set"}, s)
        out = RUN_PYTHON.handler({"code": "print(x * 2)", "thought": "read"}, s)
        self.assertIn("84", out.text)

    def test_no_output_message(self):
        s = _stub_session()
        out = RUN_PYTHON.handler({"code": "x = 1", "thought": "x"}, s)
        self.assertEqual(out.text, "(no output)")

    def test_keyboard_interrupt_returns_interrupted(self):
        s = _stub_session()
        out = RUN_PYTHON.handler(
            {"code": "raise KeyboardInterrupt()", "thought": "x"}, s)
        self.assertTrue(out.is_error)
        self.assertTrue(out.interrupted)
        self.assertEqual(out.text, "(interrupted)")


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run tests and verify they fail**

```bash
pixi run bash -c 'cd tests && python -m unittest test_llm_tools -v'
```

Expected: `ModuleNotFoundError: No module named 'toksearch.llm.tools'`.

- [ ] **Step 3: Implement `tools.py` with `run_python` only**

Create `toksearch/llm/tools.py`:

```python
# Copyright 2024 General Atomics
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Tools registered with a Session.

Two tools ship in PR 1:

- ``run_python`` — executes Python code in the Session's persistent namespace.
- ``lookup_docs`` — returns the SKILL.md body for a named skill.

Tool handlers take ``(args: dict, session)`` and return a ``ToolOutput``.
"""

import io
import traceback
from contextlib import redirect_stdout, redirect_stderr
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


@dataclass(frozen=True)
class ToolOutput:
    """Result of a tool invocation."""

    text: str
    is_error: bool = False
    interrupted: bool = False    # True only when handler caught KeyboardInterrupt


@dataclass(frozen=True)
class ToolSpec:
    """Backend-agnostic tool descriptor.

    Backends translate ``input_schema`` to their native shape on request build.
    """

    name: str
    description: str
    input_schema: dict
    handler: Callable[[dict, Any], ToolOutput]


# ----------------------------------------------------------------------
# run_python
# ----------------------------------------------------------------------

_RUN_PYTHON_DESCRIPTION = (
    "Execute a Python code string. The execution namespace persists across "
    "calls within this session, so variables set in earlier calls are "
    "available later. The namespace is pre-populated with: toksearch, plt "
    "(matplotlib.pyplot), pd (pandas), np (numpy). Returns captured stdout "
    "and stderr. Populate 'thought' with a one-sentence description of what "
    "this code does and why."
)


def _run_python_handler(args: dict, session) -> ToolOutput:
    code = args["code"]
    stdout_buf = io.StringIO()
    stderr_buf = io.StringIO()
    is_error = False
    interrupted = False
    try:
        with redirect_stdout(stdout_buf), redirect_stderr(stderr_buf):
            exec(code, session.namespace)  # noqa: S102 -- intentional
    except KeyboardInterrupt:
        return ToolOutput(text="(interrupted)", is_error=True, interrupted=True)
    except Exception:
        stderr_buf.write(traceback.format_exc())
        is_error = True
    out = stdout_buf.getvalue()
    err = stderr_buf.getvalue()
    parts = []
    if out:
        parts.append(f"stdout:\n{out}")
    if err:
        parts.append(f"stderr:\n{err}")
    text = "\n".join(parts) if parts else "(no output)"
    return ToolOutput(text=text, is_error=is_error, interrupted=interrupted)


RUN_PYTHON = ToolSpec(
    name="run_python",
    description=_RUN_PYTHON_DESCRIPTION,
    input_schema={
        "type": "object",
        "properties": {
            "code":    {"type": "string",
                        "description": "Python code to execute."},
            "thought": {"type": "string",
                        "description": "One-sentence description of what this "
                                       "code does and why."},
        },
        "required": ["code", "thought"],
    },
    handler=_run_python_handler,
)
```

- [ ] **Step 4: Run tests and verify they pass**

```bash
pixi run bash -c 'cd tests && python -m unittest test_llm_tools -v'
```

Expected: 8 tests pass.

- [ ] **Step 5: Commit**

```bash
git add toksearch/llm/tools.py tests/test_llm_tools.py
git commit -m "Add ToolSpec, ToolOutput, and run_python handler"
```

---

## Task 8: Add the `lookup_docs` handler

**Files:**
- Modify: `toksearch/llm/tools.py`
- Modify: `tests/test_llm_tools.py`

- [ ] **Step 1: Append lookup_docs tests**

Open `tests/test_llm_tools.py` and add this class after `TestRunPython`:

```python
class TestLookupDocs(unittest.TestCase):
    """``lookup_docs`` reads SKILL.md bodies from the Session's skill registry.

    The handler accesses ``session.skills`` (a dict[name -> Skill]) which the
    real Session builds at __init__ from ``extra_skill_dirs`` + the core
    ``toksearch/skills/`` directory.  These tests stub that mapping.
    """

    def _stub_session(self, skills):
        return SimpleNamespace(skills=skills)

    def test_unknown_skill_is_error(self):
        from toksearch.llm.tools import LOOKUP_DOCS
        s = self._stub_session({})
        out = LOOKUP_DOCS.handler({"skill_name": "missing"}, s)
        self.assertTrue(out.is_error)
        self.assertIn("missing", out.text)

    def test_known_skill_returns_body(self):
        from toksearch.llm.tools import LOOKUP_DOCS, Skill
        s = self._stub_session({"foo": Skill(name="foo",
                                             description="d",
                                             body="Hello body.")})
        out = LOOKUP_DOCS.handler({"skill_name": "foo"}, s)
        self.assertFalse(out.is_error)
        self.assertEqual(out.text, "Hello body.")

    def test_lookup_docs_spec_shape(self):
        from toksearch.llm.tools import LOOKUP_DOCS
        self.assertEqual(LOOKUP_DOCS.name, "lookup_docs")
        self.assertIn("skill_name", LOOKUP_DOCS.input_schema["properties"])
        self.assertEqual(LOOKUP_DOCS.input_schema["required"], ["skill_name"])


class TestDiscoverSkills(unittest.TestCase):
    def test_returns_empty_for_nonexistent_dirs(self):
        from toksearch.llm.tools import discover_skills
        skills = discover_skills([Path("/nonexistent")])
        self.assertEqual(skills, {})

    def test_parses_skill_with_frontmatter(self):
        from toksearch.llm.tools import discover_skills, parse_skill_md
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            sk = root / "myskill"
            sk.mkdir()
            (sk / "SKILL.md").write_text(
                "---\nname: myskill\ndescription: My skill\n---\n\n"
                "Body content here.\n")
            skills = discover_skills([root])
            self.assertIn("myskill", skills)
            self.assertEqual(skills["myskill"].description, "My skill")
            self.assertIn("Body content here", skills["myskill"].body)

    def test_skips_dirs_without_skill_md(self):
        from toksearch.llm.tools import discover_skills
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "not_a_skill").mkdir()
            self.assertEqual(discover_skills([root]), {})

    def test_parse_skill_md_no_frontmatter(self):
        from toksearch.llm.tools import parse_skill_md
        import tempfile
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md",
                                          delete=False) as f:
            f.write("Just body, no frontmatter.\n")
            path = Path(f.name)
        try:
            fm, body = parse_skill_md(path)
            self.assertEqual(fm, {})
            self.assertIn("Just body", body)
        finally:
            path.unlink()
```

Also add this import at the top of the test file (after the existing `from toksearch.llm.tools import` line, modify it):

```python
from toksearch.llm.tools import (
    ToolSpec,
    ToolOutput,
    RUN_PYTHON,
)
from pathlib import Path
```

- [ ] **Step 2: Run tests and verify the new ones fail**

```bash
pixi run bash -c 'cd tests && python -m unittest test_llm_tools -v'
```

Expected: `RUN_PYTHON` tests still pass; the new `TestLookupDocs` and
`TestDiscoverSkills` tests fail with `ImportError` for `LOOKUP_DOCS`, `Skill`,
`discover_skills`, `parse_skill_md`.

- [ ] **Step 3: Extend `tools.py` with `lookup_docs` and skill discovery**

Append to `toksearch/llm/tools.py`:

```python
# ----------------------------------------------------------------------
# lookup_docs + skill discovery
# ----------------------------------------------------------------------

@dataclass(frozen=True)
class Skill:
    name: str
    description: str
    body: str


def parse_skill_md(path: Path) -> tuple[dict, str]:
    """Return ``(frontmatter, body)`` for a SKILL.md file.

    Frontmatter is the YAML-ish block between two ``---`` lines at the top of
    the file (we only need ``name`` and ``description`` so we do a tiny
    line-by-line parser instead of pulling in PyYAML).  If no frontmatter is
    present, returns ``({}, full_text)``.
    """
    text = path.read_text()
    if text.startswith("---"):
        _, fm, body = text.split("---", 2)
        fm_dict = {}
        for line in fm.strip().splitlines():
            if ":" in line:
                k, _, v = line.partition(":")
                fm_dict[k.strip()] = v.strip()
        return fm_dict, body.lstrip("\n")
    return {}, text


def discover_skills(skill_dirs: list[Path]) -> dict[str, Skill]:
    """Return ``{name: Skill}`` for every SKILL.md found under the given dirs.

    Each entry in ``skill_dirs`` is searched one level deep: each subdirectory
    containing a ``SKILL.md`` becomes a skill named after the subdirectory.
    Dirs that don't exist are silently skipped.
    """
    skills: dict[str, Skill] = {}
    for d in skill_dirs:
        if not d.exists():
            continue
        for sub in sorted(d.iterdir()):
            if not sub.is_dir():
                continue
            skill_md = sub / "SKILL.md"
            if not skill_md.exists():
                continue
            fm, body = parse_skill_md(skill_md)
            skills[sub.name] = Skill(
                name=sub.name,
                description=fm.get("description", ""),
                body=body,
            )
    return skills


_LOOKUP_DOCS_DESCRIPTION = (
    "Read a documentation skill. Returns the SKILL.md body for one of the "
    "registered skills. Call this when you need detail on a specific "
    "toksearch feature beyond what's in the system prompt."
)


def _lookup_docs_handler(args: dict, session) -> ToolOutput:
    name = args["skill_name"]
    skill = session.skills.get(name)
    if skill is None:
        available = sorted(session.skills)
        return ToolOutput(
            text=f"Unknown skill: {name!r}. Available: {available}",
            is_error=True,
        )
    return ToolOutput(text=skill.body, is_error=False)


LOOKUP_DOCS = ToolSpec(
    name="lookup_docs",
    description=_LOOKUP_DOCS_DESCRIPTION,
    input_schema={
        "type": "object",
        "properties": {
            "skill_name": {
                "type": "string",
                "description": "Name of a skill registered with the Session.",
            },
        },
        "required": ["skill_name"],
    },
    handler=_lookup_docs_handler,
)
```

- [ ] **Step 4: Run tests and verify they pass**

```bash
pixi run bash -c 'cd tests && python -m unittest test_llm_tools -v'
```

Expected: all 15 tests pass.

- [ ] **Step 5: Commit**

```bash
git add toksearch/llm/tools.py tests/test_llm_tools.py
git commit -m "Add lookup_docs tool and skill discovery"
```

---

## Task 9: Add `prompts.build_system_prompt`

**Files:**
- Create: `toksearch/llm/prompts.py`
- Create: `tests/test_llm_prompts.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_llm_prompts.py`:

```python
# Copyright 2024 General Atomics
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Tests for system-prompt assembly."""

import unittest

from toksearch.llm.tools import Skill
from toksearch.llm.prompts import build_system_prompt


class TestBuildSystemPrompt(unittest.TestCase):
    def test_contains_kernel_text(self):
        sp = build_system_prompt(skills={}, namespace_entries=[])
        self.assertIn("TokSearch", sp)
        self.assertIn("run_python", sp)

    def test_lists_namespace_entries(self):
        sp = build_system_prompt(
            skills={},
            namespace_entries=[("toksearch_d3d", "DIII-D signal classes")],
        )
        self.assertIn("toksearch_d3d", sp)
        self.assertIn("DIII-D signal classes", sp)

    def test_lists_skills_in_catalog(self):
        sp = build_system_prompt(
            skills={"foo": Skill(name="foo", description="Foo skill", body=""),
                    "bar": Skill(name="bar", description="Bar skill", body="")},
            namespace_entries=[],
        )
        self.assertIn("foo", sp)
        self.assertIn("Foo skill", sp)
        self.assertIn("bar", sp)
        self.assertIn("Bar skill", sp)

    def test_empty_catalogs_omit_sections(self):
        sp = build_system_prompt(skills={}, namespace_entries=[])
        # Should still be a non-empty kernel prompt without empty bullet lists.
        self.assertNotIn("\n-\n", sp)
        self.assertNotIn(" - :", sp)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run tests and verify they fail**

```bash
pixi run bash -c 'cd tests && python -m unittest test_llm_prompts -v'
```

Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Implement `prompts.py`**

Create `toksearch/llm/prompts.py`:

```python
# Copyright 2024 General Atomics
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""System-prompt assembly.

The kernel prompt is small (~20 lines) and constant.  The dynamic parts —
the list of contributed packages and the catalog of available skills — are
built from the Session's installed contributors at construction time.
"""

from .tools import Skill


_KERNEL = """\
You are a TokSearch expert. Use the run_python tool to execute code that fetches
and analyzes fusion data. The namespace persists across tool calls in this
session - variables from earlier calls are available in later ones.

{namespace_section}{catalog_section}Rules:
- Do not include import statements; common modules are pre-imported.
- If code raises an error, read the traceback, fix it, and try again.
- When you have a result, store it in a variable named `result` or describe it
  in plain text. Do not call any tool to "finish" - just stop emitting tool calls.
"""


def build_system_prompt(
    skills: dict[str, Skill],
    namespace_entries: list[tuple[str, str]],
) -> str:
    """Build the system prompt from the registered contributors.

    Parameters
    ----------
    skills:
        Mapping of skill name to ``Skill`` (from ``tools.discover_skills``).
    namespace_entries:
        List of ``(name, description)`` for each package contributed to the
        run_python namespace (core toksearch + any extras).
    """
    if namespace_entries:
        lines = ["You have access to fusion data via the following installed packages:"]
        for name, desc in namespace_entries:
            lines.append(f"- {name}: {desc}" if desc else f"- {name}")
        namespace_section = "\n".join(lines) + "\n\n"
    else:
        namespace_section = ""

    if skills:
        lines = ["Available documentation skills (call lookup_docs(skill_name=...) to read):"]
        for name in sorted(skills):
            desc = skills[name].description
            lines.append(f"- {name}: {desc}" if desc else f"- {name}")
        catalog_section = "\n".join(lines) + "\n\n"
    else:
        catalog_section = ""

    return _KERNEL.format(namespace_section=namespace_section,
                          catalog_section=catalog_section)
```

- [ ] **Step 4: Run tests and verify they pass**

```bash
pixi run bash -c 'cd tests && python -m unittest test_llm_prompts -v'
```

Expected: 4 tests pass.

- [ ] **Step 5: Commit**

```bash
git add toksearch/llm/prompts.py tests/test_llm_prompts.py
git commit -m "Add system-prompt assembly"
```

---

## Task 10: Add `Backend` ABC, `AssistantTurn`, `Callbacks`, and `_ToolLoopBackend`

**Files:**
- Create: `toksearch/llm/backends/base.py`
- Create: `tests/test_llm_backends_base.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_llm_backends_base.py`:

```python
# Copyright 2024 General Atomics
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Tests for the Backend ABC and _ToolLoopBackend shared loop.

We construct a minimal subclass (``ScriptedToolLoopBackend``) that returns
pre-scripted ``AssistantTurn``s from ``_send_request``, which lets us verify
the loop's behavior (tool execution, history appending, max_iterations, etc.)
without any provider SDK.
"""

import unittest
from types import SimpleNamespace
from typing import Iterable

from toksearch.llm.backends.base import (
    Backend,
    AssistantTurn,
    Callbacks,
    _ToolLoopBackend,
)
from toksearch.llm.events import TurnComplete
from toksearch.llm.messages import (
    Message, TextBlock, ToolUseBlock, ToolResultBlock,
)
from toksearch.llm.tools import ToolSpec, ToolOutput


def _ok_tool(name="echo"):
    """A trivial tool that returns its input as text."""
    return ToolSpec(
        name=name,
        description="echo",
        input_schema={"type": "object",
                      "properties": {"x": {"type": "string"}}},
        handler=lambda args, sess: ToolOutput(text=args["x"], is_error=False),
    )


class ScriptedToolLoopBackend(_ToolLoopBackend):
    """A _ToolLoopBackend whose _send_request returns scripted turns."""

    name = "scripted"
    default_model = "scripted-1"

    def __init__(self, turns: Iterable[AssistantTurn]):
        self._turns = list(turns)

    def _send_request(self, *, system_prompt, history, tools, model,
                      on_text=None):
        return self._turns.pop(0)


def _make_session(tools=None):
    """Minimal Session-shaped stub for the loop to drive."""
    s = SimpleNamespace()
    s.history = []
    s.namespace = {}
    s.system_prompt = "sys"
    s.model = "scripted-1"
    s.tool_specs = tools or []
    s._tools_by_name = {t.name: t for t in s.tool_specs}
    def append_user(msg):
        s.history.append(Message(role="user", content=[TextBlock(text=msg)]))
    def append_assistant(blocks):
        s.history.append(Message(role="assistant", content=list(blocks)))
    def append_tool_result(tool_use_id, output, is_error):
        s.history.append(Message(role="user", content=[
            ToolResultBlock(tool_use_id=tool_use_id,
                            output=output, is_error=is_error)]))
    def execute_tool(block: ToolUseBlock) -> ToolOutput:
        return s._tools_by_name[block.name].handler(block.args, s)
    s._append_user = append_user
    s._append_assistant = append_assistant
    s._append_tool_result = append_tool_result
    s._execute_tool = execute_tool
    return s


class TestBackendABC(unittest.TestCase):
    def test_backend_is_abstract(self):
        with self.assertRaises(TypeError):
            Backend()


class TestToolLoop(unittest.TestCase):
    def test_end_turn_with_only_text(self):
        backend = ScriptedToolLoopBackend([
            AssistantTurn(blocks=[TextBlock(text="all done")],
                          stop_reason="end_turn"),
        ])
        sess = _make_session()
        cbs = Callbacks()
        result = backend.run_conversation(sess, "hello", cbs, max_iterations=5)
        self.assertEqual(result.stop_reason, "end_turn")
        self.assertEqual(result.final_text, "all done")
        # History: user prompt + assistant turn
        self.assertEqual(len(sess.history), 2)
        self.assertEqual(sess.history[0].role, "user")
        self.assertEqual(sess.history[1].role, "assistant")

    def test_tool_use_then_end(self):
        tool = _ok_tool("echo")
        backend = ScriptedToolLoopBackend([
            AssistantTurn(blocks=[ToolUseBlock(id="t1", name="echo",
                                               args={"x": "hi"})],
                          stop_reason="tool_use"),
            AssistantTurn(blocks=[TextBlock(text="ok")],
                          stop_reason="end_turn"),
        ])
        sess = _make_session(tools=[tool])
        calls, results = [], []
        cbs = Callbacks(on_tool_call=calls.append,
                        on_tool_result=results.append)
        out = backend.run_conversation(sess, "go", cbs, max_iterations=5)
        self.assertEqual(out.stop_reason, "end_turn")
        # One tool call, one tool result fired
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].name, "echo")
        self.assertEqual(results[0].output, "hi")
        # History: user, assistant(tool_use), user(tool_result), assistant(text)
        self.assertEqual([m.role for m in sess.history],
                         ["user", "assistant", "user", "assistant"])

    def test_max_iterations_caps_loop(self):
        # Backend keeps emitting tool_use forever
        tool = _ok_tool("echo")
        forever = [AssistantTurn(blocks=[ToolUseBlock(id=f"t{i}", name="echo",
                                                      args={"x": str(i)})],
                                 stop_reason="tool_use")
                   for i in range(10)]
        backend = ScriptedToolLoopBackend(forever)
        sess = _make_session(tools=[tool])
        out = backend.run_conversation(sess, "go", Callbacks(),
                                        max_iterations=3)
        self.assertEqual(out.stop_reason, "max_iterations")

    def test_confirm_false_aborts(self):
        tool = _ok_tool("echo")
        backend = ScriptedToolLoopBackend([
            AssistantTurn(blocks=[ToolUseBlock(id="t1", name="echo",
                                               args={"x": "hi"})],
                          stop_reason="tool_use"),
        ])
        sess = _make_session(tools=[tool])
        cbs = Callbacks(confirm=lambda call: False)
        out = backend.run_conversation(sess, "go", cbs, max_iterations=5)
        self.assertEqual(out.stop_reason, "interrupted")

    def test_on_event_catch_all_fires_for_everything(self):
        tool = _ok_tool("echo")
        backend = ScriptedToolLoopBackend([
            AssistantTurn(blocks=[ToolUseBlock(id="t1", name="echo",
                                               args={"x": "hi"})],
                          stop_reason="tool_use"),
            AssistantTurn(blocks=[TextBlock(text="done")],
                          stop_reason="end_turn"),
        ])
        sess = _make_session(tools=[tool])
        events = []
        cbs = Callbacks(on_event=events.append)
        backend.run_conversation(sess, "go", cbs, max_iterations=5)
        kinds = [type(e).__name__ for e in events]
        # ToolCall, ToolResult, TurnComplete at minimum (text deltas optional)
        self.assertIn("ToolCall", kinds)
        self.assertIn("ToolResult", kinds)
        self.assertIn("TurnComplete", kinds)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run tests and verify they fail**

```bash
pixi run bash -c 'cd tests && python -m unittest test_llm_backends_base -v'
```

Expected: `ModuleNotFoundError: No module named 'toksearch.llm.backends.base'`.

- [ ] **Step 3: Implement `backends/base.py`**

Create `toksearch/llm/backends/base.py`:

```python
# Copyright 2024 General Atomics
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Backend ABC and the shared tool-use loop.

``_ToolLoopBackend`` holds the loop that all raw-API backends (Anthropic,
OpenAI, AmSC-via-preset) share.  Subclasses implement only:

- ``_send_request(system_prompt, history, tools, model, on_text)`` returning
  an ``AssistantTurn``.
- Translation helpers between our ``ContentBlock`` / ``ToolSpec`` taxonomy and
  the provider's native shapes.

Streaming text is deferred to a follow-up PR.  In PR 1, ``_send_request`` is
expected to be non-streaming; it MAY call ``on_text`` once at the end with the
full assistant text, but the loop does not require it.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, Literal

from ..events import TextDelta, ToolCall, ToolResult, TurnComplete
from ..messages import Message, TextBlock, ToolUseBlock, ToolResultBlock


@dataclass
class AssistantTurn:
    """One assistant round-trip from a backend.

    ``blocks`` is the assistant's content this turn (text + tool_use mixed).
    ``stop_reason`` mirrors the provider's stop reason; ``"tool_use"`` means
    the loop should execute tools and call ``_send_request`` again.
    """

    blocks: list  # list[ContentBlock]
    stop_reason: Literal["tool_use", "end_turn", "max_tokens", "stop_sequence"]


@dataclass
class Callbacks:
    on_text: Callable[[str], None] | None = None
    on_tool_call: Callable[[ToolCall], None] | None = None
    on_tool_result: Callable[[ToolResult], None] | None = None
    on_event: Callable[[object], None] | None = None
    confirm: Callable[[ToolCall], bool] | None = None

    def fire_text(self, text: str) -> None:
        if self.on_text is not None:
            self.on_text(text)
        if self.on_event is not None:
            self.on_event(TextDelta(text=text))

    def fire_tool_call(self, e: ToolCall) -> None:
        if self.on_tool_call is not None:
            self.on_tool_call(e)
        if self.on_event is not None:
            self.on_event(e)

    def fire_tool_result(self, e: ToolResult) -> None:
        if self.on_tool_result is not None:
            self.on_tool_result(e)
        if self.on_event is not None:
            self.on_event(e)

    def fire_turn_complete(self, e: TurnComplete) -> None:
        if self.on_event is not None:
            self.on_event(e)


class Backend(ABC):
    """Pluggable LLM provider.

    Subclasses advance the conversation by one user-message worth of work,
    dispatching ``Callbacks`` events along the way.
    """

    name: str
    default_model: str

    @abstractmethod
    def run_conversation(
        self,
        session,
        new_user_message: str,
        callbacks: Callbacks,
        max_iterations: int,
    ) -> TurnComplete:
        ...


class _ToolLoopBackend(Backend):
    """Shared tool-use loop for raw-API backends (Anthropic, OpenAI)."""

    @abstractmethod
    def _send_request(self, *, system_prompt, history, tools, model,
                      on_text=None) -> AssistantTurn:
        ...

    def run_conversation(self, session, new_user_message, callbacks,
                          max_iterations):
        session._append_user(new_user_message)
        final_text = ""
        for _ in range(max_iterations):
            turn = self._send_request(
                system_prompt=session.system_prompt,
                history=session.history,
                tools=session.tool_specs,
                model=session.model,
                on_text=callbacks.fire_text,
            )
            session._append_assistant(turn.blocks)
            tool_use_blocks = [b for b in turn.blocks
                               if isinstance(b, ToolUseBlock)]
            text_blocks = [b for b in turn.blocks
                           if isinstance(b, TextBlock)]
            if text_blocks:
                final_text = text_blocks[-1].text
            for block in tool_use_blocks:
                call = ToolCall(
                    id=block.id, name=block.name, args=block.args,
                    thought=block.args.get("thought")
                            if block.name == "run_python" else None,
                )
                callbacks.fire_tool_call(call)
                if callbacks.confirm is not None and not callbacks.confirm(call):
                    result = TurnComplete(stop_reason="interrupted", final_text="")
                    callbacks.fire_turn_complete(result)
                    return result
                output = session._execute_tool(block)
                event = ToolResult(id=block.id, output=output.text,
                                    is_error=output.is_error)
                callbacks.fire_tool_result(event)
                session._append_tool_result(block.id, output.text,
                                             output.is_error)
            if turn.stop_reason != "tool_use":
                result = TurnComplete(stop_reason="end_turn",
                                       final_text=final_text)
                callbacks.fire_turn_complete(result)
                return result
        result = TurnComplete(stop_reason="max_iterations",
                               final_text=final_text)
        callbacks.fire_turn_complete(result)
        return result
```

- [ ] **Step 4: Run tests and verify they pass**

```bash
pixi run bash -c 'cd tests && python -m unittest test_llm_backends_base -v'
```

Expected: 6 tests pass.

- [ ] **Step 5: Commit**

```bash
git add toksearch/llm/backends/base.py tests/test_llm_backends_base.py
git commit -m "Add Backend ABC and _ToolLoopBackend shared loop"
```

---

## Task 11: Add `FakeBackend`

**Files:**
- Create: `toksearch/llm/backends/fake.py`
- Create: `tests/test_llm_backends_fake.py`

`FakeBackend` is the testing seam for Session-level tests in Task 13. It does
NOT subclass `_ToolLoopBackend` — it implements `run_conversation` directly so
tests can script the full conversation (tool calls included) rather than
working at the round-trip granularity.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_llm_backends_fake.py`:

```python
# Copyright 2024 General Atomics
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Tests for FakeBackend (the testing seam)."""

import unittest
from types import SimpleNamespace

from toksearch.llm.backends.base import AssistantTurn, Callbacks
from toksearch.llm.backends.fake import FakeBackend
from toksearch.llm.messages import (
    Message, TextBlock, ToolUseBlock, ToolResultBlock,
)
from toksearch.llm.tools import ToolSpec, ToolOutput


def _stub_session(tools=None):
    s = SimpleNamespace()
    s.history = []
    s.namespace = {}
    s.system_prompt = "sys"
    s.model = "fake-1"
    s.tool_specs = tools or []
    s._tools_by_name = {t.name: t for t in s.tool_specs}
    s._append_user = lambda msg: s.history.append(
        Message(role="user", content=[TextBlock(text=msg)]))
    s._append_assistant = lambda blocks: s.history.append(
        Message(role="assistant", content=list(blocks)))
    s._append_tool_result = lambda tid, out, err: s.history.append(
        Message(role="user", content=[ToolResultBlock(
            tool_use_id=tid, output=out, is_error=err)]))
    s._execute_tool = lambda block: s._tools_by_name[block.name].handler(
        block.args, s)
    return s


class TestFakeBackend(unittest.TestCase):
    def test_simple_text_response(self):
        backend = FakeBackend(scripted_turns=[
            AssistantTurn(blocks=[TextBlock(text="hi")],
                          stop_reason="end_turn"),
        ])
        sess = _stub_session()
        out = backend.run_conversation(sess, "hello", Callbacks(),
                                        max_iterations=5)
        self.assertEqual(out.stop_reason, "end_turn")
        self.assertEqual(out.final_text, "hi")

    def test_recording_inspects_calls(self):
        record = []
        backend = FakeBackend(scripted_turns=[
            AssistantTurn(blocks=[TextBlock(text="ok")],
                          stop_reason="end_turn"),
        ], record=record)
        sess = _stub_session()
        backend.run_conversation(sess, "do thing", Callbacks(),
                                  max_iterations=5)
        self.assertEqual(len(record), 1)
        self.assertEqual(record[0]["user_message"], "do thing")
        self.assertEqual(record[0]["model"], "fake-1")

    def test_script_exhausted_raises(self):
        backend = FakeBackend(scripted_turns=[])
        sess = _stub_session()
        with self.assertRaises(RuntimeError):
            backend.run_conversation(sess, "hello", Callbacks(),
                                      max_iterations=5)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run tests and verify they fail**

```bash
pixi run bash -c 'cd tests && python -m unittest test_llm_backends_fake -v'
```

Expected: `ModuleNotFoundError: No module named 'toksearch.llm.backends.fake'`.

- [ ] **Step 3: Implement `backends/fake.py`**

Create `toksearch/llm/backends/fake.py`:

```python
# Copyright 2024 General Atomics
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

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
```

- [ ] **Step 4: Run tests and verify they pass**

```bash
pixi run bash -c 'cd tests && python -m unittest test_llm_backends_fake -v'
```

Expected: 3 tests pass.

- [ ] **Step 5: Commit**

```bash
git add toksearch/llm/backends/fake.py tests/test_llm_backends_fake.py
git commit -m "Add FakeBackend testing seam"
```

---

## Task 12: Wire backend registry

**Files:**
- Modify: `toksearch/llm/backends/__init__.py`
- Create: `tests/test_llm_backends_registry.py`

The registry maps backend-class names (NOT preset names) to concrete classes.
PR 1 registers `"anthropic"` and `"openai"`; PR 3 adds `"claude-max"`. Preset
resolution (Task 6) maps user-facing names to backend-class names + kwargs.

- [ ] **Step 1: Write the failing test**

Create `tests/test_llm_backends_registry.py`:

```python
# Copyright 2024 General Atomics
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Tests for the backend registry."""

import unittest

from toksearch.llm.errors import LLMConfigError


class TestBackendRegistry(unittest.TestCase):
    def test_get_backend_class_anthropic(self):
        from toksearch.llm.backends import get_backend_class
        from toksearch.llm.backends.anthropic import AnthropicBackend
        self.assertIs(get_backend_class("anthropic"), AnthropicBackend)

    def test_get_backend_class_openai(self):
        from toksearch.llm.backends import get_backend_class
        from toksearch.llm.backends.openai import OpenAIBackend
        self.assertIs(get_backend_class("openai"), OpenAIBackend)

    def test_unknown_raises(self):
        from toksearch.llm.backends import get_backend_class
        with self.assertRaises(LLMConfigError):
            get_backend_class("nonexistent")


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run tests and verify they fail**

```bash
pixi run bash -c 'cd tests && python -m unittest test_llm_backends_registry -v'
```

Expected: `ImportError` for `get_backend_class` (and `AnthropicBackend` /
`OpenAIBackend` which are added in Tasks 14 and 15).

- [ ] **Step 3: Replace `toksearch/llm/backends/__init__.py`**

```python
# Copyright 2024 General Atomics
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""Backend registry and class lookup for toksearch.llm."""

from ..errors import LLMConfigError


def get_backend_class(name: str):
    """Resolve a backend-class name (e.g. 'anthropic', 'openai') to its class.

    Imports the concrete class lazily so the import of ``toksearch.llm.backends``
    doesn't require ``anthropic`` / ``openai`` SDKs to be installed.
    """
    if name == "anthropic":
        from .anthropic import AnthropicBackend
        return AnthropicBackend
    if name == "openai":
        from .openai import OpenAIBackend
        return OpenAIBackend
    raise LLMConfigError(
        f"Unknown backend: {name!r}. Known: anthropic, openai. "
        "(claude-max is added in PR 3.)")
```

- [ ] **Step 4: Note: tests still fail until Tasks 14 and 15**

After implementing Tasks 14 and 15, the registry tests pass. For now, leave
them red; they're already committed as part of the registry change but the
imports they rely on don't exist yet.

- [ ] **Step 5: Commit**

```bash
git add toksearch/llm/backends/__init__.py tests/test_llm_backends_registry.py
git commit -m "Add backend registry with lazy class lookup"
```

---

## Task 13: Add `Session`

**Files:**
- Create: `toksearch/llm/session.py`
- Create: `tests/test_llm_session.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_llm_session.py`:

```python
# Copyright 2024 General Atomics
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Session-level tests using FakeBackend.

The Session class is responsible for: building the system prompt, owning the
namespace and history, registering tools, dispatching to the backend, and
providing reset/introspection.
"""

import unittest

from toksearch.llm.backends.base import AssistantTurn
from toksearch.llm.backends.fake import FakeBackend
from toksearch.llm.messages import TextBlock, ToolUseBlock
from toksearch.llm.session import Session


def _text(s):
    return AssistantTurn(blocks=[TextBlock(text=s)], stop_reason="end_turn")


def _tool_use(name, args, id_="t1"):
    return AssistantTurn(
        blocks=[ToolUseBlock(id=id_, name=name, args=args)],
        stop_reason="tool_use",
    )


class TestSessionBasics(unittest.TestCase):
    def test_send_returns_turn_complete(self):
        backend = FakeBackend(scripted_turns=[_text("hello")])
        sess = Session(backend=backend)
        out = sess.send("hi")
        self.assertEqual(out.stop_reason, "end_turn")
        self.assertEqual(out.final_text, "hello")

    def test_namespace_pre_populated(self):
        backend = FakeBackend(scripted_turns=[_text("ok")])
        sess = Session(backend=backend)
        # toksearch and numpy are required; pandas and matplotlib are optional
        # in the test env (they live in the [llm] extra).
        self.assertIn("toksearch", sess.namespace)
        self.assertIn("np", sess.namespace)

    def test_extra_namespace_merged(self):
        backend = FakeBackend(scripted_turns=[_text("ok")])
        sess = Session(backend=backend, extra_namespace={"answer": 42})
        self.assertEqual(sess.namespace["answer"], 42)


class TestSessionPersistence(unittest.TestCase):
    def test_namespace_persists_across_send_calls(self):
        backend = FakeBackend(scripted_turns=[
            _tool_use("run_python", {"code": "x = 99", "thought": "set"}),
            _text("set"),
            _tool_use("run_python", {"code": "print(x)", "thought": "read"}, id_="t2"),
            _text("read"),
        ])
        sess = Session(backend=backend)
        sess.send("set x")
        sess.send("read x")
        self.assertEqual(sess.namespace["x"], 99)

    def test_reset_clears_namespace_and_history(self):
        backend = FakeBackend(scripted_turns=[
            _tool_use("run_python", {"code": "x = 1", "thought": "set"}),
            _text("done"),
        ])
        sess = Session(backend=backend)
        sess.send("set x")
        self.assertIn("x", sess.namespace)
        self.assertGreater(len(sess.history), 0)
        sess.reset()
        self.assertNotIn("x", sess.namespace)
        self.assertEqual(len(sess.history), 0)
        # Standard names are still there
        self.assertIn("toksearch", sess.namespace)


class TestSessionCallbacks(unittest.TestCase):
    def test_on_tool_call_fires_before_result(self):
        backend = FakeBackend(scripted_turns=[
            _tool_use("run_python", {"code": "print('hi')", "thought": "x"}),
            _text("done"),
        ])
        sess = Session(backend=backend)
        order = []
        sess.send("go",
                  on_tool_call=lambda c: order.append(("call", c.name)),
                  on_tool_result=lambda r: order.append(("result", r.output)))
        self.assertEqual(order[0][0], "call")
        self.assertEqual(order[1][0], "result")
        self.assertIn("hi", order[1][1])

    def test_confirm_false_aborts(self):
        backend = FakeBackend(scripted_turns=[
            _tool_use("run_python", {"code": "print('x')", "thought": "x"}),
        ])
        sess = Session(backend=backend)
        out = sess.send("go", confirm=lambda call: False)
        self.assertEqual(out.stop_reason, "interrupted")


class TestSessionTools(unittest.TestCase):
    def test_run_python_and_lookup_docs_registered(self):
        backend = FakeBackend(scripted_turns=[_text("ok")])
        sess = Session(backend=backend)
        names = {t.name for t in sess.tool_specs}
        self.assertEqual(names, {"run_python", "lookup_docs"})


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run tests and verify they fail**

```bash
pixi run bash -c 'cd tests && python -m unittest test_llm_session -v'
```

Expected: `ModuleNotFoundError: No module named 'toksearch.llm.session'`.

- [ ] **Step 3: Implement `session.py`**

Create `toksearch/llm/session.py`:

```python
# Copyright 2024 General Atomics
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Session — the user-facing object for toksearch.llm.

Owns: history, run_python namespace, tool registry, skill registry, backend.
"""

from pathlib import Path
from typing import Callable

from .backends.base import Backend, Callbacks
from .events import TurnComplete
from .messages import Message, TextBlock, ToolResultBlock, ToolUseBlock
from .prompts import build_system_prompt
from .tools import LOOKUP_DOCS, RUN_PYTHON, ToolOutput, discover_skills


def _default_namespace() -> dict:
    """Build the standard run_python namespace.

    Imports happen here (not at module load) so that simply importing
    ``toksearch.llm`` doesn't drag in matplotlib / pandas / numpy until the
    user actually constructs a Session.  toksearch and numpy are required;
    pandas and matplotlib are optional (matplotlib is in the ``[llm]`` extra
    but tests run without it).
    """
    import toksearch
    import numpy as np
    ns = {"toksearch": toksearch, "np": np}
    try:
        import pandas as pd
        ns["pd"] = pd
    except ImportError:
        pass
    try:
        import matplotlib
        matplotlib.use("Agg", force=False)  # idempotent; safe in non-GUI envs
        import matplotlib.pyplot as plt
        ns["plt"] = plt
    except ImportError:
        pass
    return ns


def _core_skill_dir() -> Path:
    import toksearch
    return Path(toksearch.__file__).parent / "skills"


class Session:
    """A conversational LLM session over the run_python persistent namespace."""

    def __init__(
        self,
        backend: Backend,
        model: str | None = None,
        max_iterations: int = 20,
        extra_namespace: dict | None = None,
        packages: list[str] | None = None,    # PR 2 plumbing; unused in PR 1
        extra_skill_dirs: list[Path] | None = None,
    ):
        self.backend = backend
        self.model = model or backend.default_model
        self.max_iterations = max_iterations
        self.namespace = _default_namespace()
        if extra_namespace:
            self.namespace.update(extra_namespace)
        # Skills: core toksearch + any extras the caller passed in
        skill_dirs = [_core_skill_dir()] + list(extra_skill_dirs or [])
        self.skills = discover_skills(skill_dirs)
        # Tools: fixed in PR 1; PR 4 may register more.
        self.tool_specs = [RUN_PYTHON, LOOKUP_DOCS]
        self._tools_by_name = {t.name: t for t in self.tool_specs}
        # System prompt: kernel + dynamic catalogs
        namespace_entries = [
            ("toksearch", "core pipeline, MdsSignal, ZarrSignal, fetch_dataset"),
        ]
        self.system_prompt = build_system_prompt(self.skills, namespace_entries)
        # History
        self.history: list[Message] = []

    # ---- Public API ----

    def send(
        self,
        prompt: str,
        *,
        on_text: Callable[[str], None] | None = None,
        on_tool_call=None,
        on_tool_result=None,
        on_event=None,
        confirm=None,
    ) -> TurnComplete:
        cbs = Callbacks(
            on_text=on_text,
            on_tool_call=on_tool_call,
            on_tool_result=on_tool_result,
            on_event=on_event,
            confirm=confirm,
        )
        return self.backend.run_conversation(
            self, prompt, cbs, max_iterations=self.max_iterations,
        )

    def reset(self) -> None:
        """Clear history and reset the namespace to its initial pre-populated state."""
        self.namespace = _default_namespace()
        self.history = []

    # ---- Hooks called by backends (semi-private) ----

    def _append_user(self, text: str) -> None:
        self.history.append(Message(role="user",
                                     content=[TextBlock(text=text)]))

    def _append_assistant(self, blocks) -> None:
        self.history.append(Message(role="assistant",
                                     content=list(blocks)))

    def _append_tool_result(self, tool_use_id: str, output: str,
                             is_error: bool) -> None:
        self.history.append(Message(role="user", content=[
            ToolResultBlock(tool_use_id=tool_use_id,
                            output=output, is_error=is_error)
        ]))

    def _execute_tool(self, block: ToolUseBlock) -> ToolOutput:
        spec = self._tools_by_name.get(block.name)
        if spec is None:
            return ToolOutput(text=f"Unknown tool: {block.name}",
                              is_error=True)
        return spec.handler(block.args, self)
```

- [ ] **Step 4: Run tests and verify they pass**

```bash
pixi run bash -c 'cd tests && python -m unittest test_llm_session -v'
```

Expected: 7 tests pass.

- [ ] **Step 5: Commit**

```bash
git add toksearch/llm/session.py tests/test_llm_session.py
git commit -m "Add Session — owner of state, dispatcher to backend"
```

---

## Task 14: Add `AnthropicBackend`

**Files:**
- Create: `toksearch/llm/backends/anthropic.py`
- Create: `tests/test_llm_backends_anthropic.py`

Mocks the `anthropic` SDK so tests don't require the package at test time.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_llm_backends_anthropic.py`:

```python
# Copyright 2024 General Atomics
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Tests for AnthropicBackend.

The anthropic SDK is mocked at the module-attribute level (``backend._client``)
so tests don't need the SDK installed and don't make network calls.
"""

import sys
import types
import unittest
from unittest import mock


# Install a stub `anthropic` module before any imports of the backend so the
# `import anthropic` inside backend.__init__ resolves to our stub.
_stub_anthropic = types.ModuleType("anthropic")
_stub_anthropic.Anthropic = mock.MagicMock
_stub_anthropic.AuthenticationError = type("AuthenticationError", (Exception,), {})
_stub_anthropic.RateLimitError = type("RateLimitError", (Exception,), {})
_stub_anthropic.APIStatusError = type("APIStatusError", (Exception,), {})
_stub_anthropic.APIConnectionError = type("APIConnectionError", (Exception,), {})
sys.modules.setdefault("anthropic", _stub_anthropic)

from toksearch.llm.backends.anthropic import AnthropicBackend  # noqa: E402
from toksearch.llm.backends.base import AssistantTurn  # noqa: E402
from toksearch.llm.errors import LLMAuthError  # noqa: E402
from toksearch.llm.messages import TextBlock, ToolUseBlock  # noqa: E402
from toksearch.llm.tools import ToolSpec, ToolOutput  # noqa: E402


def _make_response(text=None, tool_use=None, stop_reason="end_turn"):
    """Build a fake anthropic.types.Message-shaped response."""
    blocks = []
    if text is not None:
        blocks.append(mock.MagicMock(type="text", text=text))
    if tool_use is not None:
        b = mock.MagicMock(type="tool_use")
        b.id = tool_use["id"]
        b.name = tool_use["name"]
        b.input = tool_use["input"]
        blocks.append(b)
    resp = mock.MagicMock()
    resp.content = blocks
    resp.stop_reason = stop_reason
    return resp


class TestAnthropicSendRequest(unittest.TestCase):
    def setUp(self):
        self.backend = AnthropicBackend(api_key="sk-test")
        self.backend._client = mock.MagicMock()

    def test_text_response_becomes_assistant_turn(self):
        self.backend._client.messages.create.return_value = _make_response(
            text="hi", stop_reason="end_turn")
        turn = self.backend._send_request(
            system_prompt="sys", history=[], tools=[], model="claude-x")
        self.assertEqual(turn.stop_reason, "end_turn")
        self.assertEqual(len(turn.blocks), 1)
        self.assertIsInstance(turn.blocks[0], TextBlock)
        self.assertEqual(turn.blocks[0].text, "hi")

    def test_tool_use_response_becomes_tool_use_block(self):
        self.backend._client.messages.create.return_value = _make_response(
            tool_use={"id": "t1", "name": "run_python",
                      "input": {"code": "1+1", "thought": "x"}},
            stop_reason="tool_use")
        turn = self.backend._send_request(
            system_prompt="sys", history=[], tools=[], model="claude-x")
        self.assertEqual(turn.stop_reason, "tool_use")
        self.assertEqual(len(turn.blocks), 1)
        b = turn.blocks[0]
        self.assertIsInstance(b, ToolUseBlock)
        self.assertEqual(b.id, "t1")
        self.assertEqual(b.name, "run_python")
        self.assertEqual(b.args, {"code": "1+1", "thought": "x"})

    def test_tools_translated_to_anthropic_schema(self):
        spec = ToolSpec(
            name="echo",
            description="echo",
            input_schema={"type": "object",
                          "properties": {"x": {"type": "string"}}},
            handler=lambda a, s: ToolOutput(text="x"),
        )
        self.backend._client.messages.create.return_value = _make_response(
            text="", stop_reason="end_turn")
        self.backend._send_request(
            system_prompt="sys", history=[], tools=[spec], model="claude-x")
        kwargs = self.backend._client.messages.create.call_args.kwargs
        sent_tools = kwargs["tools"]
        self.assertEqual(sent_tools[0]["name"], "echo")
        self.assertEqual(sent_tools[0]["description"], "echo")
        self.assertEqual(sent_tools[0]["input_schema"],
                         {"type": "object",
                          "properties": {"x": {"type": "string"}}})


class TestAnthropicAuthError(unittest.TestCase):
    def test_auth_error_translated_to_llm_auth_error(self):
        # Build a fresh exception type and patch it onto the anthropic module
        # so the backend's `except anthropic.AuthenticationError` matches it
        # regardless of whether the real SDK is installed.
        import anthropic as _anthropic
        fake_auth = type("FakeAuth", (Exception,), {})
        backend = AnthropicBackend(api_key="x")
        backend._client = mock.MagicMock()
        with mock.patch.object(_anthropic, "AuthenticationError", fake_auth):
            backend._client.messages.create.side_effect = fake_auth("401")
            with self.assertRaises(LLMAuthError):
                backend._send_request(system_prompt="s", history=[],
                                       tools=[], model="m")


class TestAnthropicBaseUrl(unittest.TestCase):
    def test_base_url_passed_to_client(self):
        with mock.patch.object(_stub_anthropic, "Anthropic") as ctor:
            AnthropicBackend(api_key="x",
                              base_url="https://custom.example")._build_client()
            self.assertEqual(
                ctor.call_args.kwargs["base_url"],
                "https://custom.example")


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run tests and verify they fail**

```bash
pixi run bash -c 'cd tests && python -m unittest test_llm_backends_anthropic -v'
```

Expected: `ModuleNotFoundError: No module named 'toksearch.llm.backends.anthropic'`.

- [ ] **Step 3: Implement `backends/anthropic.py`**

Create `toksearch/llm/backends/anthropic.py`:

```python
# Copyright 2024 General Atomics
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""AnthropicBackend — driver for the Anthropic Messages API.

Also serves AmSC and any other Anthropic-compatible endpoint via the
``base_url`` constructor argument (set by preset resolution).

PR 1 uses the blocking, non-streaming API.  Streaming text deltas are deferred
to a follow-up PR.
"""

import anthropic

from ..errors import LLMAuthError, LLMBackendError, LLMRateLimitError
from ..messages import (
    Message, TextBlock, ToolResultBlock, ToolUseBlock,
)
from .base import AssistantTurn, _ToolLoopBackend


class AnthropicBackend(_ToolLoopBackend):
    name = "anthropic"
    default_model = "claude-sonnet-4-6"

    def __init__(
        self,
        api_key: str | None,
        base_url: str | None = None,
        max_tokens: int = 4096,
    ):
        self._api_key = api_key
        self._base_url = base_url
        self._max_tokens = max_tokens
        self._client = None  # lazy; tests inject a mock

    def _build_client(self):
        kwargs = {}
        if self._api_key is not None:
            kwargs["api_key"] = self._api_key
        if self._base_url is not None:
            kwargs["base_url"] = self._base_url
        self._client = anthropic.Anthropic(**kwargs)
        return self._client

    def _ensure_client(self):
        if self._client is None:
            self._build_client()
        return self._client

    # ---- Translation: ToolSpec -> Anthropic tool schema ----

    @staticmethod
    def _spec_to_native(spec) -> dict:
        return {
            "name": spec.name,
            "description": spec.description,
            "input_schema": spec.input_schema,
        }

    # ---- Translation: our Message list -> Anthropic messages= ----

    @staticmethod
    def _history_to_native(history: list[Message]) -> list[dict]:
        out = []
        for m in history:
            native_blocks = []
            for b in m.content:
                if isinstance(b, TextBlock):
                    native_blocks.append({"type": "text", "text": b.text})
                elif isinstance(b, ToolUseBlock):
                    native_blocks.append({
                        "type": "tool_use",
                        "id": b.id, "name": b.name, "input": b.args,
                    })
                elif isinstance(b, ToolResultBlock):
                    native_blocks.append({
                        "type": "tool_result",
                        "tool_use_id": b.tool_use_id,
                        "content": b.output,
                        "is_error": b.is_error,
                    })
            out.append({"role": m.role, "content": native_blocks})
        return out

    # ---- Translation: Anthropic response.content -> our ContentBlock list ----

    @staticmethod
    def _response_to_blocks(content) -> list:
        out = []
        for blk in content:
            if blk.type == "text":
                out.append(TextBlock(text=blk.text))
            elif blk.type == "tool_use":
                out.append(ToolUseBlock(id=blk.id, name=blk.name,
                                         args=dict(blk.input)))
        return out

    # ---- Main entrypoint called by _ToolLoopBackend ----

    def _send_request(self, *, system_prompt, history, tools, model,
                      on_text=None) -> AssistantTurn:
        client = self._ensure_client()
        try:
            resp = client.messages.create(
                model=model,
                max_tokens=self._max_tokens,
                system=system_prompt,
                messages=self._history_to_native(history),
                tools=[self._spec_to_native(t) for t in tools],
            )
        except anthropic.AuthenticationError as e:
            raise LLMAuthError(
                "Anthropic auth failed. Set ANTHROPIC_API_KEY or pass "
                "api_key=... to the backend.") from e
        except anthropic.RateLimitError as e:
            raise LLMRateLimitError(str(e)) from e
        except anthropic.APIConnectionError as e:
            raise LLMBackendError(f"Network error: {e}") from e
        except anthropic.APIStatusError as e:
            raise LLMBackendError(f"Anthropic API error: {e}") from e
        blocks = self._response_to_blocks(resp.content)
        text_total = "".join(b.text for b in blocks if isinstance(b, TextBlock))
        if text_total and on_text is not None:
            on_text(text_total)
        return AssistantTurn(blocks=blocks, stop_reason=resp.stop_reason)
```

- [ ] **Step 4: Run tests and verify they pass**

```bash
pixi run bash -c 'cd tests && python -m unittest test_llm_backends_anthropic -v'
```

Expected: 5 tests pass.

- [ ] **Step 5: Commit**

```bash
git add toksearch/llm/backends/anthropic.py tests/test_llm_backends_anthropic.py
git commit -m "Add AnthropicBackend with mocked SDK tests"
```

---

## Task 15: Add `OpenAIBackend`

**Files:**
- Create: `toksearch/llm/backends/openai.py`
- Create: `tests/test_llm_backends_openai.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_llm_backends_openai.py`:

```python
# Copyright 2024 General Atomics
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Tests for OpenAIBackend.

The openai SDK is mocked at the module-attribute level so tests don't need
the SDK installed and don't make network calls.
"""

import json
import sys
import types
import unittest
from unittest import mock


# Stub openai module + its error types
_stub_openai = types.ModuleType("openai")
_stub_openai.OpenAI = mock.MagicMock
_stub_openai.AuthenticationError = type("AuthenticationError", (Exception,), {})
_stub_openai.RateLimitError = type("RateLimitError", (Exception,), {})
_stub_openai.APIConnectionError = type("APIConnectionError", (Exception,), {})
_stub_openai.APIStatusError = type("APIStatusError", (Exception,), {})
sys.modules.setdefault("openai", _stub_openai)

from toksearch.llm.backends.openai import OpenAIBackend  # noqa: E402
from toksearch.llm.errors import LLMAuthError  # noqa: E402
from toksearch.llm.messages import TextBlock, ToolUseBlock  # noqa: E402
from toksearch.llm.tools import ToolSpec, ToolOutput  # noqa: E402


def _make_chat_response(text=None, tool_calls=None, finish_reason="stop"):
    """Build a fake openai chat completion response."""
    msg = mock.MagicMock()
    msg.content = text
    msg.tool_calls = []
    for tc in (tool_calls or []):
        m = mock.MagicMock()
        m.id = tc["id"]
        m.type = "function"
        m.function.name = tc["name"]
        m.function.arguments = json.dumps(tc["args"])
        msg.tool_calls.append(m)
    choice = mock.MagicMock()
    choice.message = msg
    choice.finish_reason = finish_reason
    resp = mock.MagicMock()
    resp.choices = [choice]
    return resp


class TestOpenAISendRequest(unittest.TestCase):
    def setUp(self):
        self.backend = OpenAIBackend(api_key="sk-test")
        self.backend._client = mock.MagicMock()

    def test_text_response_becomes_assistant_turn(self):
        self.backend._client.chat.completions.create.return_value = (
            _make_chat_response(text="hello", finish_reason="stop"))
        turn = self.backend._send_request(
            system_prompt="sys", history=[], tools=[], model="gpt-4o")
        self.assertEqual(turn.stop_reason, "end_turn")
        self.assertEqual(len(turn.blocks), 1)
        self.assertIsInstance(turn.blocks[0], TextBlock)
        self.assertEqual(turn.blocks[0].text, "hello")

    def test_tool_call_response(self):
        self.backend._client.chat.completions.create.return_value = (
            _make_chat_response(text=None,
                                tool_calls=[{"id": "t1", "name": "run_python",
                                             "args": {"code": "1+1",
                                                      "thought": "x"}}],
                                finish_reason="tool_calls"))
        turn = self.backend._send_request(
            system_prompt="sys", history=[], tools=[], model="gpt-4o")
        self.assertEqual(turn.stop_reason, "tool_use")
        # Should produce one ToolUseBlock (no TextBlock since content is None)
        tool_blocks = [b for b in turn.blocks if isinstance(b, ToolUseBlock)]
        self.assertEqual(len(tool_blocks), 1)
        self.assertEqual(tool_blocks[0].args,
                         {"code": "1+1", "thought": "x"})

    def test_tools_translated_to_openai_schema(self):
        spec = ToolSpec(
            name="echo",
            description="echo",
            input_schema={"type": "object",
                          "properties": {"x": {"type": "string"}}},
            handler=lambda a, s: ToolOutput(text="x"),
        )
        self.backend._client.chat.completions.create.return_value = (
            _make_chat_response(text="", finish_reason="stop"))
        self.backend._send_request(
            system_prompt="sys", history=[], tools=[spec], model="gpt-4o")
        kwargs = self.backend._client.chat.completions.create.call_args.kwargs
        sent_tools = kwargs["tools"]
        self.assertEqual(sent_tools[0]["type"], "function")
        self.assertEqual(sent_tools[0]["function"]["name"], "echo")
        self.assertEqual(sent_tools[0]["function"]["parameters"],
                         {"type": "object",
                          "properties": {"x": {"type": "string"}}})

    def test_auth_error_translated_to_llm_auth_error(self):
        import openai as _openai
        fake_auth = type("FakeAuth", (Exception,), {})
        with mock.patch.object(_openai, "AuthenticationError", fake_auth):
            self.backend._client.chat.completions.create.side_effect = (
                fake_auth("401"))
            with self.assertRaises(LLMAuthError):
                self.backend._send_request(system_prompt="s", history=[],
                                            tools=[], model="m")


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run tests and verify they fail**

```bash
pixi run bash -c 'cd tests && python -m unittest test_llm_backends_openai -v'
```

Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Implement `backends/openai.py`**

Create `toksearch/llm/backends/openai.py`:

```python
# Copyright 2024 General Atomics
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""OpenAIBackend — driver for the OpenAI Chat Completions API.

PR 1: blocking (non-streaming) API only.  Tool-arg streaming reassembly is
deferred to a follow-up PR.
"""

import json

import openai

from ..errors import LLMAuthError, LLMBackendError, LLMRateLimitError
from ..messages import (
    Message, TextBlock, ToolResultBlock, ToolUseBlock,
)
from .base import AssistantTurn, _ToolLoopBackend


class OpenAIBackend(_ToolLoopBackend):
    name = "openai"
    default_model = "gpt-4o"

    def __init__(self, api_key: str | None, base_url: str | None = None):
        self._api_key = api_key
        self._base_url = base_url
        self._client = None  # lazy

    def _build_client(self):
        kwargs = {}
        if self._api_key is not None:
            kwargs["api_key"] = self._api_key
        if self._base_url is not None:
            kwargs["base_url"] = self._base_url
        self._client = openai.OpenAI(**kwargs)
        return self._client

    def _ensure_client(self):
        if self._client is None:
            self._build_client()
        return self._client

    # ---- Translation: ToolSpec -> OpenAI tool schema ----

    @staticmethod
    def _spec_to_native(spec) -> dict:
        return {
            "type": "function",
            "function": {
                "name": spec.name,
                "description": spec.description,
                "parameters": spec.input_schema,
            },
        }

    # ---- Translation: our Message list -> OpenAI messages= ----

    @classmethod
    def _history_to_native(cls, system_prompt: str,
                            history: list[Message]) -> list[dict]:
        out = [{"role": "system", "content": system_prompt}]
        for m in history:
            if m.role == "user":
                # Could be a plain text user message OR a tool_result-only user
                # message.  OpenAI represents tool results as separate
                # role="tool" messages, so split them out.
                text_parts = [b.text for b in m.content
                              if isinstance(b, TextBlock)]
                if text_parts:
                    out.append({"role": "user",
                                "content": "\n".join(text_parts)})
                for b in m.content:
                    if isinstance(b, ToolResultBlock):
                        content = b.output
                        if b.is_error:
                            content = "[error]\n" + content
                        out.append({"role": "tool",
                                    "tool_call_id": b.tool_use_id,
                                    "content": content})
            elif m.role == "assistant":
                text_parts = [b.text for b in m.content
                              if isinstance(b, TextBlock)]
                tool_uses = [b for b in m.content
                             if isinstance(b, ToolUseBlock)]
                msg = {"role": "assistant",
                       "content": "\n".join(text_parts) if text_parts else None}
                if tool_uses:
                    msg["tool_calls"] = [{
                        "id": b.id,
                        "type": "function",
                        "function": {"name": b.name,
                                      "arguments": json.dumps(b.args)},
                    } for b in tool_uses]
                out.append(msg)
        return out

    # ---- Translation: OpenAI response -> our ContentBlock list ----

    @staticmethod
    def _response_to_blocks(message) -> list:
        out = []
        if message.content:
            out.append(TextBlock(text=message.content))
        for tc in (message.tool_calls or []):
            try:
                args = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                args = {"_raw": tc.function.arguments}
            out.append(ToolUseBlock(id=tc.id, name=tc.function.name,
                                     args=args))
        return out

    # ---- Stop-reason translation ----

    @staticmethod
    def _stop_reason(finish_reason: str) -> str:
        return "tool_use" if finish_reason == "tool_calls" else "end_turn"

    # ---- Main entrypoint ----

    def _send_request(self, *, system_prompt, history, tools, model,
                      on_text=None) -> AssistantTurn:
        client = self._ensure_client()
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=self._history_to_native(system_prompt, history),
                tools=[self._spec_to_native(t) for t in tools],
            )
        except openai.AuthenticationError as e:
            raise LLMAuthError(
                "OpenAI auth failed. Set OPENAI_API_KEY or pass api_key=... "
                "to the backend.") from e
        except openai.RateLimitError as e:
            raise LLMRateLimitError(str(e)) from e
        except openai.APIConnectionError as e:
            raise LLMBackendError(f"Network error: {e}") from e
        except openai.APIStatusError as e:
            raise LLMBackendError(f"OpenAI API error: {e}") from e
        choice = resp.choices[0]
        blocks = self._response_to_blocks(choice.message)
        text_total = "".join(b.text for b in blocks if isinstance(b, TextBlock))
        if text_total and on_text is not None:
            on_text(text_total)
        return AssistantTurn(blocks=blocks,
                              stop_reason=self._stop_reason(choice.finish_reason))
```

- [ ] **Step 4: Run tests and verify they pass**

```bash
pixi run bash -c 'cd tests && python -m unittest test_llm_backends_openai -v'
```

Expected: 4 tests pass.

- [ ] **Step 5: Re-run the registry tests, which should now pass**

```bash
pixi run bash -c 'cd tests && python -m unittest test_llm_backends_registry -v'
```

Expected: 3 tests pass.

- [ ] **Step 6: Commit**

```bash
git add toksearch/llm/backends/openai.py tests/test_llm_backends_openai.py
git commit -m "Add OpenAIBackend with mocked SDK tests"
```

---

## Task 16: Add the CLI (`toksearch chat` + `toksearch query`)

**Files:**
- Create: `toksearch/llm/cli.py`
- Create: `tests/test_llm_cli.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_llm_cli.py`:

```python
# Copyright 2024 General Atomics
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Tests for the toksearch chat / toksearch query CLI.

The Session class is mocked so the CLI tests verify wiring (subcommand
dispatch, flag plumbing, slash-command handling) without going near a real
backend.
"""

import io
import sys
import unittest
from contextlib import redirect_stdout
from unittest import mock


class TestCliQuery(unittest.TestCase):
    def _run_cli(self, argv, send_returns=None, stdin_text=""):
        from toksearch.llm import cli
        fake_session = mock.MagicMock()
        fake_session.send.return_value = send_returns or mock.MagicMock(
            stop_reason="end_turn", final_text="ok")
        buf = io.StringIO()
        exit_code = None
        with mock.patch.object(sys, "argv", argv), \
                mock.patch.object(cli, "build_session",
                                  return_value=fake_session) as build, \
                mock.patch.object(sys, "stdin",
                                  new=io.StringIO(stdin_text)), \
                redirect_stdout(buf):
            try:
                cli.main()
            except SystemExit as e:
                exit_code = e.code
        return fake_session, build, buf.getvalue(), exit_code

    def test_query_dispatches_send(self):
        sess, _, _, _ = self._run_cli(
            ["toksearch", "query", "hello"])
        sess.send.assert_called_once()
        prompt = sess.send.call_args.args[0]
        self.assertEqual(prompt, "hello")

    def test_query_backend_flag_forwarded_to_build(self):
        _, build, _, _ = self._run_cli(
            ["toksearch", "query", "--backend", "openai", "hi"])
        # build_session called with args namespace; check the attribute
        ns = build.call_args.args[0]
        self.assertEqual(ns.backend, "openai")

    def test_query_max_iterations_flag(self):
        _, build, _, _ = self._run_cli(
            ["toksearch", "query", "-n", "3", "hi"])
        ns = build.call_args.args[0]
        self.assertEqual(ns.max_iterations, 3)


class TestCliChatSlashCommands(unittest.TestCase):
    def _run_chat(self, stdin_text):
        from toksearch.llm import cli
        fake_session = mock.MagicMock()
        fake_session.send.return_value = mock.MagicMock(
            stop_reason="end_turn", final_text="agent says hi")
        buf = io.StringIO()
        exit_code = None
        with mock.patch.object(sys, "argv", ["toksearch", "chat"]), \
                mock.patch.object(cli, "build_session",
                                  return_value=fake_session), \
                mock.patch.object(sys, "stdin",
                                  new=io.StringIO(stdin_text)), \
                redirect_stdout(buf):
            try:
                cli.main()
            except SystemExit as e:
                exit_code = e.code
        return fake_session, buf.getvalue(), exit_code

    def test_chat_sends_each_nonempty_line(self):
        sess, out, _ = self._run_chat("hello\nhow are you?\n")
        self.assertEqual(sess.send.call_count, 2)

    def test_slash_quit_exits_loop(self):
        sess, out, _ = self._run_chat("hello\n/quit\nignored\n")
        self.assertEqual(sess.send.call_count, 1)

    def test_slash_reset_calls_session_reset(self):
        sess, out, _ = self._run_chat("/reset\n")
        sess.reset.assert_called_once()

    def test_slash_help_prints(self):
        sess, out, _ = self._run_chat("/help\n")
        self.assertIn("/help", out)
        self.assertIn("/reset", out)
        self.assertIn("/quit", out)

    def test_eof_exits_cleanly(self):
        # Empty stdin = immediate EOF
        sess, out, exit_code = self._run_chat("")
        self.assertIn(exit_code, (None, 0))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run tests and verify they fail**

```bash
pixi run bash -c 'cd tests && python -m unittest test_llm_cli -v'
```

Expected: `ModuleNotFoundError: No module named 'toksearch.llm.cli'`.

- [ ] **Step 3: Implement `cli.py`**

Create `toksearch/llm/cli.py`:

```python
# Copyright 2024 General Atomics
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Command-line interface for toksearch.llm.

Subcommands:
- ``toksearch query "<prompt>"`` — one-shot; runs a single turn and prints output.
- ``toksearch chat`` — interactive REPL (plain ``input()``; prompt_toolkit-based
  UI is deferred to a follow-up PR).

The CLI delegates everything substantive to ``Session``.  ``build_session(args)``
is the seam tests mock to avoid constructing a real backend.
"""

import argparse
import sys

from .config import load_config
from .errors import LLMConfigError, LLMError
from .presets import resolve_preset
from .backends import get_backend_class
from .session import Session


def build_session(args) -> Session:
    """Construct a Session from parsed CLI args."""
    cfg = load_config()
    backend_name = args.backend or cfg.backend or "anthropic"
    preset = resolve_preset(backend_name, cfg)
    backend_cls = get_backend_class(preset.backend)
    api_key = _resolve_api_key(preset, cfg)
    backend = backend_cls(api_key=api_key, base_url=preset.base_url)
    return Session(
        backend=backend,
        model=args.model or preset.model,
        max_iterations=args.max_iterations or cfg.max_iterations,
    )


def _resolve_api_key(preset, cfg) -> str | None:
    if preset.backend == "anthropic":
        return cfg.anthropic_api_key
    if preset.backend == "openai":
        return cfg.openai_api_key
    return None


# ----------------------------------------------------------------------
# `toksearch query`
# ----------------------------------------------------------------------

def _print_tool_call(call):
    print(f"\n[{call.name}] {call.thought or ''}".rstrip())
    code = call.args.get("code")
    if code:
        for line in code.splitlines():
            print(f"  {line}")


def _print_tool_result(result):
    label = "[output]" if not result.is_error else "[error]"
    body = result.output or ""
    for line in body.splitlines():
        print(f"  {line}")
    if not body:
        print(f"{label} (empty)")


def do_query(args):
    session = build_session(args)
    try:
        result = session.send(
            args.query,
            on_text=lambda t: print(t, end="", flush=True),
            on_tool_call=_print_tool_call,
            on_tool_result=_print_tool_result,
        )
    except LLMError as e:
        print(f"\nerror: {e}", file=sys.stderr)
        sys.exit(1)
    print()  # final newline
    sys.exit(0 if result.stop_reason == "end_turn" else 2)


# ----------------------------------------------------------------------
# `toksearch chat`
# ----------------------------------------------------------------------

_HELP_TEXT = """\
Slash commands:
  /help   show this help
  /reset  clear history and namespace
  /quit   exit the chat (ctrl-D also works)
"""


def do_chat(args):
    session = build_session(args)
    print(f"toksearch chat (backend: {session.backend.name}, "
          f"model: {session.model})")
    print("Type /help for commands. Ctrl-D to exit.\n")
    while True:
        try:
            line = input("you> ").strip()
        except EOFError:
            print()
            return
        if not line:
            continue
        if line == "/quit":
            return
        if line == "/help":
            print(_HELP_TEXT)
            continue
        if line == "/reset":
            session.reset()
            print("(session cleared)")
            continue
        try:
            session.send(
                line,
                on_tool_call=_print_tool_call,
                on_tool_result=_print_tool_result,
            )
        except LLMError as e:
            print(f"error: {e}", file=sys.stderr)
            continue
        print()  # spacer between turns


# ----------------------------------------------------------------------
# main
# ----------------------------------------------------------------------

def _add_common(p):
    p.add_argument("--backend", default=None,
                   help="Backend / preset name (anthropic, openai, or a user "
                        "preset from ~/.fdp/config.toml).")
    p.add_argument("--model", default=None,
                   help="Override the preset's default model.")
    p.add_argument("-n", "--max-iterations", type=int, default=None,
                   help="Cap on tool-call rounds per turn.")


def main(argv=None):
    parser = argparse.ArgumentParser(prog="toksearch")
    sub = parser.add_subparsers(dest="command", required=True)

    qp = sub.add_parser("query", help="One-shot natural-language query.")
    qp.add_argument("query", help="The prompt (quote it on the shell).")
    _add_common(qp)
    qp.set_defaults(func=do_query)

    cp = sub.add_parser("chat", help="Interactive conversation.")
    _add_common(cp)
    cp.set_defaults(func=do_chat)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests and verify they pass**

```bash
pixi run bash -c 'cd tests && python -m unittest test_llm_cli -v'
```

Expected: 8 tests pass.

- [ ] **Step 5: Commit**

```bash
git add toksearch/llm/cli.py tests/test_llm_cli.py
git commit -m "Add toksearch chat / toksearch query CLI"
```

---

## Task 17: Wire up the public package surface and console script

**Files:**
- Modify: `toksearch/llm/__init__.py`
- Modify: `setup.py`
- Modify: `pyproject.toml`
- Modify: `tests/test_llm_package.py`

- [ ] **Step 1: Update the package `__init__.py` with public re-exports**

Replace `toksearch/llm/__init__.py` with:

```python
# Copyright 2024 General Atomics
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""toksearch.llm — Conversational LLM interface for TokSearch.

See ``docs/superpowers/specs/2026-05-18-toksearch-llm-design.md``.

Quickstart
==========

::

    from toksearch.llm import Session
    from toksearch.llm.backends.anthropic import AnthropicBackend

    sess = Session(backend=AnthropicBackend(api_key="sk-..."))
    result = sess.send("Fetch ip for shot 200000 and report peak in MA.",
                       on_tool_call=lambda c: print(c.name, c.thought),
                       on_tool_result=lambda r: print(r.output))
    print(result.final_text)
"""

from .errors import (
    LLMError,
    LLMConfigError,
    LLMAuthError,
    LLMBackendError,
    LLMRateLimitError,
    LLMUserAbort,
)
from .events import TextDelta, ToolCall, ToolResult, TurnComplete
from .config import Config, load_config
from .presets import Preset, BUILTIN_PRESETS, resolve_preset
from .session import Session

__all__ = [
    "Session",
    "Config", "load_config",
    "Preset", "BUILTIN_PRESETS", "resolve_preset",
    "TextDelta", "ToolCall", "ToolResult", "TurnComplete",
    "LLMError", "LLMConfigError", "LLMAuthError",
    "LLMBackendError", "LLMRateLimitError", "LLMUserAbort",
]
```

- [ ] **Step 2: Update `setup.py`**

In `setup.py`, find the `package_data` line and update it to include `cli.py`
metadata-as-data only if necessary. Then add an `entry_points` argument to
`setup(...)` immediately after the `scripts=[...]` line:

Replace:

```python
    scripts=['scripts/toksearch_submit', 'scripts/toksearch_shape', 'scripts/toksearch_example.py'],
    # this package will read some included files in runtime, avoid installing it as .zip
    zip_safe=False,
```

with:

```python
    scripts=['scripts/toksearch_submit', 'scripts/toksearch_shape', 'scripts/toksearch_example.py'],
    entry_points={
        "console_scripts": [
            "toksearch = toksearch.llm.cli:main",
        ],
    },
    # this package will read some included files in runtime, avoid installing it as .zip
    zip_safe=False,
```

- [ ] **Step 3: Update `pyproject.toml` with the `[llm]` optional extra**

Add to the end of `pyproject.toml`:

```toml
[project]
name = "toksearch"
dynamic = ["version"]

[project.optional-dependencies]
llm = ["anthropic>=0.34", "openai>=1.50", "matplotlib"]
```

Note: `[project]` may already exist; if so, only add the `optional-dependencies`
table. Do NOT duplicate the `name`/`dynamic` keys.

- [ ] **Step 4: Re-run pip install to register the console script**

```bash
pixi run build
```

Expected: editable install picks up the new entry point. Verify:

```bash
pixi run which toksearch
pixi run toksearch query --help
```

Expected: the `toksearch` script exists; `--help` shows the `query` and `chat`
subcommands.

- [ ] **Step 5: Extend `test_llm_package.py` with the public-surface check**

Replace `tests/test_llm_package.py` with:

```python
# Copyright 2024 General Atomics
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Public-surface tests for toksearch.llm."""

import unittest


class TestPublicSurface(unittest.TestCase):
    def test_session_importable(self):
        from toksearch.llm import Session  # noqa: F401

    def test_event_types_importable(self):
        from toksearch.llm import (
            TextDelta, ToolCall, ToolResult, TurnComplete,
        )  # noqa: F401

    def test_error_types_importable(self):
        from toksearch.llm import (
            LLMError, LLMConfigError, LLMAuthError,
            LLMBackendError, LLMRateLimitError, LLMUserAbort,
        )  # noqa: F401

    def test_config_and_presets_importable(self):
        from toksearch.llm import (
            Config, load_config, Preset, BUILTIN_PRESETS, resolve_preset,
        )  # noqa: F401


class TestConsoleScript(unittest.TestCase):
    def test_cli_main_callable(self):
        from toksearch.llm.cli import main
        self.assertTrue(callable(main))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 6: Run the full LLM test suite**

```bash
pixi run bash -c 'cd tests && python -m unittest discover -p "test_llm*" -v'
```

Expected: all LLM tests pass (errors, events, messages, config, presets,
tools, prompts, backends_base, backends_fake, backends_registry,
backends_anthropic, backends_openai, session, cli, package).

- [ ] **Step 7: Run the full project test suite to ensure nothing else broke**

```bash
pixi run bash -c 'cd tests && ./testit --mock --noptdata --nod3drdb'
```

Expected: all tests pass. (`--mock --noptdata --nod3drdb` skips integration-only
tests in the existing suite.)

- [ ] **Step 8: Commit**

```bash
git add toksearch/llm/__init__.py setup.py pyproject.toml tests/test_llm_package.py
git commit -m "Wire toksearch.llm public surface and console script"
```

---

## Task 18: Manual smoke test (operator-run, not automated)

**Files:** none modified.

These steps require a live `ANTHROPIC_API_KEY` (or `OPENAI_API_KEY`).  Skip
if absent; the unit tests cover wiring.

- [ ] **Step 1: Verify `toksearch query` works end-to-end with Anthropic**

```bash
ANTHROPIC_API_KEY=sk-ant-... pixi run toksearch query --backend anthropic \
    "In Python, set x = 1 + 1 and report the value."
```

Expected: per-iteration tool-call lines (`[run_python] ...` then `[output] ...`),
then a final text response mentioning `2`.

- [ ] **Step 2: Verify `toksearch chat` REPL works**

```bash
ANTHROPIC_API_KEY=sk-ant-... pixi run toksearch chat --backend anthropic
```

Expected: greeting line, then `you> ` prompt. Type `/help` to see the slash
commands. Type a few prompts that build state across turns (e.g. "set x = 5"
then "what is x squared?"). Verify that the second turn references the
namespace set in the first. Type `/reset` to clear; type `/quit` to exit.

- [ ] **Step 3: Verify `toksearch query` works end-to-end with OpenAI**

```bash
OPENAI_API_KEY=sk-... pixi run toksearch query --backend openai \
    "In Python, compute and print 2**10."
```

Expected: per-iteration tool-call lines, final text mentioning `1024`.

- [ ] **Step 4: Verify backend selection from env**

```bash
FDP_LLM_BACKEND=openai OPENAI_API_KEY=sk-... pixi run toksearch query \
    "What is 7 * 6?"
```

Expected: same as Step 3 — the env var picks OpenAI without a `--backend` flag.

If any step fails, the failure should be in `cli.py` wiring or the backend
translation layer.  Both are covered by unit tests; a new smoke failure means
we missed a case worth adding to the unit suite.

---

## Self-Review Notes

Coverage against the PR 1 scope:
- ✓ `Session` class with sync `send()` and event callbacks → Task 13.
- ✓ Event types (`TextDelta`, `ToolCall`, `ToolResult`, `TurnComplete`) → Task 3.
- ✓ Provider-neutral history (`Message`, `ContentBlock` family) → Task 4.
- ✓ `LLMError` hierarchy → Task 2.
- ✓ `Config` + `load_config()` with TOML and env overlay → Task 5.
- ✓ Built-in `anthropic` / `openai` presets + `resolve_preset` → Task 6.
- ✓ `ToolSpec`, `run_python`, `lookup_docs`, skill discovery → Tasks 7-8.
- ✓ System-prompt assembly → Task 9.
- ✓ `Backend` ABC, `AssistantTurn`, `Callbacks`, `_ToolLoopBackend` → Task 10.
- ✓ `FakeBackend` testing seam → Task 11.
- ✓ Backend registry with lazy lookup → Task 12.
- ✓ `AnthropicBackend` (also serves AmSC via preset `base_url`) → Task 14.
- ✓ `OpenAIBackend` → Task 15.
- ✓ `toksearch chat` + `toksearch query` CLI with `/help` `/reset` `/quit` → Task 16.
- ✓ Console script + `[llm]` optional extra → Task 17.
- ✓ Public `toksearch.llm` surface → Task 17.

Explicitly deferred (called out at top of plan):
- Streaming text deltas (incremental on_text); `_send_request` is blocking.
- `prompt_toolkit`-based REPL; plain `input()` only in PR 1.
- Slash commands `/save`, `/history`, `/namespace`, `/backend`, `/model`.
- `--confirm` CLI flag (but the `confirm` callable on `Session.send` ships).
- Entry-point discovery for namespace/skills/presets contributors (PR 2).
- `ClaudeSDKBackend` (PR 3).
- AmSC preset registration (PR 4; users can still use AmSC by defining a user
  preset in `~/.fdp/config.toml`).
- Connectivity test command (`toksearch llm test`).
- Anthropic `cache_control` on system prompt.
- Retry-with-backoff in `_send_request` (deferred along with streaming).

Each deferred item is additive against the PR 1 surface — no refactor required
to add it later.
