# Skills MCP Server Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Serve `toksearch.llm.skills` from a standalone stdio MCP server (`python -m toksearch.llm.mcp`) exposing each `SKILL.md` as a `skill://<name>` resource plus a `read_skill` tool, and make `Session` consume it as an MCP client uniformly across all backends.

**Architecture:** A new `toksearch/llm/mcp/` subpackage. `server.py` builds a `FastMCP` server from entry-point-discovered skill dirs (+ extra dirs, + package filter). `__main__.py` runs it over stdio. `client.py` wraps the async stdio client behind a sync facade using a single long-lived "owner" coroutine on a daemon-thread event loop (entering and exiting the MCP async contexts in the *same task* to satisfy anyio cancel-scope rules). `Session` spawns one client at construction, builds its prompt catalog from `resources/list`, and routes `lookup_docs` through `resources/read`. No in-process consumption fallback: spawn failure raises `LLMSkillsError`.

**Tech Stack:** Python 3.11, `mcp` 1.27 (`FastMCP`, `mcp.client.stdio`, `mcp.shared.memory` for in-memory tests), `asyncio`, `unittest` (existing test style).

**Spec:** `docs/superpowers/specs/2026-06-02-skills-mcp-server-design.md`

---

## File Structure

| File | Responsibility |
|------|----------------|
| `toksearch/llm/mcp/__init__.py` | Package exports: `build_server`, `SkillsMcpClient`, `SkillMeta`. |
| `toksearch/llm/mcp/server.py` | `discover_filtered_skills()` + `build_server()` — pure, in-process testable. |
| `toksearch/llm/mcp/__main__.py` | stdio entry point; reads `TOKSEARCH_SKILL_DIRS` and `TOKSEARCH_SKILL_PACKAGES`. |
| `toksearch/llm/mcp/client.py` | `SkillMeta` dataclass + `SkillsMcpClient` sync facade. |
| `toksearch/llm/errors.py` | add `LLMSkillsError`. |
| `toksearch/llm/session.py` | replace in-process skills block with the MCP client; add `close()`. |
| `toksearch/llm/tools.py` | `lookup_docs` reads via `session._skills_client.read_skill()`. |
| `tests/test_llm_mcp_server.py` | in-memory server tests (resources, tool, filtering). |
| `tests/test_llm_mcp_client.py` | real-subprocess client tests. |
| `tests/test_llm_session.py` | update skills tests to MCP path (real temp dirs). |
| `tests/test_llm_tools.py` | update `lookup_docs` tests to client-backed path. |

**Open question resolution (from spec §Open questions #1):** the `packages` filter is applied **server-side** via a `TOKSEARCH_SKILL_PACKAGES` env var (os.pathsep-delimited entry-point names), not via MCP resource `meta`. The server already gets `(entry_point_name, dir)` pairs from `discover_skill_dirs()`, so it filters there before registering resources. The resource URI scheme stays `skill://<name>`.

---

## Task 1: `LLMSkillsError` and `SkillMeta`

**Files:**
- Modify: `toksearch/llm/errors.py`
- Create: `toksearch/llm/mcp/__init__.py`
- Create: `toksearch/llm/mcp/client.py` (SkillMeta only in this task)
- Test: `tests/test_llm_errors.py`

- [ ] **Step 1: Read the current errors module to match style**

Run: `sed -n '1,60p' toksearch/llm/errors.py`
Expected: see the `LLMError` base class and subclasses (`LLMConfigError`, etc.).

- [ ] **Step 2: Write the failing test**

Add to `tests/test_llm_errors.py`:

```python
def test_skills_error_is_llm_error():
    from toksearch.llm.errors import LLMError, LLMSkillsError
    assert issubclass(LLMSkillsError, LLMError)
    with pytest.raises(LLMError):
        raise LLMSkillsError("boom")
```

(If the file uses `unittest.TestCase`, add the equivalent method asserting `issubclass` and `self.assertRaises`.)

- [ ] **Step 3: Run test to verify it fails**

Run: `pixi run pytest tests/test_llm_errors.py -k skills_error -v`
Expected: FAIL with `ImportError: cannot import name 'LLMSkillsError'`.

- [ ] **Step 4: Add the error class**

In `toksearch/llm/errors.py`, after the last existing subclass:

```python
class LLMSkillsError(LLMError):
    """Raised when the skills MCP server cannot be spawned or queried."""
```

Also add `"LLMSkillsError"` to that module's `__all__` if it has one.

- [ ] **Step 5: Create `SkillMeta` and the package init**

Create `toksearch/llm/mcp/client.py`:

```python
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
"""Sync client facade over the standalone skills MCP server."""

from dataclasses import dataclass


@dataclass(frozen=True)
class SkillMeta:
    """Catalog entry for a skill (name + description; body fetched lazily)."""

    name: str
    description: str
```

Create `toksearch/llm/mcp/__init__.py`:

```python
# (same license header as above)
"""toksearch.llm.mcp — standalone MCP server for documentation skills."""

from .client import SkillMeta, SkillsMcpClient
from .server import build_server

__all__ = ["SkillMeta", "SkillsMcpClient", "build_server"]
```

NOTE: `__init__.py` imports `SkillsMcpClient` and `build_server`, which do not exist yet (Tasks 2 and 4). To keep this task's tests green, temporarily comment out the `from .client import ... SkillsMcpClient` and `from .server import build_server` lines and the corresponding `__all__` entries, leaving only `SkillMeta`. Re-enable them at the end of Task 4.

- [ ] **Step 6: Run tests to verify they pass**

Run: `pixi run pytest tests/test_llm_errors.py -k skills_error -v && pixi run python -c "from toksearch.llm.mcp import SkillMeta; print(SkillMeta('a','b'))"`
Expected: test PASSES; prints `SkillMeta(name='a', description='b')`.

- [ ] **Step 7: Commit**

```bash
git add toksearch/llm/errors.py toksearch/llm/mcp/__init__.py toksearch/llm/mcp/client.py tests/test_llm_errors.py
git commit -m "Add LLMSkillsError and SkillMeta for skills MCP server"
```

---

## Task 2: `build_server` and skill discovery/filtering

**Files:**
- Create: `toksearch/llm/mcp/server.py`
- Test: `tests/test_llm_mcp_server.py`

The server reuses `toksearch.llm.discovery.discover_skill_dirs` (returns `[(entry_point_name, Path), ...]`) and `toksearch.llm.tools.discover_skills` (returns `{name: Skill(name, description, body)}`).

- [ ] **Step 1: Write the failing test for discovery + filtering**

Create `tests/test_llm_mcp_server.py`:

```python
# (license header)
"""Tests for the standalone skills MCP server (in-memory, no subprocess)."""

import asyncio
from pathlib import Path
from tempfile import TemporaryDirectory

from mcp.shared.memory import create_connected_server_and_client_session as connect

from toksearch.llm.mcp.server import build_server, discover_filtered_skills


def _make_skill(root: Path, name: str, description: str, body: str) -> None:
    d = root / name
    d.mkdir()
    (d / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: {description}\n---\n\n{body}\n")


def test_discover_filtered_skills_merges_extra_dirs():
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        _make_skill(root, "alpha", "Alpha skill", "ALPHA BODY")
        skills = discover_filtered_skills(extra_dirs=[root], packages=[])
        # packages=[] excludes all entry-point skills; extra_dirs still load.
        assert "alpha" in skills
        assert skills["alpha"].description == "Alpha skill"
        assert "ALPHA BODY" in skills["alpha"].body
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pixi run pytest tests/test_llm_mcp_server.py -k merges_extra_dirs -v`
Expected: FAIL with `ModuleNotFoundError` / `ImportError` for `discover_filtered_skills`.

- [ ] **Step 3: Implement `server.py`**

Create `toksearch/llm/mcp/server.py`:

```python
# (license header)
"""Standalone skills MCP server.

Discovers ``SKILL.md`` documentation skills from the ``toksearch.llm.skills``
entry-point group (plus extra dirs) and serves each as a ``skill://<name>``
resource, with a ``read_skill`` tool for clients that don't read resources.
"""

from pathlib import Path

from mcp.server.fastmcp import FastMCP

from ..discovery import discover_skill_dirs
from ..tools import Skill, discover_skills


def discover_filtered_skills(
    extra_dirs: list[Path] | None = None,
    packages: list[str] | None = None,
) -> dict[str, Skill]:
    """Return ``{name: Skill}`` from entry-point dirs + extra dirs.

    ``packages`` (entry-point names) filters the entry-point dirs only;
    ``extra_dirs`` are always included.  ``packages=None`` means no filter;
    ``packages=[]`` excludes all entry-point dirs.
    """
    pairs = discover_skill_dirs()  # [(entry_point_name, Path), ...]
    if packages is not None:
        pairs = [(n, d) for n, d in pairs if n in packages]
    dirs = [d for _, d in pairs]
    if extra_dirs:
        dirs.extend(extra_dirs)
    return discover_skills(dirs)


def build_server(
    extra_dirs: list[Path] | None = None,
    packages: list[str] | None = None,
) -> FastMCP:
    """Build a FastMCP server exposing discovered skills as resources + tool."""
    mcp = FastMCP("toksearch-skills")
    skills = discover_filtered_skills(extra_dirs=extra_dirs, packages=packages)

    for name, skill in skills.items():
        def _make_reader(body: str):
            def _read() -> str:
                return body
            return _read
        mcp.resource(
            f"skill://{name}",
            name=name,
            description=skill.description,
            mime_type="text/markdown",
        )(_make_reader(skill.body))

    @mcp.tool(
        description="Read a documentation skill's SKILL.md body by name. "
                    "Call this for detail on a specific toksearch feature.")
    def read_skill(skill_name: str) -> str:
        s = skills.get(skill_name)
        if s is None:
            raise ValueError(
                f"Unknown skill: {skill_name!r}. Available: {sorted(skills)}")
        return s.body

    return mcp
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pixi run pytest tests/test_llm_mcp_server.py -k merges_extra_dirs -v`
Expected: PASS.

- [ ] **Step 5: Add the resource + tool behaviour tests**

Append to `tests/test_llm_mcp_server.py`:

```python
def _run(coro):
    return asyncio.run(coro)


def test_resources_list_and_read():
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        _make_skill(root, "alpha", "Alpha skill", "ALPHA BODY")
        server = build_server(extra_dirs=[root], packages=[])

        async def go():
            async with connect(server._mcp_server) as client:
                await client.initialize()
                listed = await client.list_resources()
                uris = {str(r.uri): r for r in listed.resources}
                assert "skill://alpha" in uris
                assert uris["skill://alpha"].description == "Alpha skill"
                read = await client.read_resource("skill://alpha")
                assert "ALPHA BODY" in read.contents[0].text
                assert read.contents[0].mimeType == "text/markdown"
        _run(go())


def test_read_skill_tool_ok_and_unknown():
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        _make_skill(root, "alpha", "Alpha skill", "ALPHA BODY")
        server = build_server(extra_dirs=[root], packages=[])

        async def go():
            async with connect(server._mcp_server) as client:
                await client.initialize()
                ok = await client.call_tool("read_skill", {"skill_name": "alpha"})
                assert ok.isError is False
                assert "ALPHA BODY" in ok.content[0].text
                bad = await client.call_tool("read_skill", {"skill_name": "nope"})
                assert bad.isError is True
                assert "Unknown skill" in bad.content[0].text
        _run(go())


def test_packages_filter_applies_to_entry_point_dirs(monkeypatch):
    # Two fake entry-point dirs; packages= keeps only one.
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        keep = root / "keep_pkg"; keep.mkdir()
        drop = root / "drop_pkg"; drop.mkdir()
        _make_skill(keep, "kept", "Kept", "KEPT BODY")
        _make_skill(drop, "dropped", "Dropped", "DROPPED BODY")
        monkeypatch.setattr(
            "toksearch.llm.mcp.server.discover_skill_dirs",
            lambda: [("aaa", keep), ("bbb", drop)],
        )
        skills = discover_filtered_skills(packages=["aaa"])
        assert set(skills) == {"kept"}
```

- [ ] **Step 6: Run all server tests**

Run: `pixi run pytest tests/test_llm_mcp_server.py -v`
Expected: all PASS. (The INFO log lines from FastMCP are normal.)

- [ ] **Step 7: Commit**

```bash
git add toksearch/llm/mcp/server.py tests/test_llm_mcp_server.py
git commit -m "Add build_server: skills as MCP resources + read_skill tool"
```

---

## Task 3: stdio entry point (`python -m toksearch.llm.mcp`)

**Files:**
- Create: `toksearch/llm/mcp/__main__.py`
- Test: `tests/test_llm_mcp_server.py` (add a subprocess-launch smoke test)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_llm_mcp_server.py`:

```python
import os
import sys


def test_module_launches_over_stdio():
    """`python -m toksearch.llm.mcp` starts and serves resources via stdio."""
    from mcp import ClientSession
    from mcp.client.stdio import stdio_client, StdioServerParameters

    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        _make_skill(root, "alpha", "Alpha skill", "ALPHA BODY")
        env = dict(os.environ)
        env["TOKSEARCH_SKILL_DIRS"] = str(root)
        env["TOKSEARCH_SKILL_PACKAGES"] = ""  # exclude entry-point skills
        params = StdioServerParameters(
            command=sys.executable,
            args=["-m", "toksearch.llm.mcp"],
            env=env,
        )

        async def go():
            async with stdio_client(params) as (r, w):
                async with ClientSession(r, w) as sess:
                    await sess.initialize()
                    listed = await sess.list_resources()
                    return {str(x.uri) for x in listed.resources}
        uris = asyncio.run(go())
        assert "skill://alpha" in uris
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pixi run pytest tests/test_llm_mcp_server.py -k module_launches -v`
Expected: FAIL — `No module named toksearch.llm.mcp.__main__`.

- [ ] **Step 3: Implement `__main__.py`**

Create `toksearch/llm/mcp/__main__.py`:

```python
# (license header)
"""Run the standalone skills MCP server over stdio.

Usage: ``python -m toksearch.llm.mcp``

Environment:
- ``TOKSEARCH_SKILL_DIRS``: os.pathsep-delimited extra skill directories.
- ``TOKSEARCH_SKILL_PACKAGES``: os.pathsep-delimited entry-point names to keep.
  Unset => no filter (all entry-point skills).  Set-but-empty => exclude all
  entry-point skills (extra dirs still load).
"""

import os
from pathlib import Path

from .server import build_server


def main() -> None:
    raw_dirs = os.environ.get("TOKSEARCH_SKILL_DIRS", "")
    extra_dirs = [Path(p) for p in raw_dirs.split(os.pathsep) if p]

    if "TOKSEARCH_SKILL_PACKAGES" in os.environ:
        raw_pkgs = os.environ["TOKSEARCH_SKILL_PACKAGES"]
        packages = [p for p in raw_pkgs.split(os.pathsep) if p]
    else:
        packages = None

    build_server(extra_dirs=extra_dirs, packages=packages).run("stdio")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pixi run pytest tests/test_llm_mcp_server.py -k module_launches -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add toksearch/llm/mcp/__main__.py tests/test_llm_mcp_server.py
git commit -m "Add stdio entry point for skills MCP server"
```

---

## Task 4: `SkillsMcpClient` sync facade

**Files:**
- Modify: `toksearch/llm/mcp/client.py`
- Modify: `toksearch/llm/mcp/__init__.py` (re-enable the deferred imports)
- Test: `tests/test_llm_mcp_client.py`

The client uses ONE long-lived owner coroutine on a daemon-thread event loop. The owner enters `stdio_client` + `ClientSession`, signals readiness, then services request "thunks" off an `asyncio.Queue`; on a sentinel it exits the contexts (same task — required by anyio). The sync facade enqueues thunks and blocks on `concurrent.futures.Future`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_llm_mcp_client.py`:

```python
# (license header)
"""Tests for SkillsMcpClient against a real stdio subprocess."""

from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from toksearch.llm.errors import LLMSkillsError
from toksearch.llm.mcp.client import SkillsMcpClient


def _make_skill(root: Path, name: str, description: str, body: str) -> None:
    d = root / name
    d.mkdir()
    (d / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: {description}\n---\n\n{body}\n")


def test_list_and_read_round_trip():
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        _make_skill(root, "alpha", "Alpha skill", "ALPHA BODY")
        client = SkillsMcpClient(extra_dirs=[root], packages=[])
        try:
            catalog = client.list_skills()
            assert "alpha" in catalog
            assert catalog["alpha"].description == "Alpha skill"
            assert "ALPHA BODY" in client.read_skill("alpha")
        finally:
            client.close()


def test_unknown_skill_raises():
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        _make_skill(root, "alpha", "Alpha skill", "ALPHA BODY")
        client = SkillsMcpClient(extra_dirs=[root], packages=[])
        try:
            with pytest.raises(Exception):
                client.read_skill("does-not-exist")
        finally:
            client.close()


def test_bogus_command_raises_skills_error():
    with pytest.raises(LLMSkillsError):
        SkillsMcpClient(command=["this-command-does-not-exist-xyz"])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pixi run pytest tests/test_llm_mcp_client.py -v`
Expected: FAIL — `ImportError: cannot import name 'SkillsMcpClient'`.

- [ ] **Step 3: Implement `SkillsMcpClient`**

Append to `toksearch/llm/mcp/client.py` (keep the existing `SkillMeta`):

```python
import asyncio
import atexit
import os
import sys
import threading
from concurrent.futures import Future
from pathlib import Path

from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

from ..errors import LLMSkillsError


class SkillsMcpClient:
    """Spawn ``python -m toksearch.llm.mcp`` and query it synchronously.

    A single owner coroutine on a daemon-thread event loop owns the MCP async
    contexts for their whole lifetime, so they are entered and exited in the
    same task (required by anyio cancel scopes).  Public methods enqueue
    request thunks and block on the result.
    """

    def __init__(
        self,
        extra_dirs: list[Path] | None = None,
        packages: list[str] | None = None,
        command: list[str] | None = None,
    ):
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._loop.run_forever, daemon=True,
            name="toksearch-llm-skills-mcp")
        self._thread.start()
        self._queue: asyncio.Queue | None = None
        self._params = self._build_params(extra_dirs, packages, command)
        ready: Future = Future()
        asyncio.run_coroutine_threadsafe(self._owner(ready), self._loop)
        self._closed = False
        try:
            ready.result(timeout=30)
        except Exception as e:
            self._stop_loop()
            raise LLMSkillsError(
                f"Could not start the skills MCP server: {e}") from e
        atexit.register(self.close)   # reap the child if caller forgets close()

    @staticmethod
    def _build_params(extra_dirs, packages, command) -> StdioServerParameters:
        if command is not None:
            return StdioServerParameters(command=command[0], args=command[1:])
        env = dict(os.environ)
        env["TOKSEARCH_SKILL_DIRS"] = os.pathsep.join(
            str(p) for p in (extra_dirs or []))
        if packages is not None:
            env["TOKSEARCH_SKILL_PACKAGES"] = os.pathsep.join(packages)
        return StdioServerParameters(
            command=sys.executable, args=["-m", "toksearch.llm.mcp"], env=env)

    async def _owner(self, ready: Future) -> None:
        self._queue = asyncio.Queue()
        try:
            async with stdio_client(self._params) as (r, w):
                async with ClientSession(r, w) as sess:
                    await sess.initialize()
                    self._loop.call_soon_threadsafe(ready.set_result, sess)
                    while True:
                        fut, thunk = await self._queue.get()
                        if thunk is None:
                            self._loop.call_soon_threadsafe(fut.set_result, None)
                            break
                        try:
                            res = await thunk(sess)
                            self._loop.call_soon_threadsafe(fut.set_result, res)
                        except Exception as e:   # noqa: BLE001 -- relay to caller
                            self._loop.call_soon_threadsafe(fut.set_exception, e)
        except Exception as e:   # noqa: BLE001
            if not ready.done():
                self._loop.call_soon_threadsafe(ready.set_exception, e)

    def _call(self, thunk):
        fut: Future = Future()
        self._loop.call_soon_threadsafe(self._queue.put_nowait, (fut, thunk))
        return fut.result(timeout=60)

    def _stop_loop(self) -> None:
        self._loop.call_soon_threadsafe(self._loop.stop)

    # ---- public API ----

    def list_skills(self) -> dict[str, SkillMeta]:
        async def thunk(sess):
            listed = await sess.list_resources()
            return {
                r.name: SkillMeta(name=r.name, description=r.description or "")
                for r in listed.resources
            }
        return self._call(thunk)

    def read_skill(self, name: str) -> str:
        async def thunk(sess):
            res = await sess.read_resource(f"skill://{name}")
            return res.contents[0].text
        return self._call(thunk)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._queue is not None:
            try:
                self._call(None)  # sentinel: owner exits contexts in-task
            except Exception:   # noqa: BLE001
                pass
        self._stop_loop()
        try:
            atexit.unregister(self.close)
        except Exception:   # noqa: BLE001
            pass
```

NOTE: `close()` is idempotent (guarded by `self._closed`) and registered with `atexit`, so existing `Session` tests that never call `close()` still get their subprocess reaped at interpreter exit instead of leaking.

NOTE: in `_call`, a `None` thunk is the sentinel; `read_skill`/`list_skills` always pass a coroutine factory, so they never collide with the sentinel.

- [ ] **Step 4: Re-enable deferred imports in `__init__.py`**

Edit `toksearch/llm/mcp/__init__.py` so it imports `SkillMeta`, `SkillsMcpClient` from `.client` and `build_server` from `.server`, with all three in `__all__` (undo the temporary comment-out from Task 1, Step 5).

- [ ] **Step 5: Run client tests to verify they pass**

Run: `pixi run pytest tests/test_llm_mcp_client.py -v`
Expected: all PASS. `test_bogus_command_raises_skills_error` proves spawn failure → `LLMSkillsError`; the others prove round-trip and clean `close()`.

- [ ] **Step 6: Commit**

```bash
git add toksearch/llm/mcp/client.py toksearch/llm/mcp/__init__.py tests/test_llm_mcp_client.py
git commit -m "Add SkillsMcpClient sync facade over stdio skills server"
```

---

## Task 5: `lookup_docs` reads through the client

**Files:**
- Modify: `toksearch/llm/tools.py:182-191` (`_lookup_docs_handler`)
- Test: `tests/test_llm_tools.py:101-118`

- [ ] **Step 1: Update the failing tests**

Replace the `_stub_session` / `test_unknown_skill_is_error` / `test_known_skill_returns_body` block in `tests/test_llm_tools.py` with a client-stub version:

```python
class _StubClient:
    def __init__(self, bodies):
        self._bodies = bodies

    def read_skill(self, name):
        if name not in self._bodies:
            raise ValueError(f"Unknown skill: {name!r}")
        return self._bodies[name]


class TestLookupDocs(unittest.TestCase):  # keep existing class name/style
    def _stub_session(self, bodies):
        from types import SimpleNamespace
        return SimpleNamespace(
            _skills_client=_StubClient(bodies),
            skills={k: None for k in bodies},
        )

    def test_unknown_skill_is_error(self):
        from toksearch.llm.tools import LOOKUP_DOCS
        s = self._stub_session({})
        out = LOOKUP_DOCS.handler({"skill_name": "missing"}, s)
        self.assertTrue(out.is_error)
        self.assertIn("missing", out.text)

    def test_known_skill_returns_body(self):
        from toksearch.llm.tools import LOOKUP_DOCS
        s = self._stub_session({"foo": "FOO BODY"})
        out = LOOKUP_DOCS.handler({"skill_name": "foo"}, s)
        self.assertFalse(out.is_error)
        self.assertIn("FOO BODY", out.text)
```

(Leave `test_lookup_docs_spec_shape` and the `discover_skills`/`parse_skill_md` tests unchanged — those functions still exist and back the server.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `pixi run pytest tests/test_llm_tools.py -k "unknown_skill_is_error or known_skill_returns_body" -v`
Expected: FAIL — handler still reads `session.skills[name].body` (`None.body` → AttributeError or wrong path).

- [ ] **Step 3: Rewrite the handler**

Replace `_lookup_docs_handler` in `toksearch/llm/tools.py`:

```python
def _lookup_docs_handler(args: dict, session) -> ToolOutput:
    name = args["skill_name"]
    try:
        body = session._skills_client.read_skill(name)
    except Exception:   # noqa: BLE001 -- unknown skill or transport error
        available = sorted(session.skills)
        return ToolOutput(
            text=f"Unknown skill: {name!r}. Available: {available}",
            is_error=True,
        )
    return ToolOutput(text=body, is_error=False)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pixi run pytest tests/test_llm_tools.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add toksearch/llm/tools.py tests/test_llm_tools.py
git commit -m "Route lookup_docs through the skills MCP client"
```

---

## Task 6: `Session` consumes the MCP client

**Files:**
- Modify: `toksearch/llm/session.py:75-105` (skills block) and add `close()`
- Test: `tests/test_llm_session.py:120-209`

- [ ] **Step 1: Update/replace the session skills tests**

In `tests/test_llm_session.py`, the existing `test_packages_filter_excludes_others` and `test_empty_packages_list_loads_nothing_discovered` monkeypatch `discovery._entry_points`, which no longer reaches the subprocess for skills. Keep their **namespace** assertions, and add a real-temp-dir skills test. Add this test class:

```python
class TestSessionSkillsViaMcp(unittest.TestCase):
    """Session loads skills through the standalone MCP server."""

    def _make_skill(self, root, name, description, body):
        from pathlib import Path
        d = Path(root) / name
        d.mkdir()
        (d / "SKILL.md").write_text(
            f"---\nname: {name}\ndescription: {description}\n---\n\n{body}\n")

    def test_session_lists_and_reads_skill(self):
        from tempfile import TemporaryDirectory
        from toksearch.llm import Session
        from toksearch.llm.backends.fake import FakeBackend
        with TemporaryDirectory() as tmp:
            self._make_skill(tmp, "demo", "Demo skill", "DEMO BODY")
            from pathlib import Path
            sess = Session(backend=FakeBackend(),
                           extra_skill_dirs=[Path(tmp)],
                           packages=[])  # exclude entry-point skills
            try:
                self.assertIn("demo", sess.skills)
                self.assertEqual(sess.skills["demo"].description, "Demo skill")
                self.assertIn("Demo skill", sess.system_prompt)
                body = sess._skills_client.read_skill("demo")
                self.assertIn("DEMO BODY", body)
            finally:
                sess.close()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pixi run pytest tests/test_llm_session.py -k session_lists_and_reads_skill -v`
Expected: FAIL — `Session` has no `_skills_client` / `close`, and `extra_skill_dirs` still goes through the old in-process path producing `Skill` objects (the assertion on `sess.skills["demo"].description` may pass, but `sess._skills_client` does not exist).

- [ ] **Step 3: Rewrite the skills block in `session.py`**

In `Session.__init__`, replace the imports and the skills block. Update the top-of-method import:

```python
        from .discovery import discover_namespace_contributors
```

(Remove `discover_skill_dirs` from that import — it is now used only inside the server subprocess.)

Replace the `# ---- Skills ... ----` block (current lines ~92-100) with:

```python
        # ---- Skills: standalone MCP server (see docs spec 2026-06-02) ----
        from .mcp.client import SkillsMcpClient
        from .errors import LLMSkillsError
        try:
            self._skills_client = SkillsMcpClient(
                extra_dirs=list(extra_skill_dirs or []),
                packages=packages,
            )
            self.skills = self._skills_client.list_skills()
        except LLMSkillsError:
            raise
        except Exception as e:   # noqa: BLE001
            raise LLMSkillsError(
                f"Failed to load skills via MCP server: {e}") from e
```

Note: `packages` is passed straight through. `packages=None` => no filter; `packages=[]` => only `extra_skill_dirs` skills. This matches the server semantics from Task 2.

- [ ] **Step 4: Add `Session.close()` and keep the client across `reset()`**

Add to the `Session` class (after `reset`):

```python
    def close(self) -> None:
        """Tear down the skills MCP server subprocess."""
        client = getattr(self, "_skills_client", None)
        if client is not None:
            client.close()
            self._skills_client = None
```

Leave `reset()` unchanged (it must NOT close the client — only namespace/history reset).

- [ ] **Step 5: Run the new session test**

Run: `pixi run pytest tests/test_llm_session.py -k session_lists_and_reads_skill -v`
Expected: PASS.

**Perf note:** after this task, *every* `Session(...)` construction spawns the `python -m toksearch.llm.mcp` subprocess (it imports `toksearch` in the child). Existing session tests that build a `Session` without calling `close()` no longer leak thanks to the client's `atexit` reaping (Task 4), but the suite gains one subprocess spawn per Session. This is the accepted tradeoff from the spec's Risks section. If suite wall-time becomes a problem later, a module-scoped fixture sharing one client is a follow-up — not part of this plan.

- [ ] **Step 6: Fix the legacy filter tests**

Run the full session suite: `pixi run pytest tests/test_llm_session.py -v`
For any failure in `test_packages_filter_excludes_others` / `test_empty_packages_list_loads_nothing_discovered` caused by skills assertions, narrow those tests to assert only on `sess.namespace` (namespace discovery is unchanged), and delete skills assertions there (now covered by `TestSessionSkillsViaMcp`). Re-run until green.
Expected: all PASS.

- [ ] **Step 7: Commit**

```bash
git add toksearch/llm/session.py tests/test_llm_session.py
git commit -m "Session loads skills via standalone MCP client; add Session.close()"
```

---

## Task 7: GUI catalog check + end-to-end smoke test

**Files:**
- Test: `tests/test_llm_gui_app.py` (catalog visibility)
- Test: `tests/test_llm_session.py` (e2e fake-backend lookup_docs)

- [ ] **Step 1: Inspect how the GUI test builds a Session**

Run: `grep -n "Session\|skills\|backend" tests/test_llm_gui_app.py | head -30`
Expected: identify the fixture/helper that constructs a `Session` for the app.

- [ ] **Step 2: Write the GUI catalog test**

Add to `tests/test_llm_gui_app.py` (adapt fixture construction to the file's existing style; use a temp skill dir + `FakeBackend` as in Task 6):

```python
def test_gui_session_exposes_mcp_skill_catalog(tmp_path):
    from toksearch.llm import Session
    from toksearch.llm.backends.fake import FakeBackend
    d = tmp_path / "guiskill"
    d.mkdir()
    (d / "SKILL.md").write_text(
        "---\nname: guiskill\ndescription: GUI skill\n---\n\nGUI BODY\n")
    sess = Session(backend=FakeBackend(), extra_skill_dirs=[d], packages=[])
    try:
        # The catalog the GUI renders/uses comes through MCP.
        assert "guiskill" in sess.skills
        assert "GUI skill" in sess.system_prompt
    finally:
        sess.close()
```

- [ ] **Step 3: Write the end-to-end lookup_docs smoke test**

`tests/test_llm_session.py` already defines module-level helpers `_tool_use(name, args, id_="t1")` and `_text(s)` (verified: lines ~28-37) that build `AssistantTurn`s with the correct field shapes. Reuse them. Add to `TestSessionSkillsViaMcp`:

```python
    def test_send_lookup_docs_flows_through_mcp(self):
        from tempfile import TemporaryDirectory
        from pathlib import Path
        from toksearch.llm import Session
        from toksearch.llm.backends.fake import FakeBackend
        with TemporaryDirectory() as tmp:
            self._make_skill(tmp, "demo", "Demo skill", "DEMO BODY")
            turns = [
                _tool_use("lookup_docs", {"skill_name": "demo"}),
                _text("done"),
            ]
            sess = Session(backend=FakeBackend(scripted_turns=turns),
                           extra_skill_dirs=[Path(tmp)], packages=[])
            try:
                results = []
                sess.send("read the demo skill",
                          on_tool_result=lambda r: results.append(r.output))
                assert any("DEMO BODY" in r for r in results)
            finally:
                sess.close()
```

`ToolResult.output` is the field carrying the tool text (verified in `events.py`).

- [ ] **Step 4: Run both tests**

Run: `pixi run pytest tests/test_llm_gui_app.py -k mcp_skill_catalog tests/test_llm_session.py -k send_lookup_docs_flows_through_mcp -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/test_llm_gui_app.py tests/test_llm_session.py
git commit -m "Add GUI catalog and end-to-end lookup_docs MCP tests"
```

---

## Task 8: Full suite, docs, and spec open-question resolution

**Files:**
- Modify: `docs/superpowers/specs/2026-06-02-skills-mcp-server-design.md` (mark open question #1 resolved)
- Modify: `toksearch/docs/llm.md` if it documents skills discovery (check first)

- [ ] **Step 1: Run the entire LLM test suite**

Run: `pixi run pytest tests/ -k llm -v`
Expected: all PASS, no errors, no leaked subprocess warnings. If any test that previously imported `discover_skill_dirs` from `session` breaks, fix the import (it now lives only in `mcp.server`).

- [ ] **Step 2: Manual standalone-server check (the Phase-2 proof)**

Run:
```bash
pixi run python - <<'PY'
import sys, asyncio
from mcp import ClientSession
from mcp.client.stdio import stdio_client, StdioServerParameters
params = StdioServerParameters(command=sys.executable, args=["-m","toksearch.llm.mcp"])
async def go():
    async with stdio_client(params) as (r,w):
        async with ClientSession(r,w) as s:
            await s.initialize()
            res = await s.list_resources()
            print("RESOURCES:", sorted(str(x.uri) for x in res.resources))
asyncio.run(go())
PY
```
Expected: prints the real installed skills as `skill://...` URIs (core toksearch skills, plus `toksearch_d3d` skills if installed). This is "an external MCP client connects to the standalone server" — the Phase-2 success criterion.

- [ ] **Step 3: Resolve spec open question #1**

In the spec file, under "Open questions", append to item 1: `**Resolved (plan):** the packages filter is applied server-side via TOKSEARCH_SKILL_PACKAGES; resource meta is not used. URI scheme remains skill://<name>.`

- [ ] **Step 4: Update `docs/llm.md` if needed**

Run: `grep -n "skill\|lookup_docs\|entry.point" toksearch/docs/llm.md | head`
If it describes in-process skill discovery, add a short note that skills are served via the standalone MCP server (`python -m toksearch.llm.mcp`); the entry-point group remains the discovery source. If `docs/llm.md` does not mention skills discovery mechanics, skip.

- [ ] **Step 5: Commit**

```bash
git add docs/superpowers/specs/2026-06-02-skills-mcp-server-design.md docs/llm.md
git commit -m "Resolve spec open question and document skills MCP server"
```

- [ ] **Step 6: Final full test run**

Run: `pixi run pytest tests/ -k llm`
Expected: green. Implementation complete.

---

## Self-review notes

- **Spec coverage:** standalone stdio server (Tasks 2-3), resources + read_skill tool (Task 2), Session-as-client uniform across backends (Tasks 4-6; claude_sdk untouched because its in-process `lookup_docs` calls `session._execute_tool` → the rewritten handler → the client), entry-points + extra dirs discovery (Task 2), `TOKSEARCH_SKILL_DIRS` config (Task 3), fail-loudly `LLMSkillsError` no fallback (Tasks 1, 6), GUI works against MCP (Task 7), tests mirror existing layout (all tasks), manual standalone proof (Task 8).
- **`packages` filter** (spec open question) resolved server-side via `TOKSEARCH_SKILL_PACKAGES` — consistent across Tasks 2, 3, 4, 6.
- **Type consistency:** `SkillMeta(name, description)` (Task 1) is what `list_skills()` returns (Task 4) and what `Session.skills` holds (Task 6); `build_system_prompt` reads only `.description` (already true). `discover_filtered_skills(extra_dirs, packages)` signature identical in Tasks 2/3. `SkillsMcpClient(extra_dirs, packages, command)` identical in Tasks 4/6.
- **No silent fallback:** the only `except` that swallows is `lookup_docs` (returns an error ToolOutput, matching prior behavior) and `close()` (best-effort teardown). `Session.__init__` re-raises as `LLMSkillsError`.
