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

import asyncio
import atexit
import os
import sys
import threading
from concurrent.futures import Future
from dataclasses import dataclass
from pathlib import Path

from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

from ..errors import LLMSkillsError


@dataclass(frozen=True)
class SkillMeta:
    """Catalog entry for a skill (name + description; body fetched lazily)."""

    name: str
    description: str


class SkillsMcpClient:
    """Spawn ``python -m toksearch.llm.mcp`` and query it synchronously.

    A single owner coroutine on a daemon-thread event loop owns the MCP async
    contexts for their whole lifetime, so they are entered and exited in the
    same task (required by anyio cancel scopes).  Public methods enqueue
    request thunks and block on the result.

    When ``command`` is provided, ``extra_dirs`` and ``packages`` are ignored
    (it is a full override intended for testing).
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
        self._alive = False
        self._params = self._build_params(extra_dirs, packages, command)
        ready: Future = Future()
        asyncio.run_coroutine_threadsafe(self._owner(ready), self._loop)
        self._closed = False
        try:
            ready.result(timeout=30)
        except Exception as e:
            self._loop.call_soon_threadsafe(self._loop.stop)
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
        close_fut = None
        try:
            async with stdio_client(self._params) as (r, w):
                async with ClientSession(r, w) as sess:
                    await sess.initialize()
                    self._alive = True
                    self._loop.call_soon_threadsafe(ready.set_result, sess)
                    while True:
                        fut, thunk = await self._queue.get()
                        if thunk is None:        # close sentinel
                            close_fut = fut
                            break
                        try:
                            res = await thunk(sess)
                            self._loop.call_soon_threadsafe(fut.set_result, res)
                        except Exception as e:   # noqa: BLE001 -- relay to caller
                            self._loop.call_soon_threadsafe(fut.set_exception, e)
            # contexts have now exited (subprocess torn down)
        except Exception as e:   # noqa: BLE001 -- startup or transport failure
            if not ready.done():
                self._loop.call_soon_threadsafe(ready.set_exception, e)
        finally:
            self._alive = False
            # Unblock close() (if it sent the sentinel) AFTER teardown.
            if close_fut is not None and not close_fut.done():
                self._loop.call_soon_threadsafe(close_fut.set_result, None)
            # Fail any thunks still queued by other callers.
            self._drain_pending()
            self._loop.call_soon_threadsafe(self._loop.stop)

    def _drain_pending(self) -> None:
        if self._queue is None:
            return
        while not self._queue.empty():
            try:
                fut, _thunk = self._queue.get_nowait()
            except Exception:   # noqa: BLE001
                break
            if not fut.done():
                self._loop.call_soon_threadsafe(
                    fut.set_exception,
                    LLMSkillsError("skills MCP server has exited"))

    def _call(self, thunk):
        if not self._alive:
            raise LLMSkillsError("skills MCP server is not running")
        fut: Future = Future()
        self._loop.call_soon_threadsafe(self._queue.put_nowait, (fut, thunk))
        return fut.result(timeout=60)

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
        if self._alive and self._queue is not None:
            try:
                self._call(None)  # sentinel: owner exits contexts in-task,
                                  # then stops the loop in its finally
            except Exception:   # noqa: BLE001
                pass
        else:
            # Owner never became alive (startup failed) — just stop the loop.
            self._loop.call_soon_threadsafe(self._loop.stop)
        atexit.unregister(self.close)
