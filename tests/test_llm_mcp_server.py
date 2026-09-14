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
"""Tests for the standalone skills MCP server (in-memory, no subprocess)."""

import asyncio
import os
import sys
import unittest
from contextlib import asynccontextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

try:
    # mcp 1.x: one-call in-memory client/server bridge.
    from mcp.shared.memory import (
        create_connected_server_and_client_session as _connect_v1)
except ImportError:
    # mcp >= 2.0: the helper is gone; wire the memory streams by hand.
    _connect_v1 = None
    import anyio
    from mcp import ClientSession
    from mcp.shared.memory import create_client_server_memory_streams

from toksearch.llm.mcp.server import build_server, discover_filtered_skills


def _lowlevel(server):
    """The wrapped lowlevel server: _mcp_server (1.x) / _lowlevel_server (2.x)."""
    return getattr(server, "_mcp_server", None) or server._lowlevel_server


def _is_error(result):
    """CallToolResult error flag: isError (mcp 1.x) / is_error (2.x)."""
    return getattr(result, "isError", None) if hasattr(result, "isError") \
        else result.is_error


def _mime_type(contents):
    """Resource contents mime type: mimeType (mcp 1.x) / mime_type (2.x)."""
    return getattr(contents, "mimeType", None) if hasattr(contents, "mimeType") \
        else contents.mime_type


@asynccontextmanager
async def connect(server):
    """Yield an un-initialized in-memory ClientSession for either mcp major."""
    low = _lowlevel(server)
    if _connect_v1 is not None:
        async with _connect_v1(low) as client:
            yield client
        return
    async with create_client_server_memory_streams() as (client_s, server_s):
        client_read, client_write = client_s
        server_read, server_write = server_s
        async with anyio.create_task_group() as tg:
            async def _serve():
                await low.run(server_read, server_write,
                              low.create_initialization_options(),
                              raise_exceptions=True)
            tg.start_soon(_serve)
            async with ClientSession(client_read, client_write) as client:
                yield client
            tg.cancel_scope.cancel()


def _make_skill(root: Path, name: str, description: str, body: str) -> None:
    d = root / name
    d.mkdir()
    (d / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: {description}\n---\n\n{body}\n")


class TestDiscoverFilteredSkills(unittest.TestCase):
    def test_merges_extra_dirs(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _make_skill(root, "alpha", "Alpha skill", "ALPHA BODY")
            skills = discover_filtered_skills(extra_dirs=[root], packages=[])
            # packages=[] excludes all entry-point skills; extra_dirs still load.
            self.assertIn("alpha", skills)
            self.assertEqual(skills["alpha"].description, "Alpha skill")
            self.assertIn("ALPHA BODY", skills["alpha"].body)

    def test_packages_filter_applies_to_entry_point_dirs(self):
        # Two fake entry-point dirs; packages= keeps only one.
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            keep = root / "keep_pkg"
            keep.mkdir()
            drop = root / "drop_pkg"
            drop.mkdir()
            _make_skill(keep, "kept", "Kept", "KEPT BODY")
            _make_skill(drop, "dropped", "Dropped", "DROPPED BODY")
            with mock.patch(
                "toksearch.llm.mcp.server.discover_skill_dirs",
                return_value=[("aaa", keep), ("bbb", drop)],
            ):
                skills = discover_filtered_skills(packages=["aaa"])
            self.assertEqual(set(skills), {"kept"})

    def test_packages_none_includes_all_entry_point_dirs(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            keep = root / "keep_pkg"
            keep.mkdir()
            drop = root / "drop_pkg"
            drop.mkdir()
            _make_skill(keep, "kept", "Kept", "KEPT BODY")
            _make_skill(drop, "dropped", "Dropped", "DROPPED BODY")
            with mock.patch(
                "toksearch.llm.mcp.server.discover_skill_dirs",
                return_value=[("aaa", keep), ("bbb", drop)],
            ):
                skills = discover_filtered_skills(packages=None)
            self.assertTrue({"kept", "dropped"} <= set(skills))


class TestServerInMemory(unittest.TestCase):
    def test_resources_list_and_read(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _make_skill(root, "alpha", "Alpha skill", "ALPHA BODY")
            server = build_server(extra_dirs=[root], packages=[])

            async def go():
                async with connect(server) as client:
                    await client.initialize()
                    listed = await client.list_resources()
                    uris = {str(r.uri): r for r in listed.resources}
                    self.assertIn("skill://alpha", uris)
                    self.assertEqual(
                        uris["skill://alpha"].description, "Alpha skill")
                    read = await client.read_resource("skill://alpha")
                    self.assertIn("ALPHA BODY", read.contents[0].text)
                    self.assertEqual(_mime_type(read.contents[0]),
                                     "text/markdown")
            asyncio.run(go())

    def test_read_skill_tool_ok_and_unknown(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _make_skill(root, "alpha", "Alpha skill", "ALPHA BODY")
            server = build_server(extra_dirs=[root], packages=[])

            async def go():
                async with connect(server) as client:
                    await client.initialize()
                    ok = await client.call_tool(
                        "read_skill", {"skill_name": "alpha"})
                    self.assertIs(_is_error(ok), False)
                    self.assertIn("ALPHA BODY", ok.content[0].text)
                    bad = await client.call_tool(
                        "read_skill", {"skill_name": "nope"})
                    self.assertIs(_is_error(bad), True)
                    self.assertIn("Unknown skill", bad.content[0].text)
            asyncio.run(go())


class TestStdioLaunch(unittest.TestCase):
    def test_module_launches_over_stdio(self):
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

            # wait_for guards CI against a wedged subprocess (was
            # pytest.mark.timeout before the unittest conversion).
            uris = asyncio.run(asyncio.wait_for(go(), timeout=30))
            self.assertIn("skill://alpha", uris)


if __name__ == "__main__":
    unittest.main()
