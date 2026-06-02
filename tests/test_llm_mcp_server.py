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
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

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


def test_packages_none_includes_all_entry_point_dirs(monkeypatch):
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
        skills = discover_filtered_skills(packages=None)
        assert {"kept", "dropped"} <= set(skills)


@pytest.mark.timeout(30)
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
