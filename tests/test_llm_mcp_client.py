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


@pytest.mark.timeout(60)
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


@pytest.mark.timeout(60)
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


@pytest.mark.timeout(60)
def test_bogus_command_raises_skills_error():
    with pytest.raises(LLMSkillsError):
        SkillsMcpClient(command=["this-command-does-not-exist-xyz"])
