# `toksearch.llm` PR 2 — Entry-Point Discovery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task.

**Goal:** Replace PR 1's hardcoded "toksearch is the only contributor" assumption with Python entry-point discovery, so any installed package can contribute namespace bindings, skill directories, and backend presets. This is what enables `toksearch_d3d` (PR 4) to plug in additively without core toksearch knowing about it.

**Architecture:** New module `toksearch/llm/discovery.py` reads three entry-point groups (`toksearch.llm.namespace`, `toksearch.llm.skills`, `toksearch.llm.presets`) via `importlib.metadata`. `Session.__init__` calls discovery at construction time. `resolve_preset` adds a discovered-presets tier between user-presets and built-in. Core `toksearch` declares itself as a contributor in its own `pyproject.toml` entry points. `Session(packages=[...])` and CLI `--package NAME` (repeatable) become user-facing filters that restrict discovered contributors to a named subset.

**Tech Stack:** Python 3.11, `importlib.metadata`, `unittest`, `unittest.mock`, pixi-managed env.

**Reference spec:** `docs/superpowers/specs/2026-05-18-toksearch-llm-design.md` (the `Entry-point discovery` and `Multi-device contributors` sections).

**Branch:** `feat/llm-pr2` off `main` (or off `feat/llm-pr1` if PR 1 hasn't merged yet — pick whichever fits your review flow).

## Scope notes

- Built-in presets (`anthropic`, `openai`) stay in `BUILTIN_PRESETS` as code. Entry points are for site/extra-package presets only (e.g. `amsc` in PR 4).
- Backward compat: `extra_skill_dirs=` and `extra_namespace=` kwargs on `Session.__init__` still work (additive to discovery).
- `packages=None` (default) means "load all discovered contributors." `packages=[]` means "load none." `packages=["toksearch_d3d"]` means "load only the named contributors."
- Discovery cache: results are cached per-process on the discovery module (entry points don't change at runtime).
- No `ClaudeSDKBackend` yet (PR 3); no `amsc` preset yet (PR 4).

---

## File Structure

| File | Action | Purpose |
|---|---|---|
| `toksearch/llm/discovery.py` | Create | `discover_namespace_contributors`, `discover_skill_dirs`, `discover_presets`. Cached, monkeypatch-friendly. |
| `toksearch/__init__.py` | Modify | Add module-level `__llm_description__` so core toksearch's contribution has a sentence in the system-prompt catalog. |
| `toksearch/llm/session.py` | Modify | Replace hardcoded namespace_entries and `[_core_skill_dir()]` with discovery + `packages=` filter. |
| `toksearch/llm/presets.py` | Modify | `resolve_preset` consults discovered presets between user and built-in. |
| `toksearch/llm/cli.py` | Modify | Add `--package NAME` (repeatable) flag, forward to `Session(packages=...)`. |
| `pyproject.toml` | Modify | Add `[project.entry-points."toksearch.llm.namespace"]` and `[project.entry-points."toksearch.llm.skills"]` for core toksearch's self-registration. |
| `tests/test_llm_discovery.py` | Create | Discovery functions with monkeypatched `importlib.metadata.entry_points`. |
| `tests/test_llm_session.py` | Modify | Add tests for `packages=` filter behavior. |
| `tests/test_llm_presets.py` | Modify | Add tests for discovered-preset precedence. |
| `tests/test_llm_cli.py` | Modify | Add test for `--package` flag wiring. |

---

## Task 1: Add `discovery.py` with entry-point loaders

**Files:**
- Create: `toksearch/llm/discovery.py`
- Create: `tests/test_llm_discovery.py`

Use the FULL 13-line Apache header from `setup.py` lines 1-13.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_llm_discovery.py`:

```python
# [13-line Apache header]
"""Tests for entry-point discovery.

Discovery is monkeypatched via ``mock.patch`` on
``toksearch.llm.discovery._entry_points``; tests construct fake
EntryPoint-like objects whose ``.load()`` returns the value the test wants
the discovery to surface.
"""

import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest import mock

from toksearch.llm.discovery import (
    discover_namespace_contributors,
    discover_skill_dirs,
    discover_presets,
    clear_discovery_cache,
)
from toksearch.llm.presets import Preset


def _fake_ep(name, value):
    """Return a fake EntryPoint with `.name` and `.load()`."""
    ep = mock.MagicMock()
    ep.name = name
    ep.load.return_value = value
    return ep


class _Base(unittest.TestCase):
    def setUp(self):
        clear_discovery_cache()

    def tearDown(self):
        clear_discovery_cache()


class TestDiscoverNamespace(_Base):
    def test_returns_module_with_description(self):
        mod = ModuleType("fake_pkg")
        mod.__llm_description__ = "fake description"
        with mock.patch("toksearch.llm.discovery._entry_points",
                        return_value=[_fake_ep("fake_pkg", mod)]):
            result = discover_namespace_contributors()
        self.assertEqual(len(result), 1)
        name, value, desc = result[0]
        self.assertEqual(name, "fake_pkg")
        self.assertIs(value, mod)
        self.assertEqual(desc, "fake description")

    def test_missing_description_defaults_to_empty(self):
        mod = ModuleType("fake_pkg")
        with mock.patch("toksearch.llm.discovery._entry_points",
                        return_value=[_fake_ep("fake_pkg", mod)]):
            result = discover_namespace_contributors()
        self.assertEqual(result[0][2], "")

    def test_no_entry_points_returns_empty(self):
        with mock.patch("toksearch.llm.discovery._entry_points",
                        return_value=[]):
            result = discover_namespace_contributors()
        self.assertEqual(result, [])


class TestDiscoverSkillDirs(_Base):
    def test_path_value(self):
        p = Path("/tmp/fake_skills")
        with mock.patch("toksearch.llm.discovery._entry_points",
                        return_value=[_fake_ep("fake_pkg", p)]):
            result = discover_skill_dirs()
        self.assertEqual(result, [("fake_pkg", p)])

    def test_callable_value_invoked(self):
        p = Path("/tmp/from_callable")
        with mock.patch("toksearch.llm.discovery._entry_points",
                        return_value=[_fake_ep("fake_pkg", lambda: p)]):
            result = discover_skill_dirs()
        self.assertEqual(result, [("fake_pkg", p)])

    def test_string_value_coerced_to_path(self):
        with mock.patch("toksearch.llm.discovery._entry_points",
                        return_value=[_fake_ep("fake_pkg",
                                               "/tmp/string_skills")]):
            result = discover_skill_dirs()
        self.assertEqual(result, [("fake_pkg", Path("/tmp/string_skills"))])


class TestDiscoverPresets(_Base):
    def test_preset_value(self):
        preset = Preset(backend="anthropic", base_url="https://x.example")
        with mock.patch("toksearch.llm.discovery._entry_points",
                        return_value=[_fake_ep("mysite", preset)]):
            result = discover_presets()
        self.assertEqual(result, {"mysite": preset})

    def test_non_preset_value_skipped_with_warning(self):
        with mock.patch("toksearch.llm.discovery._entry_points",
                        return_value=[_fake_ep("broken", "not a preset")]):
            with self.assertLogs("toksearch.llm.discovery", level="WARNING"):
                result = discover_presets()
        self.assertEqual(result, {})


class TestCache(_Base):
    def test_results_are_cached(self):
        mod = ModuleType("fake_pkg")
        mod.__llm_description__ = "x"
        mock_ep = _fake_ep("fake_pkg", mod)
        with mock.patch("toksearch.llm.discovery._entry_points",
                        return_value=[mock_ep]) as patched:
            discover_namespace_contributors()
            discover_namespace_contributors()
            discover_namespace_contributors()
        # Once per cache slot, not three times.
        self.assertEqual(patched.call_count, 1)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run tests, verify failing**

```bash
cd /fusion/projects/dt/sammuli/fdp_dev/repos/toksearch && pixi run bash -c 'cd tests && python -m unittest test_llm_discovery -v'
```

Expected: `ModuleNotFoundError: No module named 'toksearch.llm.discovery'`.

- [ ] **Step 3: Implement `discovery.py`**

Create `toksearch/llm/discovery.py` with FULL Apache header, then:

```python
"""Entry-point discovery for toksearch.llm contributors.

Three entry-point groups are scanned at Session construction time:

- ``toksearch.llm.namespace`` — value resolves to a module/object; bound
  under the entry-point name in the run_python namespace.  A module-level
  ``__llm_description__`` attribute, if present, populates the system-prompt
  catalog line.
- ``toksearch.llm.skills`` — value resolves to a ``Path``, a string path, or
  a no-argument callable returning one.  Pointed-to directory is scanned
  for ``SKILL.md`` files.
- ``toksearch.llm.presets`` — value resolves to a ``Preset`` instance and
  is added to the preset registry.

Results are cached on first call; ``clear_discovery_cache()`` resets them
(intended for tests).
"""

import logging
from importlib.metadata import entry_points
from pathlib import Path
from typing import Callable

from .presets import Preset

_log = logging.getLogger("toksearch.llm.discovery")

_NAMESPACE_CACHE: list | None = None
_SKILLS_CACHE: list | None = None
_PRESETS_CACHE: dict | None = None


def _entry_points(group: str):
    """Indirection for monkeypatching in tests; returns iterable of EntryPoints."""
    return entry_points(group=group)


def clear_discovery_cache() -> None:
    global _NAMESPACE_CACHE, _SKILLS_CACHE, _PRESETS_CACHE
    _NAMESPACE_CACHE = None
    _SKILLS_CACHE = None
    _PRESETS_CACHE = None


def discover_namespace_contributors() -> list[tuple[str, object, str]]:
    """Return ``[(name, value, description), ...]``."""
    global _NAMESPACE_CACHE
    if _NAMESPACE_CACHE is not None:
        return _NAMESPACE_CACHE
    out = []
    for ep in _entry_points("toksearch.llm.namespace"):
        try:
            value = ep.load()
        except Exception as e:
            _log.warning("Failed to load namespace entry point %r: %s",
                          ep.name, e)
            continue
        desc = getattr(value, "__llm_description__", "")
        out.append((ep.name, value, desc))
    _NAMESPACE_CACHE = out
    return out


def discover_skill_dirs() -> list[tuple[str, Path]]:
    """Return ``[(name, dir), ...]`` of contributor skill directories."""
    global _SKILLS_CACHE
    if _SKILLS_CACHE is not None:
        return _SKILLS_CACHE
    out = []
    for ep in _entry_points("toksearch.llm.skills"):
        try:
            value = ep.load()
        except Exception as e:
            _log.warning("Failed to load skills entry point %r: %s",
                          ep.name, e)
            continue
        if callable(value):
            value = value()
        if isinstance(value, str):
            value = Path(value)
        if not isinstance(value, Path):
            _log.warning("Skills entry point %r did not resolve to a Path: %r",
                          ep.name, value)
            continue
        out.append((ep.name, value))
    _SKILLS_CACHE = out
    return out


def discover_presets() -> dict[str, Preset]:
    """Return ``{name: Preset}`` of contributor-supplied presets."""
    global _PRESETS_CACHE
    if _PRESETS_CACHE is not None:
        return _PRESETS_CACHE
    out: dict[str, Preset] = {}
    for ep in _entry_points("toksearch.llm.presets"):
        try:
            value = ep.load()
        except Exception as e:
            _log.warning("Failed to load presets entry point %r: %s",
                          ep.name, e)
            continue
        if not isinstance(value, Preset):
            _log.warning("Presets entry point %r did not resolve to a Preset: %r",
                          ep.name, value)
            continue
        out[ep.name] = value
    _PRESETS_CACHE = out
    return out
```

- [ ] **Step 4: Run tests, verify pass**

Expected: 8 tests pass.

- [ ] **Step 5: Commit**

```bash
git -C /fusion/projects/dt/sammuli/fdp_dev/repos/toksearch add toksearch/llm/discovery.py tests/test_llm_discovery.py
git -C /fusion/projects/dt/sammuli/fdp_dev/repos/toksearch commit -m "$(cat <<'EOF'
Add toksearch.llm entry-point discovery

discover_namespace_contributors, discover_skill_dirs, and discover_presets
read the three entry-point groups via importlib.metadata. Results are
cached per-process; clear_discovery_cache() resets them for tests.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Wire discovery into `Session`

**Files:**
- Modify: `toksearch/llm/session.py`
- Modify: `tests/test_llm_session.py`

- [ ] **Step 1: Add tests for `packages=` filter and discovery integration**

Append to `tests/test_llm_session.py` (before the `if __name__` guard):

```python
class TestSessionDiscovery(unittest.TestCase):
    """Verify Session uses discovery and the packages= filter works."""

    def setUp(self):
        from toksearch.llm.discovery import clear_discovery_cache
        clear_discovery_cache()

    def tearDown(self):
        from toksearch.llm.discovery import clear_discovery_cache
        clear_discovery_cache()

    def test_namespace_includes_discovered_contributors(self):
        from types import ModuleType
        from unittest import mock
        fake = ModuleType("fake_pkg")
        fake.__llm_description__ = "fake description"
        fake_ep = mock.MagicMock()
        fake_ep.name = "fake_pkg"
        fake_ep.load.return_value = fake
        with mock.patch(
            "toksearch.llm.discovery._entry_points",
            side_effect=lambda group: [fake_ep] if group == "toksearch.llm.namespace" else [],
        ):
            backend = FakeBackend(scripted_turns=[_text("ok")])
            sess = Session(backend=backend)
        self.assertIn("fake_pkg", sess.namespace)
        self.assertIs(sess.namespace["fake_pkg"], fake)
        # The catalog line should mention the description.
        self.assertIn("fake description", sess.system_prompt)

    def test_packages_filter_excludes_others(self):
        from types import ModuleType
        from unittest import mock
        a = ModuleType("aaa"); a.__llm_description__ = "a"
        b = ModuleType("bbb"); b.__llm_description__ = "b"
        eps = [mock.MagicMock(name="aaa"), mock.MagicMock(name="bbb")]
        eps[0].name = "aaa"; eps[0].load.return_value = a
        eps[1].name = "bbb"; eps[1].load.return_value = b
        with mock.patch(
            "toksearch.llm.discovery._entry_points",
            side_effect=lambda group: eps if group == "toksearch.llm.namespace" else [],
        ):
            backend = FakeBackend(scripted_turns=[_text("ok")])
            sess = Session(backend=backend, packages=["aaa"])
        self.assertIn("aaa", sess.namespace)
        self.assertNotIn("bbb", sess.namespace)

    def test_empty_packages_list_loads_nothing(self):
        from types import ModuleType
        from unittest import mock
        a = ModuleType("aaa"); a.__llm_description__ = "a"
        ep = mock.MagicMock()
        ep.name = "aaa"; ep.load.return_value = a
        with mock.patch(
            "toksearch.llm.discovery._entry_points",
            side_effect=lambda group: [ep] if group == "toksearch.llm.namespace" else [],
        ):
            backend = FakeBackend(scripted_turns=[_text("ok")])
            sess = Session(backend=backend, packages=[])
        self.assertNotIn("aaa", sess.namespace)
```

- [ ] **Step 2: Verify the new tests fail**

The hardcoded `[("toksearch", "...")]` namespace_entries in `Session.__init__` will produce wrong content for these tests.

- [ ] **Step 3: Modify `session.py`**

Replace `Session.__init__` in `/fusion/projects/dt/sammuli/fdp_dev/repos/toksearch/toksearch/llm/session.py` with:

```python
    def __init__(
        self,
        backend: Backend,
        model: str | None = None,
        max_iterations: int = 20,
        extra_namespace: dict | None = None,
        packages: list[str] | None = None,
        extra_skill_dirs: list[Path] | None = None,
    ):
        from .discovery import (
            discover_namespace_contributors,
            discover_skill_dirs,
        )
        self.backend = backend
        self.model = model or backend.default_model
        self.max_iterations = max_iterations
        # ---- Namespace: defaults + discovered + extras ----
        self.namespace = _default_namespace()
        namespace_entries: list[tuple[str, str]] = []
        for name, value, desc in discover_namespace_contributors():
            if packages is not None and name not in packages:
                continue
            self.namespace[name] = value
            namespace_entries.append((name, desc))
        if extra_namespace:
            self.namespace.update(extra_namespace)
        # ---- Skills: discovered + extras ----
        skill_dirs: list[Path] = []
        for name, d in discover_skill_dirs():
            if packages is not None and name not in packages:
                continue
            skill_dirs.append(d)
        if extra_skill_dirs:
            skill_dirs.extend(extra_skill_dirs)
        self.skills = discover_skills(skill_dirs)
        # ---- Tools (fixed in PR 1) ----
        self.tool_specs = [RUN_PYTHON, LOOKUP_DOCS]
        self._tools_by_name = {t.name: t for t in self.tool_specs}
        # ---- System prompt ----
        self.system_prompt = build_system_prompt(self.skills, namespace_entries)
        self.history: list[Message] = []
```

Also delete the now-unused `_core_skill_dir` helper from `session.py` (it's been replaced by the entry-point contribution from core toksearch, added in Task 4).

- [ ] **Step 4: Verify all tests pass**

```bash
cd /fusion/projects/dt/sammuli/fdp_dev/repos/toksearch && pixi run bash -c 'cd tests && python -m unittest test_llm_session test_llm_discovery -v'
```

Expected: 11 (8 old + 3 new) session tests pass, plus the 8 discovery tests.

Note: the existing `test_namespace_pre_populated` test does not assert `toksearch` is in the namespace because core toksearch's namespace entry-point is registered in Task 4. After Task 4, that test will see `toksearch` in the namespace; for now it should not. The test as written in PR 1 already accommodates this — it only checks `toksearch` and `np` are present, which `_default_namespace` provides.

- [ ] **Step 5: Commit**

```bash
git -C /fusion/projects/dt/sammuli/fdp_dev/repos/toksearch add toksearch/llm/session.py tests/test_llm_session.py
git -C /fusion/projects/dt/sammuli/fdp_dev/repos/toksearch commit -m "$(cat <<'EOF'
Wire entry-point discovery into Session

Session.__init__ now loads namespace contributors and skill dirs from the
toksearch.llm.namespace and toksearch.llm.skills entry-point groups.
The packages= kwarg filters contributors to a named subset; None loads
all, an empty list loads none. extra_namespace= and extra_skill_dirs=
remain available for ad-hoc additions.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Wire discovered presets into `resolve_preset`

**Files:**
- Modify: `toksearch/llm/presets.py`
- Modify: `tests/test_llm_presets.py`

- [ ] **Step 1: Add tests for discovered-preset precedence**

Append to `tests/test_llm_presets.py`:

```python
class TestDiscoveredPresets(unittest.TestCase):
    def setUp(self):
        from toksearch.llm.discovery import clear_discovery_cache
        clear_discovery_cache()

    def tearDown(self):
        from toksearch.llm.discovery import clear_discovery_cache
        clear_discovery_cache()

    def test_discovered_preset_resolves(self):
        from unittest import mock
        preset = Preset(backend="anthropic",
                         base_url="https://site.example",
                         api_key_env="SITE_KEY")
        ep = mock.MagicMock()
        ep.name = "site"
        ep.load.return_value = preset
        with mock.patch(
            "toksearch.llm.discovery._entry_points",
            side_effect=lambda group: [ep] if group == "toksearch.llm.presets" else [],
        ):
            p = resolve_preset("site", Config())
        self.assertEqual(p.base_url, "https://site.example")

    def test_user_preset_overrides_discovered(self):
        from unittest import mock
        discovered = Preset(backend="anthropic",
                              base_url="https://discovered.example")
        ep = mock.MagicMock()
        ep.name = "site"
        ep.load.return_value = discovered
        with mock.patch(
            "toksearch.llm.discovery._entry_points",
            side_effect=lambda group: [ep] if group == "toksearch.llm.presets" else [],
        ):
            cfg = Config(user_presets={"site": {"base_url": "https://user.example"}})
            p = resolve_preset("site", cfg)
        self.assertEqual(p.base_url, "https://user.example")
        # Other fields fall through from discovered.
        self.assertEqual(p.backend, "anthropic")

    def test_discovered_does_not_override_builtin(self):
        """Discovered presets do NOT shadow built-ins of the same name."""
        from unittest import mock
        sneaky = Preset(backend="anthropic",
                         base_url="https://hijack.example")
        ep = mock.MagicMock()
        ep.name = "anthropic"
        ep.load.return_value = sneaky
        with mock.patch(
            "toksearch.llm.discovery._entry_points",
            side_effect=lambda group: [ep] if group == "toksearch.llm.presets" else [],
        ):
            p = resolve_preset("anthropic", Config())
        # Built-in wins.
        self.assertIsNone(p.base_url)
```

- [ ] **Step 2: Verify new tests fail**

- [ ] **Step 3: Modify `resolve_preset`**

Replace `resolve_preset` in `/fusion/projects/dt/sammuli/fdp_dev/repos/toksearch/toksearch/llm/presets.py` with:

```python
def resolve_preset(name: str, config: Config) -> Preset:
    """Resolve a preset name to a fully-populated ``Preset``.

    Lookup order (highest precedence first):
      1. User preset (``Config.user_presets[name]``) merged onto whatever
         base (built-in, discovered, or new) applies.
      2. Built-in preset (``BUILTIN_PRESETS[name]``).
      3. Discovered preset (``toksearch.llm.presets`` entry point).
      4. A user-only preset (no base; ``Preset(**user_preset)``).

    Built-ins shadow discovered presets of the same name -- this prevents
    a misconfigured package from silently hijacking a stable user-facing
    name like ``"anthropic"``.
    """
    from .discovery import discover_presets

    user = config.user_presets.get(name, {})
    if name in BUILTIN_PRESETS:
        base = BUILTIN_PRESETS[name]
        if user:
            return replace(base, **{k: v for k, v in user.items()
                                    if k in {"backend", "model", "base_url",
                                             "api_key_file", "api_key_env"}})
        return base
    discovered = discover_presets()
    if name in discovered:
        base = discovered[name]
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
        f"{sorted(BUILTIN_PRESETS)}. Discovered presets: {sorted(discovered)}. "
        f"Define a user preset in ~/.fdp/config.toml under "
        f"[llm.presets.{name}].")
```

- [ ] **Step 4: Run tests**

Expected: 9 (6 old + 3 new) preset tests pass.

- [ ] **Step 5: Commit**

```bash
git -C /fusion/projects/dt/sammuli/fdp_dev/repos/toksearch add toksearch/llm/presets.py tests/test_llm_presets.py
git -C /fusion/projects/dt/sammuli/fdp_dev/repos/toksearch commit -m "$(cat <<'EOF'
Consult discovered presets between user and built-in

resolve_preset now checks: user > built-in > discovered > user-only.
Built-ins still shadow discovered presets of the same name to prevent
a misconfigured package from hijacking a stable user-facing name.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Core toksearch self-registration via entry points

**Files:**
- Modify: `toksearch/__init__.py`
- Modify: `pyproject.toml`

This is the smallest task in terms of code, but it's the one that makes core toksearch a "first-class contributor" to its own discovery. Without it, a notebook user with no extra packages installed would have no `toksearch` binding in the namespace.

- [ ] **Step 1: Add `__llm_description__` to `toksearch/__init__.py`**

Open `/fusion/projects/dt/sammuli/fdp_dev/repos/toksearch/toksearch/__init__.py`. Find the line:

```python
__version__ = _version.get_versions()["version"]
```

Add immediately after it:

```python
__llm_description__ = (
    "core toksearch — Pipeline, MdsSignal, ZarrSignal, fetch_dataset, "
    "and SQL helpers"
)
```

- [ ] **Step 2: Expose a `CORE_SKILLS_DIR` constant**

Append to `/fusion/projects/dt/sammuli/fdp_dev/repos/toksearch/toksearch/llm/__init__.py`, after the existing imports:

```python
from pathlib import Path as _Path

CORE_SKILLS_DIR = _Path(__file__).parent.parent / "skills"
```

And add `"CORE_SKILLS_DIR"` to the `__all__` list.

- [ ] **Step 3: Declare core toksearch's contributor entry points in `pyproject.toml`**

Append to `/fusion/projects/dt/sammuli/fdp_dev/repos/toksearch/pyproject.toml`:

```toml

[project.entry-points."toksearch.llm.namespace"]
toksearch = "toksearch"

[project.entry-points."toksearch.llm.skills"]
toksearch = "toksearch.llm:CORE_SKILLS_DIR"
```

- [ ] **Step 4: Refresh editable install so entry points take effect**

```bash
cd /fusion/projects/dt/sammuli/fdp_dev/repos/toksearch && pixi run build 2>&1 | tail -3
```

- [ ] **Step 5: Verify the entry points are visible**

```bash
cd /fusion/projects/dt/sammuli/fdp_dev/repos/toksearch && pixi run python -c "
from importlib.metadata import entry_points
for ep in entry_points(group='toksearch.llm.namespace'):
    print('namespace:', ep.name, '->', ep.value)
for ep in entry_points(group='toksearch.llm.skills'):
    print('skills:', ep.name, '->', ep.value)
"
```

Expected output:
```
namespace: toksearch -> toksearch
skills: toksearch -> toksearch.llm:CORE_SKILLS_DIR
```

- [ ] **Step 6: Add a Session-level smoke test that core toksearch shows up in the namespace via discovery (not via `_default_namespace`)**

Append to `tests/test_llm_session.py`:

```python
class TestCoreSelfRegistration(unittest.TestCase):
    """Confirm core toksearch registers itself as a contributor.

    Without monkeypatching, discovery should find at least the core
    'toksearch' contributor (declared in pyproject.toml entry points).
    """

    def setUp(self):
        from toksearch.llm.discovery import clear_discovery_cache
        clear_discovery_cache()

    def tearDown(self):
        from toksearch.llm.discovery import clear_discovery_cache
        clear_discovery_cache()

    def test_core_toksearch_namespace_entry_loaded(self):
        from toksearch.llm.discovery import discover_namespace_contributors
        names = [n for n, _v, _d in discover_namespace_contributors()]
        self.assertIn("toksearch", names)

    def test_core_skills_dir_loaded(self):
        from toksearch.llm.discovery import discover_skill_dirs
        names = [n for n, _p in discover_skill_dirs()]
        self.assertIn("toksearch", names)

    def test_session_system_prompt_mentions_toksearch(self):
        backend = FakeBackend(scripted_turns=[_text("ok")])
        sess = Session(backend=backend)
        self.assertIn("toksearch", sess.system_prompt)
        self.assertIn("Pipeline", sess.system_prompt)  # from __llm_description__
```

- [ ] **Step 7: Run the full LLM suite**

```bash
cd /fusion/projects/dt/sammuli/fdp_dev/repos/toksearch && pixi run bash -c 'cd tests && python -m unittest discover -p "test_llm*" 2>&1 | tail -5'
```

Expected: 103 + new tests, all passing.

- [ ] **Step 8: Commit**

```bash
git -C /fusion/projects/dt/sammuli/fdp_dev/repos/toksearch add toksearch/__init__.py toksearch/llm/__init__.py pyproject.toml tests/test_llm_session.py
git -C /fusion/projects/dt/sammuli/fdp_dev/repos/toksearch commit -m "$(cat <<'EOF'
Register core toksearch as a self-contributor via entry points

toksearch.llm.namespace.toksearch -> the toksearch module (with
__llm_description__ for the system-prompt catalog).
toksearch.llm.skills.toksearch -> toksearch/skills/ via the
CORE_SKILLS_DIR re-export.

Without this, a Session in an env with no extras installed would
have no toksearch binding in the run_python namespace -- the
discovery path now treats core toksearch like any other contributor.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: CLI `--package` flag

**Files:**
- Modify: `toksearch/llm/cli.py`
- Modify: `tests/test_llm_cli.py`

- [ ] **Step 1: Add test for the `--package` flag**

Append to `tests/test_llm_cli.py` (before `if __name__`):

```python
class TestCliPackageFlag(unittest.TestCase):
    def _run_cli(self, argv):
        from toksearch.llm import cli
        fake_session = mock.MagicMock()
        fake_session.send.return_value = mock.MagicMock(
            stop_reason="end_turn", final_text="ok")
        with mock.patch.object(sys, "argv", argv), \
                mock.patch.object(cli, "build_session",
                                  return_value=fake_session) as build, \
                mock.patch.object(sys, "stdin", new=io.StringIO("")), \
                redirect_stdout(io.StringIO()):
            try:
                cli.main()
            except SystemExit:
                pass
        return build

    def test_package_flag_collected_into_list(self):
        build = self._run_cli(
            ["toksearch", "query", "--package", "toksearch",
             "--package", "toksearch_d3d", "hi"])
        ns = build.call_args.args[0]
        self.assertEqual(ns.packages, ["toksearch", "toksearch_d3d"])

    def test_no_package_flag_leaves_packages_none(self):
        build = self._run_cli(["toksearch", "query", "hi"])
        ns = build.call_args.args[0]
        # argparse default for action="append" is None when never given
        self.assertIsNone(ns.packages)
```

- [ ] **Step 2: Modify `_add_common` and `build_session` in `cli.py`**

In `/fusion/projects/dt/sammuli/fdp_dev/repos/toksearch/toksearch/llm/cli.py`, update `_add_common`:

```python
def _add_common(p):
    p.add_argument("--backend", default=None,
                   help="Backend / preset name (anthropic, openai, or a user "
                        "preset from ~/.fdp/config.toml).")
    p.add_argument("--model", default=None,
                   help="Override the preset's default model.")
    p.add_argument("-n", "--max-iterations", type=int, default=None,
                   help="Cap on tool-call rounds per turn.")
    p.add_argument("--package", dest="packages", action="append",
                   default=None,
                   help="Restrict discovered contributors to the named "
                        "package(s). Repeat the flag to allow multiple.")
```

Update `build_session` to forward `packages`:

```python
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
        packages=args.packages,
    )
```

- [ ] **Step 3: Verify all CLI tests pass**

Expected: 11 (8 old + 2 new + 1 existing TestResolveApiKey class with 3 tests) — count whatever the actual total is.

- [ ] **Step 4: Run the full LLM suite**

```bash
cd /fusion/projects/dt/sammuli/fdp_dev/repos/toksearch && pixi run bash -c 'cd tests && python -m unittest discover -p "test_llm*" 2>&1 | tail -3'
```

- [ ] **Step 5: Commit**

```bash
git -C /fusion/projects/dt/sammuli/fdp_dev/repos/toksearch add toksearch/llm/cli.py tests/test_llm_cli.py
git -C /fusion/projects/dt/sammuli/fdp_dev/repos/toksearch commit -m "$(cat <<'EOF'
Add --package flag to toksearch chat / toksearch query

Repeatable flag (argparse action="append") that maps to
Session(packages=[...]) to restrict discovered contributors.
None (no flag) means load all discovered contributors.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Self-Review Notes

Coverage against the PR 2 scope:
- ✓ `discovery.py` with three loader functions, caching, monkeypatch-friendly indirection — Task 1.
- ✓ Session uses discovery; `packages=` filter — Task 2.
- ✓ `resolve_preset` consults discovered presets with documented precedence — Task 3.
- ✓ Core toksearch registers itself via entry points + `__llm_description__` — Task 4.
- ✓ CLI `--package NAME` (repeatable) flag — Task 5.

Built-in presets (`anthropic`, `openai`) intentionally NOT moved to entry points; they stay in `BUILTIN_PRESETS` so that `pip install toksearch[llm]` always gives you those two without needing any extra contributor.

The `extra_skill_dirs=` and `extra_namespace=` kwargs on `Session.__init__` are preserved (additive to discovery). Removing them would be a follow-up cleanup.
