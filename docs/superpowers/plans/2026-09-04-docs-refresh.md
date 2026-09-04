# TokSearch Docs Refresh Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bring the TokSearch documentation back in line with the shipped software — fdp-core-first installation, a two-path "talk to your data" section, a new provenance/CMF page — and fix the one code bug the docs exposed (`fdp chat` never sets up the FDP environment).

**Architecture:** Three repositories, three independent commit streams. `fdp` gets a small `cli.py` change under TDD. `toksearch` gets documentation edits plus one new page and a nav entry. `toksearch_cmf` gets a README rewrite. The `fdp` fix must land first because the toksearch docs describe its post-fix behavior.

**Tech Stack:** Python 3.11, argparse, unittest + pytest, mkdocs-material + mkdocstrings, pixi.

---

## Environment notes (read before starting)

Three environment facts that will otherwise waste your time:

1. **Never run scripts from `repos/` root.** Namespace-package shadowing beats
   the editable install and `import toksearch_d3d` fails. Work from inside the
   repository you are editing.
2. **`pixi run` must be invoked with the shell's cwd inside that pixi project.**
   If a previous command left you elsewhere, `pixi run` fails with "could not
   find pixi.toml". Prefer `cd /abs/path/to/repo && pixi run ...` in a single
   command.
3. **The docs build needs `PYTHONNOUSERSITE=1`.** A stray pygments in
   `/home/sammuli/.local/lib/python3.10/site-packages` shadows the pixi env's
   and the build dies with `AttributeError: 'NoneType' object has no attribute
   'replace'`. With the variable set, the build succeeds.

Baseline before you start: `fdp`'s suite is **246 passed**.

---

## File Structure

| File | Responsibility | Task |
|---|---|---|
| `fdp/fdp/cli.py` | Add the `"best-effort"` `needs_env` state; mark chat/query | 1 |
| `fdp/tests/test_cli.py` | Coverage for the three env outcomes | 1 |
| `fdp/fdp/llm_shims.py` | Drop the vestigial `handle` parameter | 2 |
| `fdp/tests/test_llm_shims.py` | Update to the new `_build_llm_cmd` signature | 2 |
| `toksearch/README.md` | Landing page: talk-to-your-data + installation | 3, 4 |
| `toksearch/docs/index.md` | **Symlink to `../README.md`** — never edited directly | 3, 4 |
| `toksearch/docs/llm.md` | LLM reference: full CLI surface, agent integration | 5 |
| `toksearch/docs/LLM_Tutorial.ipynb` | Tutorial: install, backends, false claims | 6 |
| `toksearch/docs/provenance.md` | **New.** Provenance + CMF page with API blocks | 7 |
| `toksearch/mkdocs.yml` | Nav entry for the new page | 7 |
| `toksearch_cmf/README.md` | Describe the shipped 0.1.1 surface | 9 |

**`docs/index.md` is a symlink to `../README.md`** (mode `120000`, since
`03250be` in 2024). Edit `README.md` only; the site page follows automatically
and the two cannot drift. Because the same bytes are served on GitHub and on
the mkdocs site, cross-references in this shared text must be **absolute**
`https://ga-fdp.github.io/toksearch/latest/...` URLs, which resolve from both.
Do not use relative mkdocs paths here, and do not convert the symlink into a
regular file to work around that.

---

## Task 1: `fdp` — best-effort environment for `chat` and `query`

**Files:**
- Modify: `/fusion/projects/dt/sammuli/fdp_dev/repos/fdp/fdp/cli.py` (`build_parser`, `main`)
- Test: `/fusion/projects/dt/sammuli/fdp_dev/repos/fdp/tests/test_cli.py`

Today `needs_env` is a boolean. `True` means "set up the FDP environment or
exit 1"; `False` means "don't touch it". Commit `fe72baa` set `chat` and
`query` to `False` so they would run inside fdp's own dev env, where no device
contributor is installed and `setup_environment()` raises `ValueError`. The
side effect is that `fdp chat` reaches DIII-D data with no `PTDATA_LOC`, no
`default_tree_path`, and no `BEARER_TOKEN`.

Add a third state, `"best-effort"`: try, and warn-and-continue on the failures
that mean "no device is available here".

This is safe because `do_chat`/`do_query` `os.execvpe` into a fresh
`python -m toksearch.llm.cli` process. The environment is therefore in place
*before* the new process loads libfdpio or XRootD — the same shape as
`fdp run`, which is the known-good path. The failure mode to avoid is calling
`setup_environment()` and then using MDSplus *in the same process*; that is not
what happens here.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_cli.py`:

```python
class TestChatQueryEnvironment(unittest.TestCase):
    """chat/query set up the FDP environment when a device is available,
    and degrade to a warning when none is (fdp's own dev env)."""

    def test_chat_sets_up_environment(self):
        from fdp import cli
        with ExitStack() as stack:
            _patch_catalog(stack)
            stack.enter_context(mock.patch.object(
                sys, "argv", ["fdp", "chat"]))
            setup_mock = stack.enter_context(
                mock.patch.object(cli, "setup_environment"))
            ev = stack.enter_context(
                mock.patch.object(cli.llm_shims.os, "execvpe"))
            with redirect_stdout(io.StringIO()):
                cli.main()
        setup_mock.assert_called_once()
        self.assertEqual(setup_mock.call_args.kwargs.get("auto_login"), True)
        ev.assert_called_once()

    def test_query_sets_up_environment(self):
        from fdp import cli
        with ExitStack() as stack:
            _patch_catalog(stack)
            stack.enter_context(mock.patch.object(
                sys, "argv", ["fdp", "query", "hello"]))
            setup_mock = stack.enter_context(
                mock.patch.object(cli, "setup_environment"))
            ev = stack.enter_context(
                mock.patch.object(cli.llm_shims.os, "execvpe"))
            with redirect_stdout(io.StringIO()):
                cli.main()
        setup_mock.assert_called_once()
        ev.assert_called_once()

    def test_chat_survives_missing_device(self):
        """The fe72baa case: no contributor installed. Warn, then exec."""
        from fdp import cli
        buf = io.StringIO()
        with ExitStack() as stack:
            _patch_catalog(stack)
            stack.enter_context(mock.patch.object(
                sys, "argv", ["fdp", "chat"]))
            stack.enter_context(mock.patch.object(
                cli, "setup_environment",
                side_effect=ValueError("no device contributors installed")))
            ev = stack.enter_context(
                mock.patch.object(cli.llm_shims.os, "execvpe"))
            stack.enter_context(mock.patch.object(sys, "stderr", buf))
            with redirect_stdout(io.StringIO()):
                cli.main()
        ev.assert_called_once()
        self.assertIn("no device contributors installed", buf.getvalue())

    def test_chat_survives_auth_error(self):
        from fdp import cli, auth
        buf = io.StringIO()
        with ExitStack() as stack:
            _patch_catalog(stack)
            stack.enter_context(mock.patch.object(
                sys, "argv", ["fdp", "chat"]))
            stack.enter_context(mock.patch.object(
                cli, "setup_environment",
                side_effect=auth.AuthError("token acquisition failed")))
            ev = stack.enter_context(
                mock.patch.object(cli.llm_shims.os, "execvpe"))
            stack.enter_context(mock.patch.object(sys, "stderr", buf))
            with redirect_stdout(io.StringIO()):
                cli.main()
        ev.assert_called_once()
        self.assertIn("token acquisition failed", buf.getvalue())

    def test_strict_subcommand_still_exits_on_missing_device(self):
        """Regression guard: the new branch must not soften `fdp ls`."""
        from fdp import cli
        buf = io.StringIO()
        with ExitStack() as stack:
            _patch_catalog(stack)
            stack.enter_context(mock.patch.object(
                sys, "argv", ["fdp", "ls", "/"]))
            stack.enter_context(mock.patch.object(
                cli, "setup_environment",
                side_effect=ValueError("no device contributors installed")))
            stack.enter_context(mock.patch.object(sys, "stderr", buf))
            with redirect_stdout(io.StringIO()):
                with self.assertRaises(SystemExit) as cm:
                    cli.main()
        self.assertEqual(cm.exception.code, 1)
```

These reference `cli.llm_shims`. `fdp/cli.py` imports only the *functions*
today (`from .llm_shims import do_chat as _llm_do_chat`), so the module itself
is not bound and these tests will fail with `AttributeError` until Step 3 adds
the import. Patching `os.execvpe` needs a handle on the module where it is
looked up.

- [ ] **Step 2: Run the tests to verify they fail**

```bash
cd /fusion/projects/dt/sammuli/fdp_dev/repos/fdp && \
  PYTHONNOUSERSITE=1 pixi run pytest tests/test_cli.py::TestChatQueryEnvironment -v
```

Expected: `test_chat_sets_up_environment`, `test_query_sets_up_environment`
FAIL (`setup_mock.assert_called_once()` — "Expected 'setup_environment' to have
been called once. Called 0 times."). The other three should already pass; they
are the guards that the fix does not regress anything.

- [ ] **Step 3: Bind the `llm_shims` module**

`fdp/cli.py` imports only the functions out of `llm_shims`, so `cli.llm_shims`
does not resolve. Add the module import next to the existing
`from .llm_shims import ...` lines:

```python
from . import llm_shims
```

- [ ] **Step 4: Mark chat and query best-effort**

In `build_parser()`, replace:

```python
    # chat / query just execvpe into toksearch.llm.cli; no FDP env
    # setup needed, and they tolerate no device contributor being
    # installed (useful for working inside the fdp dev env).
    p_chat.set_defaults(func=do_chat, needs_env=False)
```

with:

```python
    # chat / query want the FDP environment -- the agent fetches shot data --
    # but must still run where no device contributor is installed (fdp's own
    # dev env). "best-effort" is that middle state: try, warn, continue.
    # Setting it up here is safe precisely because these subcommands execvpe
    # into a fresh process, so libfdpio and XRootD read the vars at load time.
    p_chat.set_defaults(func=do_chat, needs_env="best-effort",
                        auto_login=True)
```

and replace:

```python
    p_query.set_defaults(func=do_query, needs_env=False)
```

with:

```python
    p_query.set_defaults(func=do_query, needs_env="best-effort",
                         auto_login=True)
```

- [ ] **Step 5: Teach `main()` the third state**

Replace the environment block in `main()`:

```python
    # Pure-metadata subcommands (devices, skills, backends) don't touch
    # the FDP env and shouldn't require a device contributor to be
    # installed, so they opt out via `needs_env=False`. chat/query use
    # `needs_env="best-effort"`: they want the env when it is available
    # but must not die when it isn't.
    needs_env = getattr(args, "needs_env", True)
    if needs_env:
        best_effort = needs_env == "best-effort"
        # Device resolution can fail (e.g. no default chosen among several
        # registered tokamaks); present it as a clean message, not a traceback.
        try:
            setup_environment(
                device=args.device,
                bearer_token=args.bearer_token or None,
                auto_login=getattr(args, "auto_login", False),
            )
        except (ValueError, KeyError) as exc:
            if not best_effort:
                print(f"Error: {exc}", file=sys.stderr)
                sys.exit(1)
            print(f"Warning: continuing without the FDP environment ({exc}). "
                  f"Data access will not work in this session.",
                  file=sys.stderr)
        except auth.AuthError as exc:
            if not best_effort:
                print(f"Login failed: {exc}", file=sys.stderr)
                sys.exit(1)
            print(f"Warning: continuing without a bearer token ({exc}).",
                  file=sys.stderr)
```

- [ ] **Step 6: Run the new tests to verify they pass**

```bash
cd /fusion/projects/dt/sammuli/fdp_dev/repos/fdp && \
  PYTHONNOUSERSITE=1 pixi run pytest tests/test_cli.py::TestChatQueryEnvironment -v
```

Expected: 5 passed.

- [ ] **Step 7: Run the whole suite**

```bash
cd /fusion/projects/dt/sammuli/fdp_dev/repos/fdp && \
  PYTHONNOUSERSITE=1 pixi run pytest tests/ -q
```

Expected: 251 passed (246 baseline + 5 new). If anything else fails, it is a
regression from this change — fix it before committing.

- [ ] **Step 8: Commit**

```bash
cd /fusion/projects/dt/sammuli/fdp_dev/repos/fdp
git checkout -b fix/chat-query-environment
git add fdp/cli.py tests/test_cli.py
git commit -F - <<'MSG'
fix(cli): give fdp chat/query the FDP environment again

fe72baa marked chat/query needs_env=False so they would run inside fdp's
own dev env, where no device contributor is installed and
setup_environment raises. That fixed the crash but also meant the
preferred conversational entry point ran with no PTDATA_LOC, no
default_tree_path and no BEARER_TOKEN -- the agent could not reach any
shot data.

Add a third needs_env state, "best-effort": attempt setup, and warn and
continue on the failures that mean no device is available. Strict
subcommands are untouched and still exit 1.

Setting the environment up here is safe because chat/query execvpe into
a fresh python -m toksearch.llm.cli, so libfdpio and XRootD read the
variables at load time in the new process -- the same shape as fdp run.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_014c9v9tjK6XWLKxNrvXuzx4
MSG
```

---

## Task 2: `fdp` — drop the vestigial `handle` parameter

**Files:**
- Modify: `/fusion/projects/dt/sammuli/fdp_dev/repos/fdp/fdp/llm_shims.py`
- Modify: `/fusion/projects/dt/sammuli/fdp_dev/repos/fdp/fdp/cli.py` (`do_chat`, `do_query`, `_resolve_default_handle_or_none`)
- Test: `/fusion/projects/dt/sammuli/fdp_dev/repos/fdp/tests/test_llm_shims.py`

`_build_llm_cmd` takes a `handle` it documents as unused ("the parameter is
kept for call-site compatibility"), and `cli.py` maintains
`_resolve_default_handle_or_none` purely to produce it. The backend is a
deployment-level choice resolved inside `toksearch.llm`, so nothing will ever
consult it. Remove both.

- [ ] **Step 1: Update the tests first**

In `tests/test_llm_shims.py`, replace the two `TestBuildLlmCmd` methods:

```python
class TestBuildLlmCmd(unittest.TestCase):
    def test_basic_cmd_structure(self):
        from fdp.llm_shims import _build_llm_cmd
        cmd = _build_llm_cmd("query", ["hello"])
        self.assertEqual(cmd[:4],
                         [sys.executable, "-m", "toksearch.llm.cli",
                          "query"])
        self.assertIn("hello", cmd)

    def test_chat_passthrough(self):
        """No device is consulted: the backend is a deployment-level
        choice resolved inside toksearch.llm, not a per-device one."""
        from fdp.llm_shims import _build_llm_cmd
        cmd = _build_llm_cmd("chat", ["--gui"])
        self.assertEqual(cmd[:4],
                         [sys.executable, "-m", "toksearch.llm.cli",
                          "chat"])
        self.assertIn("--gui", cmd)
```

Delete the now-unused `_make_handle` helper and the `_FAKE_YAML` constant if
nothing else in the file references them. Check first:

```bash
cd /fusion/projects/dt/sammuli/fdp_dev/repos/fdp && \
  grep -n "_make_handle\|_FAKE_YAML" tests/test_llm_shims.py
```

- [ ] **Step 2: Run to verify failure**

```bash
cd /fusion/projects/dt/sammuli/fdp_dev/repos/fdp && \
  PYTHONNOUSERSITE=1 pixi run pytest tests/test_llm_shims.py -v
```

Expected: FAIL with `TypeError: _build_llm_cmd() missing 1 required positional
argument: 'handle'`.

- [ ] **Step 3: Remove the parameter from `llm_shims.py`**

```python
def _build_llm_cmd(subcommand: str, passthrough_args: list[str]) -> list[str]:
    """Construct argv for the `toksearch.llm.cli` delegate.

    The LLM backend/preset is a *deployment-level* choice, not a per-device
    one: toksearch.llm resolves it from ``--backend`` > ``$FDP_LLM_BACKEND``
    > ``~/.fdp/config.toml`` ``[llm].backend`` > the built-in default. No
    device is consulted here.
    """
    cmd = [sys.executable, "-m", "toksearch.llm.cli", subcommand]
    cmd.extend(passthrough_args)
    return cmd
```

Update both call sites in the same file:

```python
def do_query(args) -> None:
    passthrough = [args.query] + _common_passthrough(args)
    cmd = _build_llm_cmd("query", passthrough)
    os.execvpe(cmd[0], cmd, os.environ)


def do_chat(args) -> None:
    passthrough = _common_passthrough(args)
    cmd = _build_llm_cmd("chat", passthrough)
    env = os.environ
    if getattr(args, "gui", False):
        # Hand the GUI the FDP brand logo so it can stylize its
        # header. toksearch.llm.gui consults FDP_GUI_LOGO_PATH;
        # absence is fine and falls back to no logo. Only copy
        # os.environ when we actually need to mutate it.
        from . import main_logo_path
        logo = main_logo_path()
        if logo and "FDP_GUI_LOGO_PATH" not in os.environ:
            env = {**os.environ, "FDP_GUI_LOGO_PATH": logo}
    os.execvpe(cmd[0], cmd, env)
```

`TokamakHandle` appears in this file only on the three signatures you just
changed (lines 31, 37, 70, 76 before the edit), so delete its `TYPE_CHECKING`
block too:

```python
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .catalog import TokamakHandle
```

Confirm nothing else needs it:

```bash
cd /fusion/projects/dt/sammuli/fdp_dev/repos/fdp && \
  grep -n "TokamakHandle\|TYPE_CHECKING" fdp/llm_shims.py
```

Expected: no output.

- [ ] **Step 4: Remove the resolver from `cli.py`**

Delete `_resolve_default_handle_or_none` entirely and simplify its callers:

```python
def do_chat(args) -> None:
    _llm_do_chat(args)


def do_query(args) -> None:
    _llm_do_query(args)
```

- [ ] **Step 5: Run the full suite**

```bash
cd /fusion/projects/dt/sammuli/fdp_dev/repos/fdp && \
  PYTHONNOUSERSITE=1 pixi run pytest tests/ -q
```

Expected: 251 passed. (Task 1's tests patch `cli.llm_shims.os.execvpe`, which
is unaffected by the signature change.)

- [ ] **Step 6: Commit**

```bash
cd /fusion/projects/dt/sammuli/fdp_dev/repos/fdp
git add fdp/llm_shims.py fdp/cli.py tests/test_llm_shims.py
git commit -F - <<'MSG'
refactor(llm): drop the handle nothing consults

_build_llm_cmd documented that it ignores the handle and kept the
parameter "for call-site compatibility"; cli.py maintained
_resolve_default_handle_or_none solely to produce one. The backend is
resolved inside toksearch.llm from flags, env and config, so no device
will ever be consulted here. Remove both.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_014c9v9tjK6XWLKxNrvXuzx4
MSG
```

---

## Task 3: `toksearch` — rewrite "Talk to your data"

**Files:**
- Modify: `/fusion/projects/dt/sammuli/fdp_dev/repos/toksearch/docs/index.md`
- Modify: `/fusion/projects/dt/sammuli/fdp_dev/repos/toksearch/README.md`

Work on branch `docs/refresh-2026-09`, which already exists and holds the spec
commit.

The two files carry an identical "## Talk to your data" section today. Replace
it in both with the text below, verbatim and identically.

- [ ] **Step 1: Replace the section in `docs/index.md`**

Everything from the line `## Talk to your data` up to (but not including)
`## Installation` becomes:

````markdown
## Talk to your data

TokSearch packages its own know-how — how to build a pipeline, which signal
class a quantity needs, how to align and aggregate — as agent-readable skills.
There are two ways to use them, and both are first-class.

### The built-in conversational CLI

`toksearch chat` is a REPL with an LLM behind it. It writes the pipeline code,
runs it against a persistent Python namespace — so follow-up turns iterate on
cached results instead of re-fetching — and shows you each block before
executing it.

```bash
toksearch chat                  # interactive REPL
toksearch chat --gui            # local Gradio GUI in a browser tab
toksearch query "..."           # one-shot, for scripts and quick lookups
toksearch backends              # list the names --backend accepts here
```

```text
you> Use run_python to fetch ipmhd for shot 165920 from efit01.
[run_python] Fetch ipmhd via MdsSignal.
  pipeline = toksearch.Pipeline([165920])
  pipeline.fetch('ip', toksearch.MdsSignal(r'\ipmhd', 'efit01'))
  rec = list(pipeline.compute_serial())[0]
[output] (no output)

you> What's the peak |Ip| in MA?
[run_python] ...
[output] 1.1325
```

Backends: the Anthropic API, the OpenAI API, your Claude Max plan via the
Claude Agent SDK, or the American Science Cloud (AmSC) endpoint contributed by
`toksearch_d3d`.

### Your own agent

If you already work in Claude Code, Cursor, or Codex, point it at the same
material instead of switching tools. The skills install directly, and the same
set is served over MCP by a standalone server:

```bash
fdp skills list                  # what's available, and what's installed
fdp skills install               # → ~/.claude/skills
fdp skills install --backend cursor    # or codex, or all; -f to overwrite

claude mcp add toksearch-skills -- fdp run python -m toksearch.llm.mcp
```

The server exposes every skill as a `skill://<name>` MCP resource plus a
`read_skill` tool, so any MCP-capable client can browse and read them.

### Preferred: run either path through `fdp`

Both paths are best used from an environment installed with `fdp-core` (see
[Installation](#installation)), through the `fdp` wrappers — `fdp chat`,
`fdp query`, `fdp skills`. Those come with the device packages and configure
data access (XRootD transport, MDSplus tree paths, bearer token) before the
session starts, so the agent can actually reach shot data.

```bash
fdp chat                        # interactive, FDP environment configured
fdp query "Fetch ip for shot 200000 and report the peak in MA."
```

See the [LLM tutorial](LLM_Tutorial.ipynb) for an end-to-end walkthrough and
the [LLM Interface reference](llm.md) for the full API surface.
````

- [ ] **Step 2: Apply the same replacement to `README.md`**

Identical text, except the two trailing cross-reference links, which must be
absolute in the README (it is read on GitHub, where relative mkdocs paths do
not resolve):

```markdown
See the [LLM tutorial](https://ga-fdp.github.io/toksearch/latest/LLM_Tutorial/)
for an end-to-end walkthrough and the
[LLM Interface reference](https://ga-fdp.github.io/toksearch/latest/llm/) for
the full API surface.
```

- [ ] **Step 3: Verify the two files stayed in sync**

```bash
cd /fusion/projects/dt/sammuli/fdp_dev/repos/toksearch && \
  diff <(sed -n '/^## Talk to your data/,/^## Installation/p' README.md) \
       <(sed -n '/^## Talk to your data/,/^## Installation/p' docs/index.md)
```

Expected: differences only in the final two link lines.

- [ ] **Step 4: Commit**

```bash
cd /fusion/projects/dt/sammuli/fdp_dev/repos/toksearch
git add README.md docs/index.md
git commit -F - <<'MSG'
docs: give "talk to your data" both of its paths

The section knew only about `toksearch chat`. Since it was written the
skills MCP server shipped (toksearch >= 2.8.2) and `fdp skills install`
landed, so a user who already runs Claude Code, Cursor or Codex has a
first-class path the docs never mentioned. Present both, and say plainly
that either is best run through the `fdp` wrappers from an fdp-core
environment, which is what makes shot data reachable.

Also adds the surface that was missing: `--gui`, `toksearch backends`.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_014c9v9tjK6XWLKxNrvXuzx4
MSG
```

---

## Task 4: `toksearch` — rewrite Installation

**Files:**
- Modify: `/fusion/projects/dt/sammuli/fdp_dev/repos/toksearch/README.md`
  (`docs/index.md` is a symlink to it and needs no separate edit)

Facts this section must respect, all verified:

- `fdp-install` renders a `pixi.toml` whose only real dependency is
  `fdp-core = "*"` (or `fdp-core-latest` with `--latest`), then runs
  `pixi install`. Flags: `-d/--directory`, `--latest`, `--install-skills`,
  `--with-labeler`.
- `fdp-core` 1.4.0 blesses toksearch 2.11.1, toksearch_d3d 0.12.0,
  toksearch_mast 0.1.0, ptdata 2.3.1, imas_composer 0.2.4, fdp 0.6.0,
  cmflib 0.1.0, and toksearch_cmf 0.1.1.
- `environment.yml` **does not exist** in this repository. The current source
  instructions are unrunnable.
- TokSearch is **not** on PyPI: `https://pypi.org/pypi/toksearch/json` → 404.

- [ ] **Step 1: Replace everything from `## Installation` to end of file**

In `docs/index.md`:

````markdown
## Installation

### Recommended: the full FDP environment

Nearly everyone should install TokSearch as part of the Fusion Data Platform,
through the `fdp-core` metapackage. That gets you TokSearch together with the
device packages (`toksearch_d3d`, `toksearch_mast`), `ptdata`, `imas_composer`,
the `fdp` CLI, the XRootD/Pelican transport that reaches DIII-D data, and the
`toksearch_cmf` provenance backend — all pinned to a tested, mutually
compatible set.

```bash
conda create -n fdp-installer -c ga-fdp -c conda-forge fdp-installer
conda activate fdp-installer
fdp-install -d /path/to/project
cd /path/to/project
pixi shell
```

`fdp-install` writes a `pixi.toml` that depends on `fdp-core` and runs
`pixi install`. Substitute `mamba` or `micromamba` for `conda` if you prefer;
the `-n` name is arbitrary.

| Flag | Effect |
|---|---|
| `-d`, `--directory` | Where to install (default: the current directory) |
| `--latest` | Depend on `fdp-core-latest` (`>=` floors) instead of `fdp-core` (exact `==` pins) |
| `--install-skills` | Also install the agent skills to `~/.claude/skills/` |

Inside that environment, run your analysis through the `fdp` CLI so the
data-access environment is configured:

```bash
fdp run python my_pipeline.py
fdp run jupyter lab
```

### TokSearch on its own

If you want the framework without FDP data access — your own MDSplus server,
your own Zarr stores, or just the `Pipeline` machinery — install the package by
itself:

```bash
conda install -c ga-fdp -c conda-forge toksearch
```

or into a fresh environment:

```bash
mamba create -n toksearch -c ga-fdp -c conda-forge toksearch
```

This does **not** give you DIII-D access: no `toksearch_d3d`, no `fdp` CLI, no
XRootD/Pelican transport.

TokSearch is not published on PyPI. We intend to provide a pip install in the
future.

### From source

The repository is managed with [pixi](https://pixi.sh):

```bash
git clone https://github.com/GA-FDP/toksearch.git
cd toksearch
pixi install
pixi run build          # editable install into the pixi environment
pixi run test           # run the test suite
```

To build the documentation site locally:

```bash
pixi run -e docs docs-serve
```
````

- [ ] **Step 2: Confirm `docs/index.md` still tracks it**

```bash
cd /fusion/projects/dt/sammuli/fdp_dev/repos/toksearch && \
  git ls-files -s docs/index.md && tail -3 docs/index.md
```

Expected: mode `120000` (still a symlink), and the tail showing the new
content. If the mode is `100644`, the symlink was broken — restore it with
`rm docs/index.md && ln -s ../README.md docs/index.md` rather than maintaining
two copies.

- [ ] **Step 3: Confirm the dead instructions are gone**

```bash
cd /fusion/projects/dt/sammuli/fdp_dev/repos/toksearch && \
  grep -n "environment.yml" README.md docs/index.md
```

Expected: no output.

- [ ] **Step 4: Commit**

```bash
cd /fusion/projects/dt/sammuli/fdp_dev/repos/toksearch
git add README.md
git commit -F - <<'MSG'
docs: install the way people actually install

Installation led with `conda install -c ga-fdp toksearch`, which gets a
user the framework and none of the data access: no toksearch_d3d, no fdp
CLI, no XRootD transport. The real path for nearly everyone is
fdp-install, which drops a pixi.toml depending on the fdp-core
metapackage. Lead with that, keep the standalone install as the second
option, and say plainly what it does not include.

The "from source" section was worse than stale -- it instructed
`mamba env create -f environment.yml` and no such file exists. Rewrite it
for pixi, which is what the repository actually uses.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_014c9v9tjK6XWLKxNrvXuzx4
MSG
```

---

## Task 5: `toksearch` — correct and extend `docs/llm.md`

**Files:**
- Modify: `/fusion/projects/dt/sammuli/fdp_dev/repos/toksearch/docs/llm.md`

Verified CLI surface (from `--help` in the installed environment):

```
toksearch {query,chat,backends}
toksearch chat [--backend B] [--model M] [-n N] [--package P] [-v] [--gui] [--no-browser]
fdp   chat [--backend B] [--model M] [-n N] [--gui] [--no-browser]
fdp   skills {list,install} [--backend claude|cursor|codex|all] [--force]
```

`fdp chat`/`fdp query` forward strictly fewer flags than `toksearch chat`: no
`--package`, no `-v/--verbose`. `fdp/llm_shims.py::_common_passthrough` is the
authority.

- [ ] **Step 1: Replace the "### CLI" subsection under "## Quickstart"**

````markdown
### CLI

```bash
# One-shot
toksearch query --backend anthropic "Use run_python to compute 2 + 2."

# Interactive REPL
toksearch chat --backend anthropic

# Local Gradio GUI in a browser tab (--no-browser to skip opening it)
toksearch chat --gui

# What can --backend be, in this environment?
toksearch backends
```

`toksearch backends` prints the resolved registry — built-in backends,
backends discovered from installed packages, and your own presets:

```text
name        source      backend     model
----------  ----------  ----------  -----------------
amsc        discovered  anthropic   claude-sonnet-4-6
anthropic   built-in    anthropic   claude-sonnet-4-6  (default)
claude-max  built-in    claude-max  -
openai      built-in    openai      gpt-4o
```

Flags shared by `query` and `chat`:

| Flag | Effect |
|---|---|
| `--backend NAME` | Backend or preset name. |
| `--model NAME` | Override the preset's default model. |
| `-n`, `--max-iterations N` | Cap on tool-call rounds per turn. |
| `--package NAME` | Restrict discovered contributors to the named package(s). Repeatable. |
| `-v`, `--verbose` | Show full tool-call code and tool-result bodies instead of a one-line summary per call. |

The REPL accepts `/help`, `/reset`, and `/quit`; ctrl-D also exits.

From a DIII-D environment, the `fdp` CLI wraps the same commands with the FDP
environment configured (XRootD plugin, MDSplus tree paths, `BEARER_TOKEN`)
before the session starts, which is what lets the agent reach shot data:

```bash
fdp query "Fetch ip for shot 200000 and report peak in MA."
fdp chat                                                       # interactive
fdp chat --gui
```

`fdp chat`/`fdp query` forward `--backend`, `--model`, `-n/--max-iterations`,
`--gui`, and `--no-browser`. They do not forward `--package` or `-v`; for those,
run the underlying command inside the FDP environment instead:

```bash
fdp run toksearch chat --package toksearch_d3d -v
```

The backend default is deployment-level, not device-driven: `--backend` →
`$FDP_LLM_BACKEND` → `~/.fdp/config.toml [llm].backend` → built-in
`anthropic`. GA on-prem users who want AmSC by default should set
`backend = "amsc"` in `config.toml` (or `export FDP_LLM_BACKEND=amsc`).
````

- [ ] **Step 2: Add a "Using your own agent" section**

Insert immediately after the "## Quickstart" section's Python subsection and
before "## Configuration":

````markdown
## Using your own agent

The skills that `lookup_docs` serves to the built-in agent are the same ones an
external coding agent can read. Two delivery mechanisms:

**Installed skill files.** `fdp skills` copies the SKILL.md directories
contributed by every installed package into the agent's skills directory:

```bash
fdp skills list                            # available + install status
fdp skills install                         # → ~/.claude/skills (Claude Code)
fdp skills install --backend cursor        # or codex, or all
fdp skills install --force                 # overwrite already-installed copies
```

**The MCP server.** `python -m toksearch.llm.mcp` is a standalone stdio MCP
server that exposes each skill as a `skill://<name>` resource plus a
`read_skill` tool. Register it with Claude Code:

```bash
claude mcp add toksearch-skills -- fdp run python -m toksearch.llm.mcp
```

Wrapping it in `fdp run` means the agent's environment can also fetch data, not
just read documentation. Any MCP-capable client can connect to the same server.

Discovery for both mechanisms is the `toksearch.llm.skills` entry-point group;
extra directories can be added through the `TOKSEARCH_SKILL_DIRS` environment
variable (os.pathsep-delimited). `Session` launches its own copy of this server
as a subprocess, so the built-in agent and an external one read byte-identical
documentation.
````

- [ ] **Step 3: Shrink the duplicate MCP paragraph under "## Tools"**

Under `### lookup_docs`, replace the paragraph beginning "Skills are served via
a **standalone MCP server**…" with:

```markdown
Skills are served over MCP: `Session` launches `python -m toksearch.llm.mcp` as
a subprocess on construction. The same server can be used directly by an
external agent — see [Using your own agent](#using-your-own-agent).
```

- [ ] **Step 4: Fix the installation block**

Replace the "## Installation" section with:

````markdown
## Installation

The LLM interface ships with TokSearch; there is nothing extra to install if
you installed through `fdp-core` (see the
[installation guide](index.md#installation)). The conda recipe lists the four
backend SDKs (`anthropic`, `openai`, `claude-agent-sdk`, `mcp`) and
`matplotlib` as hard run-dependencies.

Installing TokSearch on its own works too:

```bash
conda install -c ga-fdp -c conda-forge toksearch
```

From a source checkout, the same surface comes from the `llm` extra:

```bash
pip install -e '.[llm]'
```
````

- [ ] **Step 5: Fix the "See also" list**

Replace the `fdp` CLI bullet with:

```markdown
- [`fdp` CLI](https://github.com/GA-FDP/fdp) — wraps `toksearch chat`/`query`
  with FDP environment setup, and provides `fdp skills`.
```

- [ ] **Step 6: Check for surviving false claims**

```bash
cd /fusion/projects/dt/sammuli/fdp_dev/repos/toksearch && \
  grep -n "defaults to --backend amsc\|pip install toksearch\[llm\]" docs/llm.md
```

Expected: no output.

- [ ] **Step 7: Commit**

```bash
cd /fusion/projects/dt/sammuli/fdp_dev/repos/toksearch
git add docs/llm.md
git commit -F - <<'MSG'
docs(llm): document the whole CLI, and how to use your own agent

The reference covered `query` and `chat` and nothing else: no
`toksearch backends`, no `--gui`, no `-v/--verbose`, no `--package`. Add
them, with the real `backends` output.

Add a "Using your own agent" section for `fdp skills` and the standalone
MCP server, which previously appeared only as an implementation detail of
Session, and record the flag asymmetry -- `fdp chat` forwards neither
--package nor -v, so those need `fdp run toksearch chat`.

`pip install toksearch[llm]` implied TokSearch is on PyPI. It is not;
that form only works from a source checkout.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_014c9v9tjK6XWLKxNrvXuzx4
MSG
```

---

## Task 6: `toksearch` — update `docs/LLM_Tutorial.ipynb`

**Files:**
- Modify: `/fusion/projects/dt/sammuli/fdp_dev/repos/toksearch/docs/LLM_Tutorial.ipynb`

This is a JSON notebook. Edit it with a Python script rather than by hand so
the JSON stays valid; `mkdocs-jupyter` is configured with `execute: false`, so
nothing here runs at build time, but a malformed notebook breaks the build.

Four markdown cells need changes. Cell indices below are from the current file;
re-derive them by matching on content, not by trusting the numbers.

- [ ] **Step 1: Fix the install block (cell 1, "## 1. Setup")**

Replace the fenced block under `### Install`:

````markdown
### Install

The LLM interface ships with TokSearch. If you installed the Fusion Data
Platform the usual way, you already have it:

```bash
conda create -n fdp-installer -c ga-fdp -c conda-forge fdp-installer
conda activate fdp-installer
fdp-install -d /path/to/project
cd /path/to/project && pixi shell
```

That environment carries `toksearch`, `toksearch_d3d`, the `fdp` CLI, and the
transport needed to reach DIII-D data. TokSearch on its own
(`conda install -c ga-fdp -c conda-forge toksearch`) gives you the agent but no
device access.
````

Leave the `### Credentials` and `### Verify` blocks as they are; both are
accurate.

- [ ] **Step 2: Fix the false claims in cell 10 ("## 5. A realistic DIII-D workflow")**

Replace the paragraph and code fence that read:

```
From the shell, `fdp` wraps `toksearch chat` with the FDP environment
pre-configured (XRootD plugin, MDSplus tree paths, BEARER_TOKEN):

```bash
fdp chat                   # defaults to --backend amsc
```
```

with:

````markdown
From the shell, `fdp chat` wraps `toksearch chat` and configures the FDP
environment (XRootD plugin, MDSplus tree paths, `BEARER_TOKEN`) before the
session starts, which is what lets the agent reach shot data:

```bash
fdp chat                              # backend from config; --backend to override
fdp chat --backend amsc               # GA on-prem AmSC endpoint
```

The backend is a deployment-level choice, not a per-device one: `--backend` →
`$FDP_LLM_BACKEND` → `~/.fdp/config.toml [llm].backend` → built-in `anthropic`.
Set `backend = "amsc"` in `config.toml` to make AmSC your default.
````

- [ ] **Step 3: Add the GUI and external-agent material to cell 2 ("## 2. Quickstart for new users")**

Append to the end of that cell:

````markdown
### The GUI

The same session is available as a local Gradio app, which renders matplotlib
figures inline instead of writing them to disk:

```bash
toksearch chat --gui             # opens a browser tab; --no-browser to skip
fdp chat --gui                   # same, with the FDP environment configured
```

### Or use the agent you already have

If you work in Claude Code, Cursor, or Codex, install the skills into it rather
than switching tools:

```bash
fdp skills install                                  # → ~/.claude/skills
claude mcp add toksearch-skills -- fdp run python -m toksearch.llm.mcp
```

Both deliver the same skill content the built-in agent reads through
`lookup_docs`.
````

- [ ] **Step 4: Fix the "Where to next" links in the last cell**

Replace the `fdp` CLI bullet:

```markdown
- `fdp` CLI source: [GA-FDP/fdp](https://github.com/GA-FDP/fdp) — the FDP
  environment plus `fdp chat`, `fdp query`, `fdp run`, and `fdp skills`.
  DIII-D signal classes and the `amsc` preset live in
  [GA-FDP/toksearch_d3d](https://github.com/GA-FDP/toksearch_d3d).
```

Add one bullet:

```markdown
- [Provenance and CMF](provenance.md) — recording what a pipeline run consumed
  and produced, and pushing it to CMF.
```

- [ ] **Step 5: Verify the notebook is still valid JSON**

```bash
cd /fusion/projects/dt/sammuli/fdp_dev/repos/toksearch && \
  python3 -c "import json; nb=json.load(open('docs/LLM_Tutorial.ipynb')); print(len(nb['cells']), 'cells OK')"
```

Expected: `18 cells OK` (unchanged count — all four edits are in-place).

- [ ] **Step 6: Confirm the false claims are gone**

```bash
cd /fusion/projects/dt/sammuli/fdp_dev/repos/toksearch && \
  grep -c "defaults to --backend amsc" docs/LLM_Tutorial.ipynb
```

Expected: `0`.

- [ ] **Step 7: Commit**

```bash
cd /fusion/projects/dt/sammuli/fdp_dev/repos/toksearch
git add docs/LLM_Tutorial.ipynb
git commit -F - <<'MSG'
docs(llm): correct the tutorial's two false claims about fdp chat

The tutorial said `fdp chat` defaults to `--backend amsc` and that it
comes with the FDP environment pre-configured. Neither was true: fdp
commit fe72baa stopped injecting a device-derived backend and marked
chat/query needs_env=False. The environment half is now true again (fdp
fix/chat-query-environment); the amsc default never comes back, because
the backend is deployment-level and resolves from flag, env, then config.

Also leads with the fdp-core install, and adds the GUI and the
external-agent path.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_014c9v9tjK6XWLKxNrvXuzx4
MSG
```

---

## Task 7: `toksearch` — new `docs/provenance.md` and nav entry

**Files:**
- Create: `/fusion/projects/dt/sammuli/fdp_dev/repos/toksearch/docs/provenance.md`
- Modify: `/fusion/projects/dt/sammuli/fdp_dev/repos/toksearch/mkdocs.yml`

Verified signatures — use these exactly:

```
toksearch.provenance.__all__ = ['canonical_json', 'sha256_of', 'callable_spec',
    'RunContext', 'SourceSpec', 'OpSpec', 'BackendSpec', 'CodeSpec',
    'capture_code', 'Provenance', 'safe_call', 'JsonProvenance']

JsonProvenance(pipeline_name, stage=None, path='toksearch_run.json', strict=False)
Pipeline.write(directory, field=None, fields=None, fmt=None, name=None,
               track='directory', exist_ok=False, path_field='output_path',
               on_error='skip')
Pipeline.compute_serial(provenance=None)
Pipeline.compute_multiprocessing(num_workers=None, batch_size='auto', provenance=None)
RecordSet.to_dataframe(fields=None)
RecordSet.to_parquet(path, fields=None)
CmfRun(pipeline_name, stage=..., work_dir=...)
```

The JSON sample below is **real output**, captured by running the page's own
example in a git repository. Do not paraphrase it.

- [ ] **Step 1: Write `docs/provenance.md`**

````markdown
# Provenance and CMF

A computed result is only reusable if you can say what produced it. TokSearch
derives that description itself — the shot source, the specification of every
signal fetched, the sequence of operations, the compute backend, and the git
commit of the script that ran — and hands it to a provenance backend. You do
not hand-write metadata.

## The interface

`toksearch.provenance` defines the contract and nothing else. It has no
dependency on any metadata service; the CMF implementation lives in the
separate `toksearch_cmf` package.

**`RunContext`** is the derived description of a run. Its parts:

| Component | What it carries |
|---|---|
| `SourceSpec` | Where the shots came from: `kind` (`shotlist`, `sql`, `recordset`), `count`, a hash of the shot list, and for SQL sources the `query` and `params` |
| `OpSpec` | One per pipeline operation, in order: `fetch`, `map`, `keep`, `where`, `align`, `write`, each with its detail |
| `BackendSpec` | Which compute backend ran it, and its configuration |
| `CodeSpec` | The executing script, its `argv`, the git `commit`, and whether the tree was `dirty` |
| `signals` | The `Signal.spec()` of every fetched signal, keyed by record field |

`RunContext.input_identity()` hashes only the source, the signals, and the
device — deliberately excluding operations, backend, and code. Two runs that
read the same data have the same input identity even if they process it
differently.

**`Provenance`** is the ABC a backend implements:

| Hook | When |
|---|---|
| `on_compute_start(ctx)` | Before the backend runs |
| `on_compute_end(ctx, recordset)` | After compute returns |
| `output(*paths, **custom_properties)` | Declare artifacts toksearch did not write itself |
| `metrics(name, values)` | Record a named set of metrics |
| `finalize()` | Flush and close the record |

Hooks are invoked through `safe_call`, which converts any exception into a
`RuntimeWarning` and carries on. Losing a provenance record is bad; losing a
completed multi-hour compute is worse. Set `strict=True` on the backend to make
failures propagate instead — appropriate in CI, not in production runs.

## Recording a run

Pass a backend to any of the four `compute_*` methods. `JsonProvenance` needs
no external services, which makes it the right thing to try first:

```python
import xarray as xr
from toksearch import Pipeline
from toksearch.provenance import JsonProvenance

def peaks(rec):
    rec["peaks"] = xr.Dataset(
        {"double": ("shot", [rec.shot * 2])},
        coords={"shot": ("shot", [rec.shot])},
    )

pipeline = Pipeline([180000, 180001])
pipeline.map(peaks)
pipeline.keep(["peaks"])
pipeline.write("out", field="peaks", fmt="netcdf")

run = JsonProvenance("demo", stage="explore", path="run.json")
results = pipeline.compute_serial(provenance=run)
run.metrics("coverage", {"requested": 2, "returned": len(results)})
run.finalize()
```

That writes `out/180000.nc` and `out/180001.nc`, and this `run.json`:

```json
{
  "context": {
    "backend": {"config": {}, "kind": "SerialRecordSet"},
    "code": {
      "argv": ["demo.py"],
      "commit": "b06208582106642adfc5666eef5c3c43628204d8",
      "dirty": false,
      "repo_root": "/path/to/demo",
      "script": "demo.py"
    },
    "device": null,
    "ops": [
      {"op": "map",
       "detail": {"func": {"module": "__main__", "name": "peaks",
                           "source_sha256": "30b447cd1662a427..."}}},
      {"op": "keep", "detail": {"fields": ["peaks"]}},
      {"op": "write",
       "detail": {"directory": "/path/to/demo/out", "fields": ["peaks"],
                  "fmt": "netcdf", "func": null, "name": null,
                  "on_error": "skip", "path_field": "output_path",
                  "track": "directory"}}
    ],
    "parent_run": null,
    "signals": {},
    "source": {"count": 2, "hash": "8964cb18ba55ec4e...", "kind": "shotlist",
               "params": null, "query": null}
  },
  "input_identity": "e320ff068b07e8d0aa91bdcd7c7b8c2c454cf1dee2879fab1ef8718668e15d36",
  "metrics": {"coverage": {"requested": 2, "returned": 2}},
  "outputs": [{"path": "/path/to/demo/out", "source": "pipeline_write"}],
  "pipeline_name": "demo",
  "run_id": "bd9348f4471b4f2fb14ca64aea5fcdf6",
  "stage": "explore"
}
```

`source_sha256` is the hash of the mapped function's source, so a change to
`peaks` produces a different record. `commit` and `dirty` come from the git
repository containing the script; outside a repository both are `null`.

Add a `fetch` and the `signals` block fills in with that signal's full
specification — class, module, expression, tree name, dims, and every
constructor field — which is what makes the input identity meaningful.

### Chaining runs

`Pipeline.compute` copies the backend's `run_id` onto the returned
`RecordSet`. Building a new pipeline from that recordset —
`Pipeline(previous_results)` — reads it back as `RunContext.parent_run`, so a
multi-stage analysis links itself without any bookkeeping on your part.

## Getting data out

### `Pipeline.write`

Write one file per record, in the worker that produced it. This is the
recommended way to get data out of a pipeline: writing per shot in the workers
is faster than concatenating on the driver, and more honest — concatenation is
a transformation and deserves its own stage rather than hiding inside a writer.

Two forms. **Declarative**, which appends the operation immediately:

```python
pipeline.write("out/peaks", field="ds", fmt="netcdf")
pipeline.write("out/peaks", fields=["ip", "betan"], fmt="netcdf")   # xr.merge
```

**Decorator**, when the file's content needs computing:

```python
@pipeline.write("out/peaks", fmt="netcdf")
def shot_file(rec):
    return rec["ds"]
```

A two-argument function takes `(record, path)`, writes the file itself, and
returns the path it wrote.

| Argument | Meaning |
|---|---|
| `directory` | Output directory; created if absent |
| `field` / `fields` | One record field, or several merged with `xarray.merge` |
| `fmt` | `netcdf`, `parquet`, `npy`, `npz`, `json`; inferred from the object when omitted |
| `name` | `(record) -> str` basename without extension; defaults to the shot number |
| `track` | `directory` (one artifact for the whole directory) or `file` (one per shot) |
| `exist_ok` | Off by default — two runs interleaving into one directory silently corrupt the directory's content hash, and `flock` is not cross-client on BeeGFS, so nothing else prevents it |
| `path_field` | Record field that receives the written path |
| `on_error` | `skip` (default) writes no file for a record that already failed, so the directory — and the provenance hash over it — covers exactly the shots that completed; `write` writes anyway |

Read the results back with `xarray.open_mfdataset('out/peaks/*.nc')` where
`dask` is installed. Without it, open per file:

```python
import glob
import xarray as xr

ds = xr.concat(
    [xr.open_dataset(f) for f in sorted(glob.glob("out/peaks/*.nc"))],
    dim="shot",
)
```

Without `netCDF4`/`h5netcdf`, xarray writes NetCDF3 via scipy, which has no
int64 — integer coordinates read back as int32.

### Driver-side collection

When the per-shot results are small enough to gather, `RecordSet` collapses
them directly:

```python
results = pipeline.compute_multiprocessing(num_workers=16)
df = results.to_dataframe()                       # all scalar fields
df = results.to_dataframe(fields=["ip_max_ma"])   # a subset
results.to_parquet("peaks.parquet")
```

## Recording to CMF

[CMF](https://github.com/HewlettPackard/cmf) (Common Metadata Framework) is a
git- and DVC-backed metadata store. `toksearch_cmf` provides `CmfRun`, a
`Provenance` implementation that records a run there. It owns every `cmflib`
and `dvc` dependency; TokSearch core knows about neither.

`toksearch_cmf` is part of the `fdp-core` metapackage, so an FDP environment
already has it.

```python
from toksearch import MdsSignal, Pipeline
from toksearch_d3d import PtDataSignal
from toksearch_cmf import CmfRun

run = CmfRun("betan-ip-study", stage="assemble", work_dir=".")

pipeline = Pipeline(shots)
pipeline.fetch("ip", PtDataSignal("ip"))
pipeline.fetch(
    "betan",
    MdsSignal(r"\betan", "efit01", location="remote://atlas.gat.com"),
)
pipeline.map(peaks)
pipeline.keep(["peaks"])
pipeline.write("peaks", field="peaks", fmt="netcdf")

results = pipeline.compute_multiprocessing(num_workers=8, provenance=run)

run.metrics("coverage", {"requested": len(shots), "returned": len(results)})
run.finalize()
```

Nothing there hand-writes a `cmflib.log_dataset` call. TokSearch derives the
run description and `CmfRun` records it.

**Prerequisites.** cmflib records the executing script's commit and hands
output paths to DVC, so the script must run from inside a git repository that
has a remote and DVC initialised, with the script itself committed there.
`CmfRun` checks for the repository up front rather than failing after a long
compute.

**Run it with `python -m fdp run`, not `fdp run`.** In any environment
carrying cmflib, graphviz arrives transitively (`cmflib → dvc → pydot →
graphviz`) and installs its own layout engine at `bin/fdp`. `fdp` 0.6.0
declared graphviz as a dependency so the installer's link order gives the FDP
CLI the file back (verified on pixi/rattler and micromamba 2.9.0). That still
leaves two ways to lose the collision: an `fdp` older than 0.6.0, or an
installer whose link order isn't guaranteed the way those two are.
`python -m fdp` sidesteps the question either way:

```bash
python -m fdp run python betan_ip_peaks_cmf.py
```

A complete working example is
[`examples/betan_ip_peaks_cmf.py`](https://github.com/GA-FDP/toksearch_cmf/blob/main/examples/betan_ip_peaks_cmf.py)
in the `toksearch_cmf` repository. The prerequisite passage above is kept
deliberately in sync with the same passage in that repository's README; if you
change one, change the other.

## API reference

::: toksearch.provenance.Provenance
    handler: python
    options:
        show_root_heading: True
        members_order: source

::: toksearch.provenance.RunContext
    handler: python
    options:
        show_root_heading: True

::: toksearch.provenance.SourceSpec
    handler: python
    options:
        show_root_heading: True

::: toksearch.provenance.OpSpec
    handler: python
    options:
        show_root_heading: True

::: toksearch.provenance.BackendSpec
    handler: python
    options:
        show_root_heading: True

::: toksearch.provenance.CodeSpec
    handler: python
    options:
        show_root_heading: True

::: toksearch.provenance.JsonProvenance
    handler: python
    options:
        show_root_heading: True

::: toksearch.provenance.safe_call
    handler: python
    options:
        show_root_heading: True

### Pipeline output operations

`Pipeline.write` is rendered in full on the [Pipeline](pipeline.md) reference
page, and `RecordSet.to_dataframe` / `RecordSet.to_parquet` on the
[RecordSet](record_set.md) page. Their arguments are tabulated under
[Getting data out](#getting-data-out) above.
````

Do **not** add `::: toksearch.Pipeline.write` or
`::: toksearch.record.record_set.RecordSet.to_dataframe` blocks here.
`docs/pipeline.md` already renders all of `toksearch.Pipeline` and
`docs/record_set.md` all of `toksearch.record.record_set.RecordSet`, so a
second block registers a duplicate anchor and mkdocs-autorefs warns. Cross-link
instead, which is what the text above does.

- [ ] **Step 2: Add the nav entry**

In `mkdocs.yml`, under `Tutorials`, after the
`"Combining data after pipeline computation"` line:

```yaml
    - "Provenance and CMF": provenance.md
```

- [ ] **Step 3: Verify every symbol on the page actually exists**

```bash
cd /fusion/projects/dt/sammuli/fdp_dev/repos/toksearch_d3d && PYTHONNOUSERSITE=1 pixi run python -c "
import toksearch.provenance as p
for n in ['Provenance','RunContext','SourceSpec','OpSpec','BackendSpec',
          'CodeSpec','JsonProvenance','safe_call']:
    assert hasattr(p, n), n
from toksearch import Pipeline
from toksearch.record.record_set import RecordSet
for n in ['write','compute_serial','compute_multiprocessing']:
    assert hasattr(Pipeline, n), n
for n in ['to_dataframe','to_parquet']:
    assert hasattr(RecordSet, n), n
print('all symbols present')
"
```

Expected: `all symbols present`.

Note: `RecordSet` is **not** re-exported at the toksearch top level
(`toksearch.RecordSet` raises `AttributeError`). The identifier that resolves is
`toksearch.record.record_set.RecordSet`, which is what `docs/record_set.md`
already uses. The verification script above imports it by that path.

- [ ] **Step 4: Commit**

```bash
cd /fusion/projects/dt/sammuli/fdp_dev/repos/toksearch
git add docs/provenance.md mkdocs.yml
git commit -F - <<'MSG'
docs: document provenance and the CMF backend

toksearch.provenance, `compute_*(provenance=...)`, Pipeline.write and
RecordSet.to_dataframe/to_parquet all shipped in 2.11.x, and
toksearch_cmf 0.1.1 is blessed by fdp-core 1.4.0. None of it appeared on
the site.

New page covering the interface, how a run gets recorded, how to get data
out per shot rather than by concatenating on the driver, and the CmfRun
walkthrough with its git+DVC prerequisites. The sample run.json is real
captured output, not an illustration.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_014c9v9tjK6XWLKxNrvXuzx4
MSG
```

---

## Task 8: `toksearch` — verify the docs build

**Files:** none modified unless the build reports a problem.

- [ ] **Step 1: Build the site**

```bash
cd /fusion/projects/dt/sammuli/fdp_dev/repos/toksearch && \
  PYTHONNOUSERSITE=1 pixi run -e docs docs-build 2>&1 | tail -30
```

Expected: ends with `INFO - Documentation built in N seconds`. The
`DeprecationWarning: autorefs 'span' elements` messages are pre-existing noise
and are not a failure.

**If it fails with `AttributeError: 'NoneType' object has no attribute
'replace'` in pygments**, you forgot `PYTHONNOUSERSITE=1`.

- [ ] **Step 2: Check for warnings naming the files this work touched**

```bash
cd /fusion/projects/dt/sammuli/fdp_dev/repos/toksearch && \
  PYTHONNOUSERSITE=1 pixi run -e docs docs-build 2>&1 | \
  grep -iE "warning|error" | grep -viE "DeprecationWarning|autorefs" | \
  grep -iE "provenance|index|llm|README"
```

Expected: no output. Anything here — an unresolved mkdocstrings identifier, a
broken relative link — must be fixed before the task closes.

- [ ] **Step 3: Confirm the new page reached the built site**

```bash
cd /fusion/projects/dt/sammuli/fdp_dev/repos/toksearch && \
  test -f site/provenance/index.html && \
  grep -c "Provenance and CMF" site/provenance/index.html
```

Expected: a count of at least 1.

- [ ] **Step 4: Do not commit the build output**

`site/` is untracked and not in `.gitignore`. Leave it untracked; do not
`git add` it. If you want it ignored, that is a separate change and not part of
this work.

```bash
cd /fusion/projects/dt/sammuli/fdp_dev/repos/toksearch && git status --short
```

Expected: `?? site/` and nothing else uncommitted.

---

## Task 9: `toksearch_cmf` — rewrite the README

**Files:**
- Modify: `/fusion/projects/dt/sammuli/fdp_dev/repos/toksearch_cmf/README.md`

The file still says "Scaffold only: `CmfRun` is not implemented yet." `CmfRun`
shipped; `toksearch_cmf` 0.1.1 is blessed by `fdp-core` 1.4.0.

- [ ] **Step 1: Replace the file**

````markdown
# toksearch_cmf

CMF provenance backend for [toksearch](https://github.com/GA-FDP/toksearch).
Records curated toksearch pipeline runs to the
[Common Metadata Framework](https://github.com/HewlettPackard/cmf).

This package owns every `cmflib` and `dvc` dependency; `toksearch` core knows
nothing about either. Core produces a `RunContext`
(`toksearch.provenance`) describing the run — shot source, signal
specifications, operation sequence, compute backend, git commit — and this
package consumes it.

## Install

`toksearch_cmf` is part of the `fdp-core` metapackage, so an environment
created with `fdp-install` already has it. Standalone:

```bash
conda install -c ga-fdp -c conda-forge toksearch_cmf
```

## Use

```python
from toksearch_cmf import CmfRun

run = CmfRun("betan-ip-study", stage="assemble", work_dir=".")

results = pipeline.compute_multiprocessing(num_workers=16, provenance=run)

run.metrics("coverage", {"requested": len(shots), "returned": len(results)})
run.finalize()
```

Nothing there hand-writes a `cmflib.log_dataset` call. toksearch derives the
run description; `CmfRun` records it. Output directories declared with
`Pipeline.write` are picked up automatically; use `run.output(path, ...)` for
artifacts toksearch did not write itself.

## Prerequisites

cmflib records the executing script's commit and hands output paths to DVC. The
script must therefore run from inside a git repository that has a remote and an
initialised DVC, with the script itself committed there. `CmfRun` checks for the
repository up front rather than failing after a long compute.

Run it as `python -m fdp run python your_script.py`. In any environment carrying
cmflib, graphviz arrives transitively (`cmflib → dvc → pydot → graphviz`) and
installs its own layout engine at `bin/fdp`; `fdp` 0.6.0 fixed the link order so
the FDP CLI keeps the file, but `python -m fdp` is unambiguous regardless.

## Example

[`examples/betan_ip_peaks_cmf.py`](examples/betan_ip_peaks_cmf.py) is a
complete curated pipeline against real DIII-D data: per-shot βN and Ip peaks
from two different signal classes (`PtDataSignal` reads a PTData diagnostic,
`MdsSignal` reads an EFIT equilibrium quantity), written one netCDF file per
shot and recorded to a local CMF store.

## Documentation

See [Provenance and CMF](https://ga-fdp.github.io/toksearch/latest/provenance/)
in the TokSearch documentation for the full picture, including the
`toksearch.provenance` interface this package implements.
````

- [ ] **Step 2: Confirm the stale claim is gone**

```bash
cd /fusion/projects/dt/sammuli/fdp_dev/repos/toksearch_cmf && \
  grep -c "Scaffold only" README.md
```

Expected: `0` (grep exits 1; that is fine).

- [ ] **Step 3: Confirm the example path referenced actually exists**

```bash
cd /fusion/projects/dt/sammuli/fdp_dev/repos/toksearch_cmf && \
  test -f examples/betan_ip_peaks_cmf.py && echo present
```

Expected: `present`.

- [ ] **Step 4: Commit**

```bash
cd /fusion/projects/dt/sammuli/fdp_dev/repos/toksearch_cmf
git checkout -b docs/readme-refresh
git add README.md
git commit -F - <<'MSG'
docs: describe the package that shipped

The README still said "Scaffold only: CmfRun is not implemented yet."
CmfRun landed in 089873c and 0.1.1 is blessed by fdp-core 1.4.0. Describe
the actual surface, the git+DVC prerequisite, the `python -m fdp run`
caveat, and point at the worked example and the TokSearch provenance page.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_014c9v9tjK6XWLKxNrvXuzx4
MSG
```

---

## Final verification

- [ ] **`fdp` suite is green**

```bash
cd /fusion/projects/dt/sammuli/fdp_dev/repos/fdp && \
  PYTHONNOUSERSITE=1 pixi run pytest tests/ -q
```

Expected: 251 passed.

- [ ] **`toksearch` suite is unaffected** (this work touches no source, but
      prove it)

```bash
cd /fusion/projects/dt/sammuli/fdp_dev/repos/toksearch && \
  PYTHONNOUSERSITE=1 pixi run test 2>&1 | tail -5
```

- [ ] **Docs build clean** — Task 8, Steps 1–3.

- [ ] **No stale claim survives anywhere**

```bash
cd /fusion/projects/dt/sammuli/fdp_dev/repos
grep -rn "environment.yml" toksearch/README.md toksearch/docs/*.md
grep -rn "defaults to --backend amsc" toksearch/docs/
grep -rn "Scaffold only" toksearch_cmf/README.md
```

Expected: no output from any of the three.

- [ ] **`toksearch query` is still documented** — the user explicitly asked for
      this; it exists in `toksearch/llm/cli.py:295` and in `release-2.11.1`.

```bash
cd /fusion/projects/dt/sammuli/fdp_dev/repos/toksearch && \
  grep -c "toksearch query" docs/llm.md docs/index.md README.md
```

Expected: a nonzero count for `docs/llm.md` and for the index/README (they use
the `toksearch query "..."` form in the CLI block).
