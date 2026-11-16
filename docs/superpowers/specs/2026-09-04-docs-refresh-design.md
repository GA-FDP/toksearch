# TokSearch docs refresh: talk-to-your-data, installation, provenance/CMF

**Date:** 2026-09-04
**Status:** approved, not yet implemented
**Scope:** `toksearch` (docs), `fdp` (one code fix + tests), `toksearch_cmf` (README)

## Problem

The published TokSearch documentation has drifted from the shipped software in
four independent ways.

**Installation is wrong about how people actually install.** `README.md` and
`docs/index.md` present `conda install -c ga-fdp toksearch` as the way in. The
real path for essentially every user is `fdp-install`, which drops a
`pixi.toml` depending on the `fdp-core` metapackage (currently 1.4.0). A bare
`toksearch` install gets no `toksearch_d3d`, no `fdp` CLI, no XRootD transport,
and therefore no DIII-D data. The "Installation from Source" section is worse
than stale: it instructs `mamba env create -f environment.yml`, and
`environment.yml` does not exist in the repository. The repo is pixi-based.

**"Talk to your data" describes only half the story.** It presents
`toksearch chat` as the single conversational surface. Since then the skills
MCP server shipped (`python -m toksearch.llm.mcp`, toksearch >= 2.8.2), along
with `fdp skills install` for Claude Code / Cursor / Codex. A user with Claude
Code already installed has a first-class path that the docs never mention. The
section also omits `toksearch chat --gui` (Gradio), the `toksearch backends`
subcommand, and the `-v/--verbose` and `--package` flags.

**Provenance and CMF are entirely undocumented.** `toksearch.provenance`
(`RunContext`, the `Provenance` ABC, `JsonProvenance`), the
`compute_*(provenance=...)` keyword, `Pipeline.write`, and
`RecordSet.to_dataframe`/`to_parquet` all shipped in 2.11.x. `toksearch_cmf`
0.1.1 (`CmfRun`) is blessed by `fdp-core` 1.4.0. None of it appears on the
site. `toksearch_cmf`'s own README still reads "Scaffold only: `CmfRun` is not
implemented yet."

**Two claims about `fdp chat` are false.** The LLM tutorial says `fdp chat`
runs "with the FDP environment pre-configured (XRootD plugin, MDSplus tree
paths, BEARER_TOKEN)" and that it "defaults to `--backend amsc`". Commit
`fe72baa` in the `fdp` repo made both untrue: `fdp chat` and `fdp query` are
marked `needs_env=False`, so `main()` never calls `setup_environment()`, and
`_build_llm_cmd` stopped injecting a device-derived backend.

Confirmed against the installed `fdp` 0.6.0:

```
needs_env for chat: False
needs_env for run:  True
```

The reason `fe72baa` did this was legitimate: `setup_environment()` raises when
no device contributor is installed, which broke `fdp chat --help` inside fdp's
own dev environment. But turning the environment off entirely is a bigger
hammer than the problem needed, and it leaves the preferred conversational
entry point unable to reach DIII-D data.

## Decisions taken

Confirmed with the user before writing this spec:

1. "Talk to your data" presents the built-in CLI and the bring-your-own-agent
   path as **co-equal**, with a note that the `fdp` variants (from an
   `fdp-core` install) are the **preferred** way to run either.
2. `toksearch query` **stays documented**. It exists in
   `toksearch/llm/cli.py:295` and in the released `release-2.11.1` tag, and
   `fdp query` wraps it. The belief that it had been removed was mistaken.
3. Provenance/CMF gets a **dedicated page plus API-reference blocks**.
4. The stale `toksearch_cmf` README **is fixed** in the same effort.
5. The `fdp chat` environment gap is **fixed in code**, and the docs are
   written against the fixed behavior.

## Design

### Part 1 — `fdp`: best-effort environment for `chat` and `query`

Introduce a third state for the existing `needs_env` flag. Today it is a
boolean: `True` means "set up the environment or exit 1", `False` means "do not
touch the environment". Add `"best-effort"`: attempt setup, and on a failure
that means *no device is available*, continue without it.

In `build_parser()`:

```python
p_chat.set_defaults(func=do_chat, needs_env="best-effort", auto_login=True)
p_query.set_defaults(func=do_query, needs_env="best-effort", auto_login=True)
```

In `main()`:

```python
needs_env = getattr(args, "needs_env", True)
if needs_env:
    best_effort = needs_env == "best-effort"
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
        # chat/query are useful with no device installed at all -- an LLM
        # session that never fetches shot data is a legitimate use. Warn,
        # don't die.
        print(f"Warning: continuing without FDP environment ({exc})",
              file=sys.stderr)
    except auth.AuthError as exc:
        if not best_effort:
            print(f"Login failed: {exc}", file=sys.stderr)
            sys.exit(1)
        print(f"Warning: continuing without a bearer token ({exc})",
              file=sys.stderr)
```

Why this is correct rather than a re-introduction of the `fe72baa` bug: the
recorded FDP failure mode is that calling `setup_environment()` *in-process and
then using MDSplus in that same process* does not work, because libfdpio and
XRootD read their configuration at C-library load time. `fdp chat` does not do
that. It `os.execvpe`s into a fresh `python -m toksearch.llm.cli` process, so
the environment is in place before the new process loads any C library. That is
exactly the `fdp run` pattern, and `fdp run` is the known-good path.

`auto_login=True` matches `fdp run`: an interactive session that is about to
fetch shot data should acquire a token the same way. If login fails (the known
pelican stdout-TTY issue is one way it can), the `best-effort` branch warns and
proceeds rather than blocking a session that may never need a token.

Cleanup in code being touched: `_resolve_default_handle_or_none` and the
`handle` parameter threaded through `do_chat`/`do_query` into `_build_llm_cmd`
are vestigial — `_build_llm_cmd` documents that it does not consult the handle
and keeps the parameter "for call-site compatibility". Drop the parameter and
the resolver.

Tests in `fdp/tests/test_llm_shims.py` (or `test_cli.py`, whichever the
existing `needs_env` coverage lives closest to):

- `chat` and `query` call `setup_environment` when a device resolves.
- `chat` and `query` survive `setup_environment` raising `ValueError` (the bare
  dev env case `fe72baa` was fixing) and still exec the delegate.
- `chat` survives `auth.AuthError` and still execs.
- A non-best-effort subcommand (`ls`) still exits 1 on `ValueError` — the
  regression guard for the new branch.
- `_build_llm_cmd` no longer takes a handle.

### Part 2 — `toksearch`: `README.md` and `docs/index.md`

These two files are near-identical today and must stay in sync.

**"Talk to your data"** becomes two co-equal subsections plus a preferred-path
note:

- *Built-in CLI* — `toksearch chat`, `toksearch query`, `toksearch chat --gui`,
  `toksearch backends`. Keep the existing transcript example; it is accurate.
- *Your own agent* — `fdp skills install` (Claude Code, Cursor, Codex) and
  `claude mcp add toksearch-skills -- fdp run python -m toksearch.llm.mcp`.
  Same skills, consumed by an agent the user already has.
- *Note:* both are preferred as their `fdp` variants (`fdp chat`, `fdp query`,
  `fdp skills`) from an `fdp-core` / `fdp-core-latest` install, because those
  come with the device packages and the data-access environment.

**Installation** is reordered:

1. **Recommended: the FDP environment.** `conda create -n fdp-installer -c
   ga-fdp -c conda-forge fdp-installer`, `conda activate fdp-installer`,
   `fdp-install -d /path/to/project`, `cd` there, `pixi shell`. Note
   `fdp-install --latest` for `fdp-core-latest` (`>=` floors instead of exact
   pins). State plainly what this gets: `toksearch`, `toksearch_d3d`,
   `toksearch_mast`, `ptdata`, `imas_composer`, the `fdp` CLI, XRootD/Pelican
   transport, and `toksearch_cmf`.
2. **TokSearch alone**, for users who want the framework without FDP data
   access: the existing `conda install -c ga-fdp -c conda-forge toksearch`,
   explicitly flagged as not providing DIII-D access.
3. **From source**, rewritten for pixi: clone, `pixi install`, `pixi run build`,
   `pixi run test`. Delete the `environment.yml` instructions.

Keep the "In the near future, we will provide a way to install TokSearch
directly from PyPI using pip" line: `https://pypi.org/pypi/toksearch/json`
returns 404, so TokSearch is still not on PyPI and the sentence is accurate.
The `pip install toksearch[llm]` line in `docs/llm.md` and the tutorial
therefore describes installing *from a source checkout* with the extra, not
from PyPI, and must be reworded to say so.

### Part 3 — `toksearch`: `docs/llm.md`

- Quickstart CLI section gains `toksearch backends`, `--gui`, `-v/--verbose`,
  and `--package`.
- Correct the `fdp` wrapper paragraph to describe the post-fix behavior, and
  document the flag asymmetry: `fdp chat`/`fdp query` forward only
  `--backend`, `--model`, `-n/--max-iterations`, `--gui`, and `--no-browser`.
  For `--package` or `-v`, use `fdp run toksearch chat ...`.
- Drop the "defaults to `--backend amsc`" claim wherever it appears; the
  precedence chain (`--backend` → `$FDP_LLM_BACKEND` →
  `~/.fdp/config.toml [llm].backend` → built-in `anthropic`) is already
  documented correctly and stays.
- New section **"Using your own agent"**: the standalone MCP server, what it
  serves (`skill://<name>` resources plus a `read_skill` tool),
  `TOKSEARCH_SKILL_DIRS`, `fdp skills list/install --backend
  claude|cursor|codex [--force]`, and the `claude mcp add` line. The existing
  MCP paragraph under "Tools" shrinks to a cross-reference so the server is
  described in one place.

### Part 4 — `toksearch`: `docs/provenance.md` (new)

Structure:

1. **Why** — a computed result is only reusable if you can say what produced
   it. TokSearch derives the run description itself (shot source, signal specs,
   operation sequence, backend, git commit); the user does not hand-write
   metadata.
2. **The interface** — `Provenance` ABC (`on_compute_start`, `on_compute_end`,
   `output`, `metrics`, `finalize`), `RunContext` and its parts (`SourceSpec`,
   `OpSpec`, `BackendSpec`, `CodeSpec`), and `safe_call`'s error isolation: a
   provenance backend that fails must not fail the pipeline.
3. **Using it** — `provenance=` on all four `compute_*` methods, demonstrated
   with `JsonProvenance` (no external services, good for a first look).
4. **Getting data out** — `Pipeline.write` in both forms (declarative
   `field=`/`fields=`, and the decorator form), the `track`, `exist_ok`, and
   `on_error` arguments and why their defaults are what they are, plus
   `RecordSet.to_dataframe` / `to_parquet` for the driver-side path.
5. **Recording to CMF** — `toksearch_cmf.CmfRun`, adapted from
   `toksearch_cmf/examples/betan_ip_peaks_cmf.py`. Must state the
   prerequisites: a git repository with a remote, an initialised DVC, and the
   script committed there. Must carry the `python -m fdp run` caveat — in any
   environment with cmflib, `fdp run` can resolve to graphviz's `bin/fdp`
   (fixed in `fdp` 0.6.0 by depending on graphviz for link order, but
   `python -m fdp` is the guaranteed form).
6. **API reference** — mkdocstrings blocks for `Provenance`, `RunContext`,
   `SourceSpec`, `OpSpec`, `BackendSpec`, `CodeSpec`, `JsonProvenance`,
   `Pipeline.write`, `RecordSet.to_dataframe`, `RecordSet.to_parquet`.

Code in this page is illustrative and not executed by the build (it is a `.md`
page, not a notebook). Every snippet must nonetheless be derived from the
working example or from the actual signatures, not invented.

### Part 5 — `toksearch`: `mkdocs.yml`

Add under Tutorials, after "Combining data after pipeline computation":

```yaml
- "Provenance and CMF": provenance.md
```

The API-reference blocks live inside `provenance.md`, so no second nav entry is
needed.

### Part 6 — `toksearch_cmf`: `README.md`

Replace the "Scaffold only" line. Cover: what `CmfRun` records, the
`CmfRun(pipeline_name, stage=..., work_dir=...)` signature, the git+DVC
prerequisite, a pointer to `examples/betan_ip_peaks_cmf.py`, and the fact that
this package owns every `cmflib`/`dvc` dependency while toksearch core knows
nothing about either. Separate repository, separate commit.

## Non-goals

- No change to the `toksearch` LLM CLI surface. `query` stays exactly as it is.
- No rewrite of the other tutorial notebooks (Overview, Working with Signals,
  Xarray, Parallelization, Aggregating Data). They are out of scope unless a
  claim in them contradicts something this work changes.
- No new provenance backends. `JsonProvenance` and `CmfRun` are what ship.
- No `fdp` release. The code fix lands on `main`; blessing it through
  `fdp-core` is a separate decision.

## Verification

- `pixi run -e docs docs-build` succeeds with no broken-link or missing-file
  warnings for the new page and nav entry.
- `fdp` test suite passes, including the new `needs_env` cases.
- Every CLI invocation printed in the docs is checked against `--help` output
  from the installed package, not from memory. Specifically: `toksearch
  {query,chat,backends} --help`, `fdp {chat,query,backends,skills} --help`,
  `fdp-install --help`.
- Every symbol named in `provenance.md` is confirmed to exist via
  `python -c "import toksearch.provenance as p; print(p.__all__)"` and
  `inspect.signature`.
