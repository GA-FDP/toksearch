# Skills MCP server for `toksearch.llm`

**Status:** Draft for review
**Date:** 2026-06-02
**Owner:** sammuli
**Issue:** GA-FDP/toksearch#37 (Phase 2: MCP server for `toksearch.llm.skills`)
**Design context:** `2026-05-22-platform-architecture-handoff.md` (§ Phase 2)

## Motivation

The 2026-05-22 platform-architecture handoff laid out a phased move toward a
schema- and MCP-based platform: each layer (`fdp`, `toksearch`,
`toksearch_d3d`) eventually exposes an MCP server so any agent — in any
language — can self-describe and consume its API without importing the
implementation package. Phase 1 (fdp-schema catalog migration) has landed.

Phase 2 is the **smallest concrete test of the MCP-based API pattern**: take
the simplest existing `toksearch.llm` entry-point group — `toksearch.llm.skills`
(documentation skills, each a `SKILL.md` file) — and serve it from a standalone
MCP server instead of loading it in-process. It is deliberately small so the
MCP plumbing (server, stdio transport, a sync client bridge, resource +
tool surfaces) gets proven end-to-end on low-stakes content before any
higher-value API (pipeline construction, catalog resolution) is wrapped.

The substrate is already in place: `mcp >=1.23` (env has 1.27.1) and
`claude-agent-sdk >=0.2` are run-deps of `toksearch`, and the `claude_sdk`
backend already runs an *in-process* SDK MCP server for `run_python` /
`lookup_docs`. This spec adds the first *standalone* MCP server.

## Current state

`toksearch.llm.skills` is an entry-point group. Each entry resolves to a
directory; `discovery.discover_skill_dirs()` collects them, and
`tools.discover_skills()` scans each one level deep for `SKILL.md` files,
parsing frontmatter (`name`, `description`) and body via
`tools.parse_skill_md()`. The result is a `dict[str, Skill]` held on the
`Session` as `self.skills`.

Skills surface two ways:
1. **Catalog** — `prompts.build_system_prompt()` lists each skill's name +
   description in the system prompt (names + descriptions only; no bodies).
2. **Body on demand** — the `lookup_docs` tool (`tools.LOOKUP_DOCS`) returns
   `session.skills[name].body`.

Contributors today: `toksearch` itself (4 core skills in `toksearch/skills/`,
via `CORE_SKILLS_DIR`) and `toksearch_d3d` (4 skills, via its own
`toksearch.llm.skills` entry point).

## Goals

- A **standalone stdio MCP server** (`python -m toksearch.llm.mcp`) that any
  MCP client (the chat GUI, Claude Code, another agent) can launch and query.
- Skills exposed as MCP **resources** (`skill://<name>`, the canonical
  representation) **and** via a `read_skill` **tool** (so models that don't
  proactively read resources still work).
- The chat client (`Session`) consumes skills **as an MCP client**, uniformly
  across all backends (anthropic, openai, claude-max), so "the GUI works
  against MCP" is a single code path regardless of backend.
- Zero churn for contributor packages: the `toksearch.llm.skills` entry-point
  group remains the discovery source.
- The standalone server also accepts **extra skill dirs via config**
  (`TOKSEARCH_SKILL_DIRS`), per the handoff's "data not just code" direction.

## Non-goals

- Wrapping `run_python` or any pipeline operation as a standalone MCP server
  (that is Phase 3+, pulled by demand).
- Migrating `toksearch.llm.namespace` or `toksearch.llm.presets` to MCP.
- A language-neutral re-spec of the skill format. `SKILL.md` + frontmatter
  stays exactly as-is.
- Any permanent in-process consumption fallback (see Decisions).

## Key decisions

These were settled during brainstorming (see issue #37 discussion):

1. **Standalone stdio server**, not in-process-only. Truest test of the
   pattern; reusable by external clients.
2. **Resources + a lookup tool**, not resources-only or tools-only. Resources
   are the canonical MCP representation; the tool guarantees any model can pull
   a body.
3. **Session is the MCP client** for all backends (Approach A). One subprocess,
   one connection, identical behavior everywhere. The `claude_sdk` backend does
   *not* get its own separate `mcp_servers` wiring for skills — its in-process
   `lookup_docs` SDK tool proxies to `session._execute_tool`, which hits the
   same client.
4. **Discovery = entry-points + extra configured dirs.** The server seeds from
   `discover_skill_dirs()` and merges `TOKSEARCH_SKILL_DIRS`.
5. **No permanent fallback; fail loudly.** If the skills server can't spawn,
   `Session.__init__` raises `LLMSkillsError`. The entry-point *group* stays
   (Sense 1 of "parallel"); the Session's in-process *consumption* path is
   replaced, not kept as a runtime branch (no Sense-2 dual path). This keeps a
   single hot path and prevents a silent fallback from masking MCP regressions.
   `mcp` is a hard run-dep, so "mcp missing" cannot occur in a real install.

## High-level design

### Architecture

```
                    python -m toksearch.llm.mcp   (standalone, stdio)
                    ┌───────────────────────────────────────────┐
                    │  build_server(extra_dirs) -> FastMCP        │
   entry points ───▶│   resources: skill://<name>  (list/read)    │◀── any external
   + $TOKSEARCH_     │   tool:      read_skill(skill_name)         │    MCP client
     SKILL_DIRS      └───────────────────────────────────────────┘
                                      ▲ stdio
                                      │
            ┌─────────────────────────┴───────────────────────────┐
            │  SkillsMcpClient (sync facade, daemon-thread loop)   │
            │   list_skills() -> {name: SkillMeta}                 │
            │   read_skill(name) -> body                           │
            └─────────────────────────┬───────────────────────────┘
                                      │
                                  Session
            ┌─────────────────────────┼───────────────────────────┐
       build_system_prompt      lookup_docs tool            (all backends:
       (names + descriptions)   -> client.read_skill()       anthropic / openai /
                                                              claude-max)
```

### Package layout

New subpackage `toksearch/llm/mcp/`, parallel to `backends/` and `gui/`:

| File | Responsibility |
|------|----------------|
| `mcp/__init__.py` | Exports `build_server`, `SkillsMcpClient`. |
| `mcp/server.py` | `build_server(extra_dirs: list[Path] \| None) -> FastMCP`. Pure, importable, testable without a subprocess. Registers resources + the `read_skill` tool. |
| `mcp/__main__.py` | `python -m toksearch.llm.mcp`. Reads `TOKSEARCH_SKILL_DIRS` (os.pathsep-delimited), builds the server, calls `.run("stdio")`. |
| `mcp/client.py` | `SkillsMcpClient` — sync facade over the async stdio client; daemon-thread event loop; `list_skills()`, `read_skill()`, `close()`. |

Reused unchanged as server internals: `discovery.discover_skill_dirs`,
`tools.discover_skills`, `tools.parse_skill_md`, `tools.Skill`.

## Detailed design

### Server (`mcp/server.py`)

```python
def build_server(extra_dirs: list[Path] | None = None) -> FastMCP:
    mcp = FastMCP("toksearch-skills")
    skill_dirs = [d for _, d in discover_skill_dirs()]
    if extra_dirs:
        skill_dirs.extend(extra_dirs)
    skills = discover_skills(skill_dirs)   # {name: Skill(name, description, body)}

    # Resources: one skill:// URI per skill. list returns name+description;
    # read returns the body. Registered with a closure over `skills`.
    for name, skill in skills.items():
        @mcp.resource(f"skill://{name}", name=name,
                      description=skill.description, mime_type="text/markdown")
        def _read(body=skill.body) -> str:
            return body

    @mcp.tool(description="Read a documentation skill's SKILL.md body by name.")
    def read_skill(skill_name: str) -> str:
        s = skills.get(skill_name)
        if s is None:
            raise ValueError(
                f"Unknown skill: {skill_name!r}. Available: {sorted(skills)}")
        return s.body

    return mcp
```

Notes:
- Skills are discovered once at `build_server` time. The server process is
  short-lived (spawned per `Session`); a long-running external server restarts
  to pick up newly-installed packages, consistent with how entry points work
  today.
- The `read_skill` tool error mirrors the current `lookup_docs` unknown-skill
  message so behavior is preserved.

### Entry point (`mcp/__main__.py`)

```python
import os
from pathlib import Path
from .server import build_server

def main() -> None:
    raw = os.environ.get("TOKSEARCH_SKILL_DIRS", "")
    extra = [Path(p) for p in raw.split(os.pathsep) if p]
    build_server(extra_dirs=extra).run("stdio")

if __name__ == "__main__":
    main()
```

### Client (`mcp/client.py`)

`SkillsMcpClient` wraps the async `mcp` stdio client behind a sync API, reusing
the daemon-thread event-loop bridge pattern from
`backends/claude_sdk.py` (persistent loop on a daemon thread;
`run_coroutine_threadsafe` for each call).

```python
class SkillsMcpClient:
    def __init__(self, extra_dirs: list[Path] | None = None,
                 command: list[str] | None = None): ...
    def list_skills(self) -> dict[str, "SkillMeta"]: ...   # name -> (name, description)
    def read_skill(self, name: str) -> str: ...            # resources/read or read_skill tool
    def close(self) -> None: ...                            # cancel loop, terminate subprocess
```

- **Spawn:** `StdioServerParameters(command=sys.executable,
  args=["-m", "toksearch.llm.mcp"], env={... "TOKSEARCH_SKILL_DIRS": ...})`.
  `command` is overridable for tests.
- **Connect:** `stdio_client(...)` → `ClientSession(...)` → `initialize()`,
  then `list_resources()` to build the catalog (`SkillMeta(name, description)`
  parsed from each resource's name/description).
- **`read_skill`:** calls `read_resource(f"skill://{name}")` (canonical path);
  the `read_skill` tool is the equivalent for tool-only clients. The Session's
  `lookup_docs` handler uses `read_skill()`.
- **Lifecycle:** `close()` cancels the loop and terminates the child; registered
  via `atexit`. Spawn/connect failure raises `LLMSkillsError`.

### Session integration (`session.py`)

Replace the in-process skills block (current lines ~92–100):

```python
# ---- Skills: standalone MCP server ----
extra = list(extra_skill_dirs or [])
try:
    self._skills_client = SkillsMcpClient(extra_dirs=extra)
    catalog = self._skills_client.list_skills()
except Exception as e:
    raise LLMSkillsError(...) from e
if packages is not None:
    catalog = {n: m for n, m in catalog.items() if n in packages}
self.skills = catalog                       # {name: SkillMeta}; bodies fetched lazily
```

- `build_system_prompt(self.skills, namespace_entries)` is **unchanged** — it
  reads only names + descriptions.
- `Session.close()` (new) calls `self._skills_client.close()`. `Session.reset()`
  keeps the client (resets only namespace/history).

Note on the `packages` filter: today it filters skill *dirs* before scanning;
now it filters the *catalog* by skill name. The skill names contributed by a
package match the directory names under that package's skill dir, so the filter
must key on the entry-point/package name. **Resolution:** the server tags each
resource's `meta` with the contributing entry-point name, and `list_skills()`
surfaces it, so the Session filters by package as before. (If `meta`
propagation proves awkward in the `mcp` API, fall back to filtering by skill
name set — documented in the plan.)

### Tool wiring (`tools.py`)

`_lookup_docs_handler` changes its body source:

```python
def _lookup_docs_handler(args, session) -> ToolOutput:
    name = args["skill_name"]
    try:
        body = session._skills_client.read_skill(name)
    except Exception:
        return ToolOutput(text=f"Unknown skill: {name!r}. "
                               f"Available: {sorted(session.skills)}", is_error=True)
    return ToolOutput(text=body, is_error=False)
```

All three backends reach this handler:
- anthropic / openai: call `ToolSpec.handler(args, session)` directly.
- claude-max: the in-process SDK `lookup_docs` tool calls
  `session._execute_tool(block)` → same handler → same client.

### Errors (`errors.py`)

Add `LLMSkillsError(LLMError)` for skills-server spawn/connect failures.

## Testing

New + updated tests, mirroring `tests/test_llm_*.py`:

- `test_llm_mcp_server.py` — unit-test `build_server()` with no subprocess (in-
  memory FastMCP test transport): `resources/list` yields the expected
  `skill://` URIs + descriptions for a temp skill dir; `resources/read` returns
  the body; `read_skill` returns the body and raises cleanly on unknown name;
  `extra_dirs` are merged.
- `test_llm_mcp_client.py` — spawn the real `python -m toksearch.llm.mcp` over
  stdio against a temp skill dir: `list_skills()` / `read_skill()` round-trip;
  `LLMSkillsError` on a bogus server command; `close()` reaps the subprocess;
  `TOKSEARCH_SKILL_DIRS` honored.
- `test_llm_session.py` (update) — `Session` populates `self.skills` via the
  client; `lookup_docs` reads bodies through it; `packages` filter still works.
- `test_llm_gui_app.py` (update) — a GUI Session exposes the skills catalog
  (unit-level "GUI works against MCP").
- End-to-end smoke test with the `fake` backend: a full `send()` that triggers
  `lookup_docs`, proving the resource read flows through MCP.

## Migration plan

- **Kept in parallel (Sense 1):** the `toksearch.llm.skills` entry-point group
  is still the discovery source; `discover_skill_dirs` / `discover_skills` /
  `parse_skill_md` remain as server internals. Contributor packages
  (`toksearch_d3d`) change nothing.
- **Replaced:** the Session's in-process skill loading and the `lookup_docs`
  body source now go through the standalone MCP server. No silent fallback.
- **Manual verification:**
  1. Launch the chat GUI / `fdp chat` on a real backend; confirm a skill lookup
     works.
  2. Independently connect to `python -m toksearch.llm.mcp` from an external MCP
     client (e.g. Claude Code or the `mcp` dev inspector) and confirm
     `resources/list`, `resources/read`, and the `read_skill` tool — the
     standalone-server proof.

## Risks

- **Subprocess spawn latency / sandbox restrictions.** One spawn per `Session`
  construction. Cost is comparable to today's entry-point import. In sandboxes
  that block subprocess+stdio, `Session` construction now fails loudly — an
  accepted tradeoff of decision #5; documented in the error message.
- **`@mcp.resource` dynamic registration in a loop.** Closure-over-loop-variable
  is a classic Python footgun; the design binds `body=skill.body` as a default
  arg. Verified pattern; covered by `test_llm_mcp_server.py`.
- **`packages` filter keying.** Mitigated via resource `meta`; fallback
  documented above.
- **Async-bridge correctness.** Reuses the proven `claude_sdk.py` daemon-loop
  pattern; `close()`/atexit must reliably reap children to avoid orphans —
  explicitly tested.

## Out of scope

- A second `mcp_servers` wiring for the claude-max backend (Approach B).
- Caching skill bodies in the Session (the lookup is cheap and infrequent).
- Hot-reload of skills within a running server process.

## Open questions

1. Does the installed `mcp` 1.27 API expose resource `meta` on
   `list_resources()` results conveniently enough to carry the package name for
   the `packages` filter, or do we fall back to name-set filtering? (Resolve in
   the plan's first task by spiking the API.)
   **Resolved (implementation):** the packages filter is applied server-side via the TOKSEARCH_SKILL_PACKAGES env var; resource meta is not used. URI scheme remains skill://<name>.
2. Should `Session.close()` be surfaced in the CLI/GUI teardown explicitly, or
   is atexit reaping sufficient for interactive use? (Lean: atexit is enough;
   add explicit `close()` in the GUI's shutdown path if tests show orphans.)
