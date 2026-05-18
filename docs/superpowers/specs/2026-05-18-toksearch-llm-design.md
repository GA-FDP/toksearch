# toksearch.llm — Conversational LLM interface for TokSearch

**Status:** Draft for review
**Date:** 2026-05-18
**Owner:** sammuli
**Supersedes:** `toksearch_d3d/agents/claude_toksearch_agent.py` (current)

## Motivation

The current LLM-driven query capability (`toksearch_d3d.agents.claude_toksearch_agent`,
exposed via `fdp query "..."`) works but has three structural limits:

1. **One-shot only.** A single prompt produces one answer; there is no conversational
   follow-up. Useful exploratory workflows ("plot Ip for this shot", "now overlay another
   shot", "save it as PNG") would require either re-fetching from scratch each time or
   stitching together unrelated `fdp query` invocations.
2. **Locked to one LLM provider.** Hardcoded to the AmSC Anthropic-compatible endpoint
   with model `claude-sonnet-4-6` and `~/amsc_api_key`. No way to use a direct Anthropic
   API key, an OpenAI key, or a Claude Max plan.
3. **Lives in `toksearch_d3d`** despite being a generic capability. The same agent
   pattern is useful for any TokSearch user (MAST via `ZarrSignal`, future devices), but
   the current location couples it to DIII-D-specific install paths.

This spec replaces the current implementation with a library-first, multi-backend,
conversational design that lives in core `toksearch` and lets device packages
(`toksearch_d3d`, future siblings) plug in additively.

## Goals

- **Conversational.** Multi-turn dialog over a persistent Python namespace so follow-up
  turns iterate on cached results instead of re-fetching.
- **Provider-agnostic.** Anthropic, OpenAI, AmSC (Anthropic-compatible), and Claude Max
  (via Claude Agent SDK) backends, behind a uniform `Session` API. AmSC and any other
  Anthropic-compatible site endpoint are configured as **presets**, not separate classes.
- **Library-first.** Python `Session` class is the primary API; CLI commands (`toksearch
  chat`, `toksearch query`, `fdp chat`, `fdp query`) are thin wrappers.
- **Multi-device per session.** A single session can mix DIII-D and MAST data (or any
  other device whose package is installed); contributors auto-discover via Python entry
  points.
- **Show-then-run transparency.** The REPL displays the agent's code and one-sentence
  rationale before executing, with auto-approve as the default and ctrl-C to interrupt.
  Optional `--confirm` flag for prompt-before-execute.
- **Design for future on-disk persistence** without implementing it now — keep history
  in a provider-neutral, JSON-serializable shape.

## Non-goals

- A fully sandboxed code-execution environment (same blast radius as Claude Code's Bash
  tool; `exec()` runs in-process).
- Claude-Code-like generic agent capabilities (Read/Edit/Bash/Glob/Grep). The Claude
  Agent SDK backend is **strictly a way to bill the Claude Max plan** while keeping
  identical behavior to the raw-API backends. Users who want full Claude Code should
  run `claude` directly; the `fdp skills install` machinery already exists for that.
- Auto-save / resume of session history (designed-for but not implemented).
- Image / multimedia output in the terminal (REPL surfaces saved-file paths; a future
  Jupyter wrapper can inline figures).

## High-level design

### Architecture

Four layers, top to bottom:

```
┌──────────────────────────────────────────────────────────────────┐
│  CLI                                                             │
│  toksearch chat | toksearch query | fdp chat | fdp query         │
└────────────────────────────┬─────────────────────────────────────┘
                             │
┌────────────────────────────▼─────────────────────────────────────┐
│  Session  (toksearch.llm.session)                                │
│  - owns history, namespace, tool registry, backend instance      │
│  - sync send() with optional event callbacks                     │
└──────────┬──────────────────────────────────────┬────────────────┘
           │                                      │
┌──────────▼──────────────┐         ┌─────────────▼────────────────┐
│  Backend (ABC)          │         │  Tools                       │
│  - run_conversation(    │         │  - run_python (exec in       │
│       session, prompt,  │◄────────│    persistent namespace)     │
│       callbacks, ...)   │         │  - lookup_docs (read SKILL.md│
└─┬───────────────────┬───┘         │    from registered skill     │
  │                   │             │    dirs)                     │
  │                   │             └──────────────────────────────┘
┌─▼─────────────┐  ┌──▼──────────────────┐
│_ToolLoopBackend│  │ ClaudeSDKBackend   │
│ Anthropic │ OpenAI│ (Claude Agent SDK +│
│ (AmSC = Anthropic │  in-process MCP    │
│  preset)          │  for run_python)   │
└───────────────────┘  └─────────────────┘
```

- **Session** owns all state; backends are stateless adapters.
- **`_ToolLoopBackend`** holds the shared tool-loop logic for raw API providers; subclasses
  implement only the provider-specific request/response translation.
- **`ClaudeSDKBackend`** is structurally different: the Claude Agent SDK drives its own
  tool loop in the `claude` subprocess; we expose `run_python` and `lookup_docs` as
  custom **in-process MCP tools** so the persistent namespace is shared.

### Package layout

```
toksearch/llm/                  # core repo
├── __init__.py                 # public re-exports
├── session.py                  # Session class
├── events.py                   # TextDelta, ToolCall, ToolResult, TurnComplete
├── messages.py                 # provider-neutral history representation
├── tools.py                    # run_python, lookup_docs, ToolSpec
├── prompts.py                  # kernel system prompt + catalog assembly
├── errors.py                   # LLMError hierarchy
├── config.py                   # Config dataclass + load_config()
├── discovery.py                # entry-point discovery for namespace/skills/presets
├── presets.py                  # backend presets registry
├── cli.py                      # `toksearch chat` / `toksearch query`
└── backends/
    ├── __init__.py             # backend registry, get_backend()
    ├── base.py                 # Backend ABC, _ToolLoopBackend
    ├── anthropic.py            # also serves AmSC via preset
    ├── openai.py               # OpenAI Chat Completions
    └── claude_sdk.py           # Claude Agent SDK

toksearch_d3d/                  # device contributor
├── toksearch_d3d/llm/
│   └── __init__.py             # skills_path, AMSC_PRESET, namespace marker
├── toksearch_d3d/skills/       # unchanged
└── pyproject.toml              # entry points registering the contributor
```

### Public API (Python)

```python
from toksearch.llm import Session, TextDelta, ToolCall, ToolResult, TurnComplete

session = Session(
    backend="anthropic",        # name or Backend instance; default from config/env
    model=None,                  # backend's default if None
    max_iterations=20,
    extra_namespace=None,        # dict merged into the run_python namespace
    packages=None,               # override entry-point discovery (None = all installed)
    extra_skill_dirs=None,       # additional skill directories
    config=None,                 # explicit Config; default loads from ~/.fdp/config.toml
)

result: TurnComplete = session.send(
    "plot Ip for shot 200000",
    on_text=lambda t: print(t, end=""),
    on_tool_call=lambda c: print(f"[{c.name}] {c.thought}"),
    on_tool_result=lambda r: ...,
    on_event=None,               # catch-all alternative
    confirm=None,                # None = auto-approve; callable returning bool to gate
)

session.history     # list[Message] — provider-neutral
session.namespace   # dict — the run_python persistent namespace
session.reset()     # clear history + namespace
```

### CLI

```
toksearch chat [--backend NAME] [--model NAME] [--package NAME ...] [--confirm]
toksearch query "<prompt>" [--backend NAME] [--max-iterations N] [--quiet] [--debug]

fdp chat ...      # shim: setup_environment() + re-exec `toksearch chat`
fdp query ...     # shim: setup_environment() + re-exec `toksearch query`
                  # (preserves existing flags byte-for-byte)
```

In `fdp chat`, the REPL prints each `run_python` invocation (code + thought) before
executing it, then prints the captured output, then the agent's text response. Slash
commands (`/help`, `/reset`, `/history`, `/namespace`, `/backend`, `/model`, `/save`,
`/quit`) operate on the live Session.

## Detailed design

### Events

All four event types are frozen dataclasses. The Session dispatches them to the
appropriate `on_<kind>` callback (and to `on_event` if provided).

```python
@dataclass(frozen=True)
class TextDelta:
    text: str

@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    args: dict
    thought: str | None      # populated for run_python; None for lookup_docs

@dataclass(frozen=True)
class ToolResult:
    id: str
    output: str
    is_error: bool

@dataclass(frozen=True)
class TurnComplete:
    stop_reason: Literal["end_turn", "max_iterations", "interrupted"]
    final_text: str
```

### History — `Message` and `ContentBlock`

Provider-neutral so the same shape works for Anthropic, OpenAI, and Claude SDK; also so
future on-disk persistence can serialize without provider-specific objects.

```python
@dataclass
class Message:
    role: Literal["user", "assistant"]
    content: list[ContentBlock]

# Tagged union; one variant per kind:
@dataclass
class TextBlock:        kind = "text";        text: str

@dataclass
class ToolUseBlock:     kind = "tool_use";    id: str; name: str; args: dict

@dataclass
class ToolResultBlock:  kind = "tool_result"; tool_use_id: str
                        output: str; is_error: bool
```

Backends translate to/from their native shapes on the boundary; the Session sees only
this union.

### Tools

Two `ToolSpec`s registered with the Session at construction. Both are
backend-agnostic; backends translate the `input_schema` to native form.

**`run_python`**

- Schema: `{"code": str (required), "thought": str (required)}`.
- Handler executes `exec(code, session.namespace)` with stdout/stderr captured.
- The namespace persists for the Session's lifetime; pre-populated with `toksearch`,
  `plt` (`matplotlib.pyplot`), `pd` (`pandas`), `np` (`numpy`), plus everything
  contributed via entry-point discovery (e.g., `toksearch_d3d`) and anything in
  `extra_namespace`.
- `KeyboardInterrupt` during execution: caught, returns `(interrupted)` as the tool
  result with `is_error=True` so the model can react.
- Other exceptions: traceback written to stderr; tool result has `is_error=True`.

**`lookup_docs`**

- Schema: `{"skill_name": str (required, enum populated from discovered skills)}`.
- Handler returns the SKILL.md body for the named skill.
- Skill discovery: union of `toksearch/skills/` (built-in) and every directory pointed
  to by a registered `toksearch.llm.skills` entry point, plus any `extra_skill_dirs`
  passed at Session construction.

### System prompt

Small kernel + dynamic catalog assembled at Session init:

```
You are a TokSearch expert. Use the run_python tool to execute code that fetches
and analyzes fusion data. The namespace persists across tool calls in this
session — variables from earlier calls are available in later ones.

You have access to fusion data via the following installed packages:
- toksearch (core): Pipeline, MdsSignal, ZarrSignal, fetch_dataset
- toksearch_d3d: DIII-D signals (PtDataSignal, ImasSignal), FDP/Pelican data access
  [...one line per contributor...]

Available documentation skills (call lookup_docs(skill_name=...) to read):
- toksearch-pipeline: core pipeline workflow
- toksearch-mds: MDSplus signal class
- toksearch-d3d-ptdata: DIII-D PTData signals
  [...one line per discovered skill...]

Rules:
- Do not include import statements; common modules are pre-imported.
- If code raises an error, read the traceback, fix it, and try again.
- When you have a result, store it in a variable named `result` or describe it
  in plain text. Do not call any tool to "finish" — just stop emitting tool calls.
```

Per-contributor descriptions come from a module-level `__llm_description__` attribute
on the package object resolved by the `toksearch.llm.namespace` entry point.

### Backends

#### `Backend` ABC

```python
class Backend(ABC):
    name: str
    default_model: str

    @abstractmethod
    def run_conversation(
        self,
        session: "Session",
        new_user_message: str,
        callbacks: Callbacks,
        max_iterations: int,
    ) -> TurnComplete: ...
```

The backend advances the conversation by one user message's worth of work, dispatching
callbacks along the way, and returns when the assistant has either stopped emitting
tool calls (`end_turn`), been interrupted, or hit `max_iterations`.

#### `_ToolLoopBackend` (Anthropic, OpenAI, AmSC)

Implements the standard loop:

```
1. Append the user message to history.
2. Loop up to max_iterations:
   a. Call self._send_request(system_prompt, history, tools, model, on_text)
      → returns AssistantTurn(blocks, stop_reason).
   b. Append the assistant turn to session.history.
   c. For each tool_use block:
      - Fire on_tool_call.
      - If confirm is set and returns False: return TurnComplete("interrupted").
      - session._execute_tool(block) → ToolOutput.
      - Fire on_tool_result.
      - Append a user-role tool_result message to history.
   d. If stop_reason != "tool_use": return TurnComplete(stop_reason, final_text).
3. Return TurnComplete("max_iterations", ...).
```

Subclasses implement only `_send_request()` and the translation to/from `ContentBlock`.

**`AnthropicBackend`**: uses `anthropic.Anthropic(api_key=..., base_url=...)`. Adds
`cache_control: ephemeral` to the system prompt block. Streams text deltas via
`client.messages.stream(...)`. Native tool/result schema maps to `ContentBlock`
near-identity.

**`OpenAIBackend`**: uses `openai.OpenAI(api_key=...)`. Translates our `ToolSpec` to
`{"type": "function", "function": {"name", "description", "parameters"}}`. Accumulates
streamed partial tool-arg JSON across chunks before emitting a single `ToolCall`.
Prepends `[error]\n` to tool results when `is_error=True` since OpenAI has no native
flag.

**AmSC**: not a separate class. Registered as a preset (see Presets below).

#### `ClaudeSDKBackend` (Claude Max plan)

Uses `claude_agent_sdk.ClaudeSDKClient` configured with `ClaudeAgentOptions`:

- `system_prompt`: the Session's system prompt (same content as raw backends).
- `mcp_servers`: an in-process MCP server (created via `create_sdk_mcp_server`) exposing
  `run_python` and `lookup_docs`. The server's tool handlers call back into the
  Session's namespace directly — no out-of-process MCP transport.
- `allowed_tools`: `["mcp__toksearch__run_python", "mcp__toksearch__lookup_docs"]`.
  Claude Code's built-in Read/Edit/Bash/Glob/Grep are **disabled** by default. This is
  deliberate — see Non-goals.
- `permission_mode`: `"bypassPermissions"`. We surface code via callbacks before the
  tool runs; the SDK's own permission prompts would be redundant.

The client is constructed lazily on first `send()` and reused across the Session's
lifetime. `connect()` failure (e.g. `claude` CLI not authenticated) raises `LLMAuthError`
with explicit remediation.

`run_conversation` calls `client.query(new_user_message)`, iterates messages from
`client.receive_response()`, translates each into a `ContentBlock` appended to
`session.history`, dispatches the matching callback, and returns `TurnComplete` when a
`ResultMessage` arrives.

### Backend selection

Precedence (highest first):

1. Explicit kwarg: `Session(backend="openai")` or `Session(backend=MyBackend(...))`.
2. CLI flag: `--backend openai`.
3. Env var: `FDP_LLM_BACKEND=openai`.
4. Config file: `~/.fdp/config.toml` → `[llm] backend = "openai"`.
5. Default: `"amsc"` (preserves the current `fdp query` behavior for existing users).

Backend names resolve through the **preset** layer first: a preset can override
`backend` (the implementation class), `base_url`, `model`, and `api_key_file`.

### Presets

```python
@dataclass(frozen=True)
class Preset:
    backend: str                # which Backend class to instantiate
    model: str | None = None
    base_url: str | None = None
    api_key_file: str | None = None
    api_key_env: str | None = None
    extra: dict | None = None   # backend-specific kwargs
```

Built-in presets in core toksearch:
- `anthropic` → `AnthropicBackend`, `model="claude-sonnet-4-6"`,
  `api_key_env="ANTHROPIC_API_KEY"`.
- `openai` → `OpenAIBackend`, `model="gpt-4o"`, `api_key_env="OPENAI_API_KEY"`.
- `claude-max` → `ClaudeSDKBackend`, no key.

Site presets via entry points (`toksearch.llm.presets`):
- `amsc` → `AnthropicBackend`, `model="claude-sonnet-4-6"`,
  `base_url="https://api.i2-core.american-science-cloud.org"`,
  `api_key_file="~/amsc_api_key"`. Registered by `toksearch_d3d`.

User presets via `~/.fdp/config.toml`:

```toml
[llm.presets.mysite]
backend = "anthropic"
base_url = "https://llm.mysite.gov"
api_key_env = "MYSITE_KEY"
model = "claude-sonnet-4-6"
```

### Entry-point discovery

Three groups, all consumed at Session construction:

- **`toksearch.llm.namespace`** — values are import strings resolving to a Python module
  or object. The Session injects them into the `run_python` namespace under the entry
  point's name. A module-level `__llm_description__` attribute, if present, is used in
  the system-prompt catalog.
- **`toksearch.llm.skills`** — values resolve to a `Path` (or callable returning a Path)
  to a directory containing `SKILL.md` subdirectories. Added to the skill registry.
- **`toksearch.llm.presets`** — values resolve to a `Preset` instance. Added to the
  preset registry.

Discovery happens once per process; results cached on the `discovery` module.
`Session(packages=[...])` restricts the namespace + skills entry points to the named
subset; presets are always loaded.

### Configuration

`~/.fdp/config.toml` (XDG-friendly fallback to `~/.config/fdp/config.toml`):

```toml
[llm]
backend = "amsc"               # default backend / preset name
model = "claude-sonnet-4-6"    # override preset's model
max_iterations = 20

# Optional credentials (env vars take precedence)
anthropic_api_key = "..."
openai_api_key = "..."

[llm.presets.mysite]
# user-defined presets, see above
```

Loaded by `toksearch.llm.config.load_config()` with env-var overlay then CLI-flag
overlay. No secrets are committed to the repo; the file lives in the user's home.

### Error handling

Custom hierarchy:

```
LLMError
├── LLMConfigError          # bad config, unknown backend, unknown model
├── LLMAuthError            # 401/403, missing key, claude CLI not authenticated
├── LLMBackendError         # 5xx, network failures after retries, MCP transport failure
├── LLMRateLimitError       # 429 after honoring Retry-After up to 60s
└── LLMUserAbort            # KeyboardInterrupt between tool calls
```

Tool exceptions are **not** errors at the Session/backend level — they're captured into
`is_error=True` tool results and returned to the model, matching current behavior.

`_ToolLoopBackend._send_request` retries 5xx and connection errors up to 3× with
exponential backoff (1s, 2s, 4s). Auth errors and 4xx (other than 429) do not retry.

REPL handles each exception class with a one-line user message; one-shot `query` exits
non-zero with the same message on stderr.

### Connectivity test

`toksearch llm test [--backend NAME]` sends a one-token request and asserts a response
is received. Cached in `~/.fdp/.last_check` per backend; `chat`/`query` runs it
automatically once per 24h. Failures print remediation and abort.

### CLI subprocess / re-exec

The current `fdp query` runs the agent in a subprocess to ensure libfdpio + XRootD see
FDP env vars at C-library load time. For `fdp chat` (long-running REPL with terminal
control), a piped subprocess is awkward.

Solution: `fdp` shims set the FDP env vars, set `FDP_LLM_ENV_READY=1`, then
`os.execvpe(sys.executable, [sys.executable, "-m", "toksearch.llm.cli", subcommand,
*args], os.environ)`. The new process starts fresh (C deps see env at load), keeps
direct terminal control (no pipe), and uses `FDP_LLM_ENV_READY` to detect that env is
already set on re-entry. `toksearch chat` invoked directly (no `fdp`) does no re-exec.

## Testing

Test seam: a `FakeBackend` that takes a scripted list of `AssistantTurn`s and returns
them in order. Lets Session-level tests run without any LLM.

```
tests/
├── test_llm_session.py          # Session behavior with FakeBackend
├── test_llm_tools.py            # run_python, lookup_docs
├── test_llm_discovery.py        # entry-point discovery, packages= filter
├── test_llm_presets.py          # preset resolution, env-var precedence
├── test_llm_backends/
│   ├── test_anthropic.py        # request shape, response translation, caching
│   ├── test_openai.py           # tool schema translation, tool-arg streaming
│   └── test_claude_sdk.py       # MCP server registration, tool plumbing
├── test_llm_cli.py              # subcommand dispatch, /commands, re-exec guard
└── test_llm_integration.py      # @pytest.mark.integration, real API
```

Existing `test_fdp_query.py` cases continue to pass — `fdp query` behavior is preserved.

## Migration plan

Five sequential PRs, none blocking earlier ones from shipping:

1. **`toksearch` PR 1 — library core.** Adds `toksearch/llm/` with Session, events,
   messages, tools, prompts, errors, config, `AnthropicBackend`, `OpenAIBackend`, CLI
   subcommands, `FakeBackend`, tests. No entry-point discovery yet — Session takes
   explicit `packages=` and `extra_skill_dirs=`. Adds `[llm]` optional-dep extra
   (`anthropic`, `openai`, `prompt-toolkit`). Installs `toksearch` console script.
2. **`toksearch` PR 2 — discovery.** Adds entry-point discovery for `namespace`,
   `skills`, and `presets`. Core registers itself as the `toksearch` namespace
   contributor; built-in presets (`anthropic`, `openai`). `Session(packages=...)`
   becomes an override.
3. **`toksearch` PR 3 — Claude Agent SDK backend.** Adds `claude_sdk.py` and the
   in-process MCP server. Adds optional dep on `claude-agent-sdk` + `mcp` (only required
   if the user selects the `claude-max` backend; importing `toksearch.llm` without the
   extras prints a clear hint).
4. **`toksearch_d3d` PR 4 — contributor wiring.** Declares the three entry points
   (`toksearch_d3d` namespace, `toksearch_d3d/skills` skill dir, `amsc` preset).
   Replaces `toksearch_d3d/agents/claude_toksearch_agent.py` with a deprecation shim
   that calls `toksearch.llm`. Updates `fdp.cli` to re-exec into `toksearch.llm.cli`
   for both `chat` and `query`. Bumps `toksearch>=<release>` accordingly.
5. **`toksearch_d3d` PR 5 — cleanup.** Removes the deprecation shim and `agents/`
   entirely after one release.

## Risks

- **Dependency surface in core toksearch.** Adding `anthropic`/`openai`/`claude-agent-sdk`
  as optional extras keeps the core install lean for users who don't want the LLM stack.
- **Entry-point discovery cost.** ~10ms at Session init on modern Python; not a concern.
- **Two-repo coordination.** PR ordering is strict; toksearch_d3d PR 4 cannot land until
  toksearch PRs 1–3 are released. Standard practice.
- **Naming collisions** between contributors are detected at Session init with a clear
  error message.
- **`exec()` blast radius.** Same as Claude Code's Bash tool: the agent can in principle
  read/write any file in your filesystem and hit any network endpoint. Documented; not
  sandboxed. Show-then-run mitigates accidental damage; `--confirm` is available for
  high-risk sessions.
- **Streaming complexity** (especially OpenAI tool-arg streaming). ~50 LoC of accumulator
  logic per backend. Worth it for REPL UX; not free.

## Out of scope (explicit non-features)

- Auto-save and resume of session history. The Message schema is designed to be
  serializable; future work will add `/load` and a session-store. `/save` is implemented
  now as a forcing function for the schema.
- Persistent namespace serialization across sessions. Many toksearch objects don't
  pickle reliably; design will need separate thought.
- Multimodal output in the REPL (figures inline). Future Jupyter wrapper.
- Subagents, hooks, the broader Claude Agent SDK feature set. If a user needs them,
  they should run `claude` directly.
- Sandboxing of `run_python`. Same posture as Claude Code's Bash tool.

## Open questions

- **Slash-command syntax in the REPL.** `/reset` vs `\reset` vs `:reset`. Defaulting to
  `/` for muscle memory with chat UIs; happy to revisit.
- **Default `max_iterations`.** Current is 10; this design proposes 20. Conversational
  flow tends to need more rounds than one-shot. May want to tune after first usage.
- **Whether to keep `--api-key-file` on `fdp query`** for backward compatibility, or
  deprecate in favor of `--backend amsc --config <file>`. Leaning toward keep-and-route
  internally.
