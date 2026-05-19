# LLM Interface

TokSearch ships with `toksearch.llm` — a conversational interface that lets
you ask for fusion data in plain English and have an LLM write the pipeline
code for you. The agent uses a **persistent Python namespace** so successive
turns iterate on cached results instead of re-fetching, which is what makes
multi-turn analysis viable when each fetch takes seconds to minutes.

```text
you> Fetch ip for shot 200000.
agent> [run_python]
         ip = PtDataSignal('ip').fetch(200000)
       [output] (no output)
       Done. The result is in `ip`.

you> What's the peak value in MA?
agent> [run_python]
         print(np.nanmax(np.abs(ip['data'])) / 1e6)
       [output] 1.4193  ← did NOT re-fetch
       Peak |Ip| is 1.42 MA.
```

## Backends

Four backend names ship in core TokSearch:

| Name | Provider | Credentials |
|---|---|---|
| `anthropic` | [Anthropic Messages API](https://docs.claude.com/en/api/overview) | `ANTHROPIC_API_KEY` |
| `openai` | [OpenAI Chat Completions](https://platform.openai.com/docs/api-reference) | `OPENAI_API_KEY` |
| `claude-max` | Claude Max plan via the [Claude Agent SDK](https://github.com/anthropics/claude-agent-sdk-python) | the `claude` CLI (run `claude login`) |
| `amsc` *(registered by `toksearch_d3d`)* | American Science Cloud (AmSC) Anthropic-compatible endpoint at `api.i2-core.american-science-cloud.org` | `~/amsc_api_key` |

Additional backends can be registered by any installed package — see
[Contributors](#contributors).

## Installation

```bash
# Conda (recommended; matches FDP-on-prem usage)
conda install -c ga-fdp toksearch         # backends bundled by default

# Pip
pip install toksearch[llm]
```

The conda recipe lists the four backend SDKs (`anthropic`, `openai`,
`claude-agent-sdk`, `mcp`) and `matplotlib` as hard run-dependencies; pip users
get the same surface via the `[llm]` optional-dependency extra.

## Quickstart

### CLI

```bash
# One-shot
toksearch query --backend anthropic "Use run_python to compute 2 + 2."

# Interactive REPL
toksearch chat --backend anthropic
```

The REPL accepts `/help`, `/reset`, and `/quit`; ctrl-D also exits. From a
DIII-D environment, the `fdp` script wraps the same commands with the
FDP environment pre-configured (XRootD plugin, MDSplus tree paths, etc.):

```bash
fdp query "Fetch ip for shot 200000 and report peak in MA."  # default: --backend amsc
fdp chat                                                       # interactive
```

### Python

```python
from toksearch.llm import Session
from toksearch.llm.backends.anthropic import AnthropicBackend

sess = Session(backend=AnthropicBackend(api_key="sk-..."))
result = sess.send(
    "Use run_python to compute 2 + 2.",
    on_tool_call=lambda c: print(f"[{c.name}] {c.thought}"),
    on_tool_result=lambda r: print(r.output),
)
print(result.final_text)
```

For end-to-end examples including the persistent-namespace pattern and a
real DIII-D workflow, see the [LLM Tutorial](LLM_Tutorial.ipynb).

## Configuration

Resolution precedence (highest first):

1. CLI flags: `--backend`, `--model`, `-n / --max-iterations`, `--package`
2. Environment variables: `FDP_LLM_BACKEND`, `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`
3. `~/.fdp/config.toml`:

    ```toml
    [llm]
    backend = "anthropic"          # or openai, claude-max, amsc, or a user preset name
    model = "claude-sonnet-4-6"    # overrides the backend's default
    max_iterations = 20
    anthropic_api_key = "sk-..."   # only if not using env var

    [llm.presets.mysite]
    backend = "anthropic"          # the underlying backend class
    base_url = "https://llm.mysite.gov"
    api_key_env = "MYSITE_KEY"
    model = "claude-sonnet-4-6"
    ```

4. Built-in defaults

## Tools

The agent has exactly two tools, registered with every Session:

### `run_python`

Executes a Python code string in the Session's persistent namespace. The
namespace lives for the Session's lifetime — variables defined in one turn
are available in all subsequent turns. Pre-populated with:

- `toksearch` (and `toksearch_d3d` if installed)
- `np` (`numpy`)
- `pd` (`pandas`, if available)
- `plt` (`matplotlib.pyplot`, if available)

The agent must populate a `thought` field with a one-sentence description of
what each code block does and why. This is what the REPL prints before
execution — it's the load-bearing transparency mechanism (see
[Show-then-run](#show-then-run)).

### `lookup_docs`

Returns the body of a registered skill (a `SKILL.md` file). The Session's
system prompt lists the available skill names with one-line descriptions; the
agent calls `lookup_docs(skill_name=...)` when it needs the details.

Core TokSearch ships with skills covering Pipeline basics, MdsSignal,
the backends, datasets, and API exploration. Device packages add their own:
`toksearch_d3d` contributes skills for `PtDataSignal`, `ImasSignal`, FDP CLI,
and a DIII-D quickstart.

## Show-then-run

The REPL prints each `run_python` block's `thought` and code **before**
executing it. This is the default UX — auto-approval with transparency, no
confirmation prompts. The realistic threat model isn't a malicious agent; it's
the agent making an expensive mistake (e.g. firing off a
`compute_multiprocessing` over 10,000 shots when you wanted 100). Surfacing
the code before execution lets you ctrl-C out before it commits to anything.

A `confirm=` callback is exposed on `Session.send` for callers who want to gate
each call programmatically:

```python
def review(call):
    print(call.args.get("code"))
    return input("Run this? [Y/n] ") != "n"

sess.send("...", confirm=review)
```

## Contributors

Any installed package can contribute three things to `toksearch.llm` via Python
entry points:

| Entry-point group | Value resolves to | Effect |
|---|---|---|
| `toksearch.llm.namespace` | a Python module / object | Bound under the entry-point name in the run_python namespace. A module-level `__llm_description__` populates the system-prompt catalog. |
| `toksearch.llm.skills` | a `Path` (or callable returning one) | Directory scanned for `SKILL.md` files. |
| `toksearch.llm.presets` | a `toksearch.llm.presets.Preset` instance | Added to the preset registry under the entry-point name. |

Example: how `toksearch_d3d` plugs in (in its `pyproject.toml`):

```toml
[project.entry-points."toksearch.llm.namespace"]
toksearch_d3d = "toksearch_d3d"

[project.entry-points."toksearch.llm.skills"]
toksearch_d3d = "toksearch_d3d.llm:skills_path"

[project.entry-points."toksearch.llm.presets"]
amsc = "toksearch_d3d.llm:AMSC_PRESET"
```

Sessions auto-discover all installed contributors on construction. Filter to a
subset with `Session(packages=["toksearch_d3d"])` or `toksearch chat --package
toksearch_d3d` (repeatable).

## API reference

::: toksearch.llm.Session
    handler: python
    options:
        show_root_heading: True
        members_order: source

::: toksearch.llm.Preset
    handler: python
    options:
        show_root_heading: True

::: toksearch.llm.Config
    handler: python
    options:
        show_root_heading: True

### Events

The four event types dispatched to `Session.send()` callbacks:

::: toksearch.llm.events.TextDelta
    handler: python
    options:
        show_root_heading: True

::: toksearch.llm.events.ToolCall
    handler: python
    options:
        show_root_heading: True

::: toksearch.llm.events.ToolResult
    handler: python
    options:
        show_root_heading: True

::: toksearch.llm.events.TurnComplete
    handler: python
    options:
        show_root_heading: True

### Exceptions

::: toksearch.llm.errors.LLMError
    handler: python
    options:
        show_root_heading: True
        members:
            - __init__

All exceptions raised by `toksearch.llm` inherit from `LLMError`. Specific
subclasses: `LLMConfigError`, `LLMAuthError`, `LLMBackendError`,
`LLMRateLimitError`, `LLMUserAbort` (the last also inherits from
`KeyboardInterrupt` so REPL frames can use a single `except KeyboardInterrupt`
handler).

## See also

- [LLM Tutorial](LLM_Tutorial.ipynb) — end-to-end walkthrough including a DIII-D plot.
- [`fdp` CLI](https://github.com/GA-FDP/toksearch_d3d) — wraps `toksearch chat`/`query` with FDP environment setup for DIII-D users.
- [Claude Agent SDK](https://github.com/anthropics/claude-agent-sdk-python) — underlies the `claude-max` backend.
