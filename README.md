![](ts_logo_blue.png)

# Welcome to TokSearch

TokSearch is a Python package for parallel retrieving, processing, and filtering of arbitrary-dimension fusion experimental data. TokSearch provides a high level API for extracting information from many shots, along with useful classes for low level data retrieval and manipulation.

The fundamental class in TokSearch is the ```Pipeline```. A ```Pipeline``` object takes a list of shots and, for each shot in the list, creates a dict-like object called a ```Record```. The ```Pipeline``` object then provides methods for defining a sequence of processing steps to apply to each record. These processing steps include:

- Passing user-defined functions to the pipeline via the ```map``` method.

or...

- Using a set of built-in methods, such as ```fetch```, ```fetch_dataset```, ```align```, ```keep```, or ```discard```.

The ```Pipeline``` also provides a ```where``` method which takes as input a user-defined function that returns a boolean value. If the function evaluates to ```False``` for a record, then that record is removed from the pipeline.


## Talk to your data

TokSearch packages its own know-how — how to build a pipeline, which signal
class a quantity needs, how to align and aggregate — as agent-readable skills.
There are two ways to use them, and both are first-class. Pick your agent
below — the built-in CLI or your own — then, either way, run it through `fdp`.

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

Both paths are best used from an environment installed with the `fdp-core`
metapackage (the standard FDP install; see [Installation](#installation)),
through the `fdp` wrappers — `fdp chat`,
`fdp query`, `fdp skills`. Those come with the device packages and configure
data access (XRootD transport, MDSplus tree paths, bearer token) before the
session starts, so the agent can actually reach shot data.

```bash
fdp chat                        # interactive, FDP environment configured
fdp query "Fetch ip for shot 200000 and report the peak in MA."
```

See the [LLM tutorial](https://ga-fdp.github.io/toksearch/latest/LLM_Tutorial/)
for an end-to-end walkthrough and the
[LLM Interface reference](https://ga-fdp.github.io/toksearch/latest/llm/) for
the full API surface.


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
| `--latest` | Depend on `fdp-core-latest` (`>=` floors, e.g. `toksearch >=2.11.1`) instead of `fdp-core` (exact `==` pins, e.g. `toksearch ==2.11.1`) |
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

