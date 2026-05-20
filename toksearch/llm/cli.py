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
"""Command-line interface for toksearch.llm.

Subcommands:
- ``toksearch query "<prompt>"`` — one-shot; runs a single turn and prints output.
- ``toksearch chat`` — interactive REPL (plain ``input()``; prompt_toolkit-based
  UI is deferred to a follow-up PR).

The CLI delegates everything substantive to ``Session``.  ``build_session(args)``
is the seam tests mock to avoid constructing a real backend.
"""

import argparse
import sys

from .config import load_config
from .errors import LLMConfigError, LLMError
from .presets import BUILTIN_PRESETS, resolve_preset
from .backends import get_backend_class
from .session import Session


def build_session(args) -> Session:
    """Construct a Session from parsed CLI args."""
    cfg = load_config()
    # Default backend is "anthropic" in PR 1; PR 4 will register the "amsc"
    # preset (via toksearch_d3d entry point) and the default will move to
    # "amsc" to preserve the existing fdp query behavior for GA-on-prem users.
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


def _resolve_api_key(preset, cfg) -> str | None:
    """Resolve the API key for a preset.

    Lookup order:
      1. ``preset.api_key_env``: read from os.environ.
      2. ``preset.api_key_file``: read from disk (~ expanded).
      3. Built-in preset hardcoded fallback (``cfg.anthropic_api_key`` for
         ``anthropic``, ``cfg.openai_api_key`` for ``openai``).
      4. None.
    """
    import os
    from pathlib import Path
    if preset.api_key_env:
        val = os.environ.get(preset.api_key_env)
        if val:
            return val
    if preset.api_key_file:
        path = Path(preset.api_key_file).expanduser()
        if path.exists():
            try:
                return path.read_text().strip()
            except OSError:
                pass
    if preset.backend == "claude-max":
        # ClaudeSDKBackend uses OAuth via the `claude` CLI; no API key.
        return None
    if preset.backend == "anthropic":
        return cfg.anthropic_api_key
    if preset.backend == "openai":
        return cfg.openai_api_key
    return None


# ----------------------------------------------------------------------
# `toksearch query`
# ----------------------------------------------------------------------

class _ToolPrinter:
    """Pretty-printer for tool calls and results.

    Compact mode (default): one-line summary per call -- the LLM's
    `thought` if present, otherwise a key arg fallback. Successful
    results print nothing; errors show the first line. Verbose mode
    (`-v`) prints the full code and full body, matching the previous
    behavior.
    """

    _ARG_FALLBACKS = ("skill_name", "name", "query")

    def __init__(self, verbose: bool):
        self.verbose = verbose

    def tool_call(self, call):
        detail = (call.thought or "").strip()
        if not detail:
            for key in self._ARG_FALLBACKS:
                value = call.args.get(key)
                if value:
                    detail = str(value)
                    break
        print(f"\n[{call.name}] {detail}".rstrip())
        if self.verbose:
            code = call.args.get("code")
            if code:
                for line in code.splitlines():
                    print(f"  {line}")

    def tool_result(self, result):
        body = result.output or ""
        if result.is_error:
            first = body.splitlines()[0] if body else "(empty)"
            print(f"  [error] {first}")
            if self.verbose and body:
                for line in body.splitlines()[1:]:
                    print(f"  {line}")
        elif self.verbose:
            if not body:
                print("  [output] (empty)")
            else:
                for line in body.splitlines():
                    print(f"  {line}")
        # Compact + success: print nothing -- the LLM's reply will
        # reflect what it learned from the tool.


def _print_session_header(command_name, session, *, file=sys.stderr):
    """One-line header announcing the resolved backend and model."""
    print(f"toksearch {command_name} (backend: {session.backend.name}, "
          f"model: {session.model})", file=file)


def do_query(args):
    session = build_session(args)
    printer = _ToolPrinter(verbose=args.verbose)
    _print_session_header("query", session)
    try:
        result = session.send(
            args.query,
            on_text=lambda t: print(t, end="", flush=True),
            on_tool_call=printer.tool_call,
            on_tool_result=printer.tool_result,
        )
    except LLMError as e:
        print(f"\nerror: {e}", file=sys.stderr)
        sys.exit(1)
    print()  # final newline
    sys.exit(0 if result.stop_reason == "end_turn" else 2)


# ----------------------------------------------------------------------
# `toksearch chat`
# ----------------------------------------------------------------------

_HELP_TEXT = """\
Slash commands:
  /help   show this help
  /reset  clear history and namespace
  /quit   exit the chat (ctrl-D also works)
"""


def do_chat(args):
    if getattr(args, "gui", False):
        from .gui import launch_gui
        launch_gui(args,
                    open_browser=getattr(args, "open_browser", True))
        return
    session = build_session(args)
    printer = _ToolPrinter(verbose=args.verbose)
    _print_session_header("chat", session, file=sys.stdout)
    print("Type /help for commands. Ctrl-D to exit.\n")
    while True:
        try:
            line = input("you> ").strip()
        except EOFError:
            print()
            return
        if not line:
            continue
        if line == "/quit":
            return
        if line == "/help":
            print(_HELP_TEXT)
            continue
        if line == "/reset":
            session.reset()
            print("(session cleared)")
            continue
        try:
            session.send(
                line,
                on_text=lambda t: print(t, end="", flush=True),
                on_tool_call=printer.tool_call,
                on_tool_result=printer.tool_result,
            )
        except LLMError as e:
            print(f"error: {e}", file=sys.stderr)
            continue
        print()  # spacer between turns


# ----------------------------------------------------------------------
# `toksearch backends`
# ----------------------------------------------------------------------

def do_backends(args):
    """Print the names that ``--backend`` will accept.

    Lists built-in presets, entry-point-discovered presets (e.g. amsc
    from toksearch_d3d), and user presets defined under
    ``[llm.presets.<name>]`` in ``~/.fdp/config.toml``. Marks the
    current default.
    """
    from .discovery import discover_presets

    cfg = load_config()
    discovered = discover_presets()
    user_presets = cfg.user_presets or {}

    default_name = cfg.backend or "anthropic"

    rows: list[tuple[str, str, str, str]] = []
    for name, preset in BUILTIN_PRESETS.items():
        rows.append((name, "built-in", preset.backend, preset.model or "-"))
    for name, preset in discovered.items():
        if name in BUILTIN_PRESETS:
            continue  # built-ins shadow discovered of the same name
        rows.append((name, "discovered", preset.backend, preset.model or "-"))
    for name, kv in user_presets.items():
        if name in BUILTIN_PRESETS or name in discovered:
            continue  # user overrides are merged onto the base; don't double-list
        rows.append((name, "user", str(kv.get("backend", "?")),
                     str(kv.get("model", "-"))))
    rows.sort()

    if not rows:
        print("No backend presets available.")
        return

    headers = ("name", "source", "backend", "model")
    widths = [max(len(h), max(len(r[i]) for r in rows)) for i, h in enumerate(headers)]
    fmt = "  ".join(f"{{:<{w}}}" for w in widths) + "  {}"
    print(fmt.format(*headers, ""))
    print("  ".join("-" * w for w in widths))
    for name, source, backend, model in rows:
        marker = "(default)" if name == default_name else ""
        print(fmt.format(name, source, backend, model, marker).rstrip())


# ----------------------------------------------------------------------
# main
# ----------------------------------------------------------------------

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
    p.add_argument("-v", "--verbose", action="store_true",
                   help="Show full tool-call code and tool-result bodies. "
                        "Default prints a one-line summary per call.")


def main(argv=None):
    parser = argparse.ArgumentParser(prog="toksearch")
    sub = parser.add_subparsers(dest="command", required=True)

    qp = sub.add_parser("query", help="One-shot natural-language query.")
    qp.add_argument("query", help="The prompt (quote it on the shell).")
    _add_common(qp)
    qp.set_defaults(func=do_query)

    cp = sub.add_parser("chat", help="Interactive conversation.")
    _add_common(cp)
    cp.add_argument("--gui", action="store_true",
                     help="Launch the local Gradio GUI instead of the "
                          "terminal REPL.")
    cp.add_argument("--no-browser", dest="open_browser",
                     action="store_false", default=True,
                     help="When --gui is set, do not open a browser tab.")
    cp.set_defaults(func=do_chat)

    bp = sub.add_parser(
        "backends",
        help="List the names that --backend accepts (built-in, discovered, "
             "and user presets).")
    bp.set_defaults(func=do_backends)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
