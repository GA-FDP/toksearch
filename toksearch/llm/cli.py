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
from .presets import resolve_preset
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
    if preset.backend == "anthropic":
        return cfg.anthropic_api_key
    if preset.backend == "openai":
        return cfg.openai_api_key
    return None


# ----------------------------------------------------------------------
# `toksearch query`
# ----------------------------------------------------------------------

def _print_tool_call(call):
    print(f"\n[{call.name}] {call.thought or ''}".rstrip())
    code = call.args.get("code")
    if code:
        for line in code.splitlines():
            print(f"  {line}")


def _print_tool_result(result):
    label = "[output]" if not result.is_error else "[error]"
    body = result.output or ""
    for line in body.splitlines():
        print(f"  {line}")
    if not body:
        print(f"{label} (empty)")


def do_query(args):
    session = build_session(args)
    try:
        result = session.send(
            args.query,
            on_text=lambda t: print(t, end="", flush=True),
            on_tool_call=_print_tool_call,
            on_tool_result=_print_tool_result,
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
    session = build_session(args)
    print(f"toksearch chat (backend: {session.backend.name}, "
          f"model: {session.model})")
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
                on_tool_call=_print_tool_call,
                on_tool_result=_print_tool_result,
            )
        except LLMError as e:
            print(f"error: {e}", file=sys.stderr)
            continue
        print()  # spacer between turns


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


def main(argv=None):
    parser = argparse.ArgumentParser(prog="toksearch")
    sub = parser.add_subparsers(dest="command", required=True)

    qp = sub.add_parser("query", help="One-shot natural-language query.")
    qp.add_argument("query", help="The prompt (quote it on the shell).")
    _add_common(qp)
    qp.set_defaults(func=do_query)

    cp = sub.add_parser("chat", help="Interactive conversation.")
    _add_common(cp)
    cp.set_defaults(func=do_chat)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
