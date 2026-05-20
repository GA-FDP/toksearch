# `toksearch.llm` Chat GUI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship a local Gradio-based chat GUI for `toksearch.llm` with a split-pane layout (chat on the left, interactive plot pane on the right), plotly-preferred figure capture for `run_python`, and `--gui` entry-point wiring on both `toksearch chat` and `fdp chat`.

**Architecture:** New `toksearch/llm/gui/` sub-package wraps the existing `Session`. Gradio handlers run on worker threads; a `queue.Queue` drains `(kind, payload)` events from `Session.send` callbacks plus matplotlib / plotly capture hooks; a pure `_apply_event` function folds events into UI state which is yielded back to the browser via Gradio's generator protocol. Same `build_session(args)` path as the CLI — no new agent code.

**Tech Stack:** Python 3.11, Gradio ≥4, plotly, kaleido, matplotlib, `unittest` + `unittest.mock`, pixi-managed env. Apache 2.0 license header on every new file (full 13-line form — see existing files in this repo).

**Reference spec:** `docs/superpowers/specs/2026-05-20-llm-chat-gui-design.md`

**Reference for plan style:** `docs/superpowers/plans/2026-05-18-toksearch-llm-pr1.md`

## Scope notes — what's deferred from the spec

- **Jupyter integration.** Called out as a follow-up project in the spec; not in this plan.
- **Per-tab multi-session.** One `Session` per Gradio server; multi-tab shares conversation. Spec lists this as v2.
- **Side-by-side plot pinning.** Single current view + history gallery only. Spec lists this as v2.
- **Persistence across server restarts.** Figures and chat live in memory; spec lists this as v2.
- **Selenium / browser-driven end-to-end tests.** Spec calls these out; v1 ships unit tests + a manual smoke test.

---

## File structure

| File | Action | Purpose |
|---|---|---|
| `toksearch/llm/gui/__init__.py` | Create | Public surface: `launch_gui(host, port, open_browser, args)`. |
| `toksearch/llm/gui/figure_capture.py` | Create | `wrap_run_python_handler(handler, on_figure)` (matplotlib) and `install_plotly_renderer(on_figure)` (plotly). |
| `toksearch/llm/gui/bridge.py` | Create | Typed events (`Text`, `ToolCall`, `ToolResult`, `Figure`, `Error`, `Done`), `_apply_event` pure folder, `run_turn` generator that bridges `Session.send` to Gradio yields. |
| `toksearch/llm/gui/app.py` | Create | `build_app(session)` returning `gr.Blocks` with chat pane, current-plot, history gallery, input, /reset and /clear-figures buttons. |
| `toksearch/llm/gui/__main__.py` | Create | `python -m toksearch.llm.gui` entry point — argparse + `launch_gui()`. |
| `toksearch/llm/cli.py` | Modify | Add `--gui` to the `chat` subparser; if set, `do_chat` delegates to `launch_gui(args)` instead of entering the REPL. |
| `toksearch/llm/tools.py` | Modify | Add `px` and `go` to `run_python` namespace; append one paragraph to `_RUN_PYTHON_DESCRIPTION` preferring plotly. |
| `recipe/recipe.yaml` | Modify | Add `gradio >=4`, `plotly`, and `python-kaleido` to conda run-deps. |
| `pixi.toml` | Modify | Same three deps for the dev env. |
| `tests/test_llm_gui_figure_capture.py` | Create | Unit tests for matplotlib + plotly capture. |
| `tests/test_llm_gui_bridge.py` | Create | Unit tests for `_apply_event` + integration test with fake `Session`. |
| `tests/test_llm_gui_app.py` | Create | Smoke test that `build_app(session)` returns a `gr.Blocks` with the expected named components. |
| `tests/test_llm_gui_launch.py` | Create | Verifies `launch_gui` calls `gr.Blocks.launch` with the right kwargs and that `toksearch chat --gui` routes there. |

The fdp side (`fdp chat --gui` flag) lives in the separate GA-FDP/fdp repo and is broken out into its own task at the end.

---

## Task 1: Skeleton + dependencies + plotly in run_python namespace

**Files:**
- Create: `toksearch/llm/gui/__init__.py`
- Create: `toksearch/llm/gui/figure_capture.py`
- Create: `toksearch/llm/gui/bridge.py`
- Create: `toksearch/llm/gui/app.py`
- Create: `toksearch/llm/gui/__main__.py`
- Modify: `toksearch/llm/tools.py:55-95` (description) and the `_run_python_handler` namespace setup
- Modify: `recipe/recipe.yaml` (run-deps section)
- Modify: `pixi.toml` (dependencies section)

- [ ] **Step 1: Create the package skeleton with license headers**

Each new file starts with the full 13-line Apache 2.0 header used elsewhere in the repo (copy from an existing file like `toksearch/llm/cli.py` if unsure). For each of the five new files, the *body* is a one-line module docstring describing the file's purpose, e.g.:

`toksearch/llm/gui/__init__.py`:
```python
# Copyright 2024 General Atomics
# (... 13-line Apache header ...)
"""Local Gradio chat GUI for toksearch.llm."""
```

The other four files get equivalent one-line docstrings. No code yet.

- [ ] **Step 2: Add plotly imports to the `run_python` namespace**

In `toksearch/llm/tools.py`, find the section that builds the persistent namespace (search for `"plt": matplotlib.pyplot`). Add two entries:

```python
import plotly.express as _px
import plotly.graph_objects as _go
# ...
namespace.update({
    "plt": matplotlib.pyplot,
    "px": _px,
    "go": _go,
    "pd": pandas,
    "np": numpy,
    "toksearch": toksearch,
})
```

Match the exact local-binding style already used for `plt` / `pd` / `np` — if those are imported at module top, do the same for `_px` and `_go`.

- [ ] **Step 3: Update the `run_python` tool description**

In `toksearch/llm/tools.py`, append a paragraph to `_RUN_PYTHON_DESCRIPTION`. After the existing text, add:

```python
_RUN_PYTHON_DESCRIPTION = (
    # ... existing description ...
    "\n\nPrefer plotly (`px`, `go`) for visualizations — the GUI "
    "renders them as interactive figures with zoom/pan/hover. Use "
    "matplotlib (`plt`) only for cases plotly handles poorly "
    "(animations, specialized scientific plots). Either way, call "
    "`fig.show()` or `plt.show()` to send the figure to the side "
    "pane in the GUI; in CLI mode the call is captured as text "
    "output."
)
```

- [ ] **Step 4: Add new deps to the conda recipe**

In `recipe/recipe.yaml`, add three entries under `requirements.run`, alphabetised among the existing ones:

```yaml
  run:
    - python
    # ... existing deps ...
    - gradio >=4
    - plotly
    - python-kaleido
```

- [ ] **Step 5: Add new deps to pixi.toml**

In `pixi.toml`, add to `[dependencies]`:

```toml
gradio = ">=4"
plotly = "*"
python-kaleido = "*"
```

- [ ] **Step 6: Re-lock and verify the env builds**

Run: `pixi install`
Expected: lock regenerates and exits cleanly. No further checks at this stage; tests come in later tasks.

- [ ] **Step 7: Commit**

```bash
git add toksearch/llm/gui/ toksearch/llm/tools.py recipe/recipe.yaml pixi.toml pixi.lock
git commit -m "Bootstrap toksearch.llm.gui package and add plotly to run_python

Empty subpackage with one-line docstrings, plus the supporting deps
(gradio, plotly, kaleido) and plotly imports in the run_python
namespace. Tool description now prefers plotly for visualizations.
"
```

---

## Task 2: matplotlib figure capture

**Files:**
- Create: `tests/test_llm_gui_figure_capture.py`
- Modify: `toksearch/llm/gui/figure_capture.py`

- [ ] **Step 1: Write the failing test for matplotlib capture**

In `tests/test_llm_gui_figure_capture.py`:

```python
# (Apache header)
"""Tests for toksearch.llm.gui.figure_capture."""

import unittest
from unittest import mock

import matplotlib
matplotlib.use("Agg")  # headless backend for CI


class TestMatplotlibCapture(unittest.TestCase):
    def test_wrap_emits_each_figure_then_closes(self):
        from toksearch.llm.gui.figure_capture import wrap_run_python_handler
        from toksearch.llm.tools import ToolOutput

        emitted = []
        on_figure = lambda kind, payload: emitted.append((kind, payload))

        # The inner handler is what the spec replaces -- here we fake
        # it to just create two matplotlib figures and return an OK
        # ToolOutput. We do NOT close the figures; the wrapper must.
        def inner(args, session):
            import matplotlib.pyplot as plt
            plt.figure()
            plt.figure()
            return ToolOutput(text="ok", is_error=False)

        wrapped = wrap_run_python_handler(inner, on_figure)
        result = wrapped({"code": ""}, mock.Mock())

        self.assertEqual(result.text, "ok")
        self.assertEqual(len(emitted), 2)
        for kind, fig in emitted:
            self.assertEqual(kind, "matplotlib")
            self.assertEqual(fig.__class__.__name__, "Figure")

        # After the call, pyplot's registry must be empty.
        import matplotlib.pyplot as plt
        self.assertEqual(plt.get_fignums(), [])
```

- [ ] **Step 2: Verify the test fails**

Run: `pixi run python -m unittest tests.test_llm_gui_figure_capture -v`
Expected: ImportError on `wrap_run_python_handler` (not yet defined).

- [ ] **Step 3: Implement matplotlib capture**

In `toksearch/llm/gui/figure_capture.py`, append below the docstring:

```python
from typing import Callable

OnFigure = Callable[[str, object], None]
"""Callback signature: ``on_figure(kind, payload)``.

``kind`` is the literal ``"matplotlib"`` or ``"plotly"``; payload is
either a ``matplotlib.figure.Figure`` or a ``dict`` (plotly's
fig_dict)."""


def wrap_run_python_handler(handler, on_figure: OnFigure):
    """Wrap a ``run_python`` handler to emit matplotlib figures.

    After the inner handler returns, every figure registered with
    ``pyplot.get_fignums()`` is delivered to ``on_figure`` and then
    closed. The inner handler's return value is passed through
    unchanged.
    """

    def wrapped(args, session):
        result = handler(args, session)
        import matplotlib.pyplot as plt
        for num in plt.get_fignums():
            on_figure("matplotlib", plt.figure(num))
        plt.close("all")
        return result

    return wrapped
```

- [ ] **Step 4: Verify the test passes**

Run: `pixi run python -m unittest tests.test_llm_gui_figure_capture -v`
Expected: PASS (1 test).

- [ ] **Step 5: Commit**

```bash
git add toksearch/llm/gui/figure_capture.py tests/test_llm_gui_figure_capture.py
git commit -m "Add matplotlib figure capture to GUI's run_python wrapper"
```

---

## Task 3: plotly figure capture via custom renderer

**Files:**
- Modify: `tests/test_llm_gui_figure_capture.py` (add a class)
- Modify: `toksearch/llm/gui/figure_capture.py`

- [ ] **Step 1: Write the failing test for plotly renderer**

Append to `tests/test_llm_gui_figure_capture.py`:

```python
class TestPlotlyCapture(unittest.TestCase):
    def test_install_renderer_routes_show_to_callback(self):
        import plotly.io as pio
        import plotly.graph_objects as go
        from toksearch.llm.gui.figure_capture import install_plotly_renderer

        emitted = []
        on_figure = lambda kind, payload: emitted.append((kind, payload))

        uninstall = install_plotly_renderer(on_figure)
        try:
            fig = go.Figure(data=[go.Scatter(x=[1, 2], y=[3, 4])])
            fig.show()  # should hit our renderer
        finally:
            uninstall()

        self.assertEqual(len(emitted), 1)
        kind, payload = emitted[0]
        self.assertEqual(kind, "plotly")
        # payload is plotly's fig_dict (a dict with 'data' and 'layout')
        self.assertIn("data", payload)
        self.assertIn("layout", payload)

    def test_uninstall_restores_previous_renderer(self):
        import plotly.io as pio
        from toksearch.llm.gui.figure_capture import install_plotly_renderer

        before = pio.renderers.default
        uninstall = install_plotly_renderer(lambda *a: None)
        self.assertNotEqual(pio.renderers.default, before)
        uninstall()
        self.assertEqual(pio.renderers.default, before)
```

- [ ] **Step 2: Verify the new tests fail**

Run: `pixi run python -m unittest tests.test_llm_gui_figure_capture -v`
Expected: ImportError on `install_plotly_renderer`.

- [ ] **Step 3: Implement plotly renderer**

Append to `toksearch/llm/gui/figure_capture.py`:

```python
_RENDERER_NAME = "toksearch_gui"


def install_plotly_renderer(on_figure: OnFigure):
    """Register a plotly renderer that fires ``on_figure("plotly", fig_dict)``.

    Returns a callable that restores the previous default renderer
    when invoked. Safe to call ``install_plotly_renderer`` multiple
    times — the latest callback wins.
    """
    import plotly.io as pio
    from plotly.io._base_renderers import ExternalRenderer

    class _GuiRenderer(ExternalRenderer):
        def render(self, fig_dict):
            on_figure("plotly", fig_dict)

    pio.renderers[_RENDERER_NAME] = _GuiRenderer()
    previous = pio.renderers.default
    pio.renderers.default = _RENDERER_NAME

    def uninstall():
        pio.renderers.default = previous

    return uninstall
```

- [ ] **Step 4: Verify all figure_capture tests pass**

Run: `pixi run python -m unittest tests.test_llm_gui_figure_capture -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add tests/test_llm_gui_figure_capture.py toksearch/llm/gui/figure_capture.py
git commit -m "Add plotly figure capture via custom external renderer"
```

---

## Task 4: bridge — event types and `_apply_event`

**Files:**
- Create: `tests/test_llm_gui_bridge.py`
- Modify: `toksearch/llm/gui/bridge.py`

- [ ] **Step 1: Write the failing test for the event types**

In `tests/test_llm_gui_bridge.py`:

```python
# (Apache header)
"""Tests for toksearch.llm.gui.bridge."""

import unittest
from unittest import mock


class TestEventTypes(unittest.TestCase):
    def test_event_types_are_frozen_dataclasses(self):
        from dataclasses import is_dataclass, FrozenInstanceError
        from toksearch.llm.gui import bridge

        for cls in (bridge.Text, bridge.ToolCall, bridge.ToolResult,
                    bridge.Figure, bridge.Error, bridge.Done):
            self.assertTrue(is_dataclass(cls))

        ev = bridge.Text(text="hi")
        with self.assertRaises(FrozenInstanceError):
            ev.text = "no"
```

- [ ] **Step 2: Verify the test fails**

Run: `pixi run python -m unittest tests.test_llm_gui_bridge -v`
Expected: ImportError on `bridge.Text`.

- [ ] **Step 3: Implement the event types**

Append to `toksearch/llm/gui/bridge.py`:

```python
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Text:
    text: str


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    thought: str | None
    args: dict


@dataclass(frozen=True)
class ToolResult:
    id: str
    output: str
    is_error: bool


@dataclass(frozen=True)
class Figure:
    kind: str        # "matplotlib" | "plotly"
    payload: Any     # matplotlib.figure.Figure or fig_dict


@dataclass(frozen=True)
class Error:
    message: str
    traceback: str


@dataclass(frozen=True)
class Done:
    pass
```

- [ ] **Step 4: Verify the event test passes**

Run: `pixi run python -m unittest tests.test_llm_gui_bridge.TestEventTypes -v`
Expected: PASS.

- [ ] **Step 5: Write the failing test for `_apply_event`**

Append to `tests/test_llm_gui_bridge.py`:

```python
class TestApplyEvent(unittest.TestCase):
    def _empty(self):
        from toksearch.llm.gui.bridge import ChatState, FigState
        return ChatState(messages=[], pending_calls={}), FigState(figures=[])

    def test_text_appends_to_last_assistant_message(self):
        from toksearch.llm.gui.bridge import _apply_event, Text
        chat, figs = self._empty()
        chat, figs = _apply_event(chat, figs, Text(text="Hello "))
        chat, figs = _apply_event(chat, figs, Text(text="world"))
        self.assertEqual(chat.messages[-1]["role"], "assistant")
        self.assertEqual(chat.messages[-1]["content"], "Hello world")

    def test_tool_call_then_result_produces_one_details_block(self):
        from toksearch.llm.gui.bridge import _apply_event, ToolCall, ToolResult
        chat, figs = self._empty()
        chat, figs = _apply_event(chat, figs, ToolCall(
            id="t1", name="run_python", thought="Fetch shots", args={}))
        chat, figs = _apply_event(chat, figs, ToolResult(
            id="t1", output="ok", is_error=False))
        body = chat.messages[-1]["content"]
        self.assertIn("<details>", body)
        self.assertIn("[run_python]", body)
        self.assertIn("Fetch shots", body)
        self.assertIn("ok", body)

    def test_tool_call_with_no_thought_falls_back_to_skill_name(self):
        from toksearch.llm.gui.bridge import _apply_event, ToolCall, ToolResult
        chat, figs = self._empty()
        chat, figs = _apply_event(chat, figs, ToolCall(
            id="t1", name="lookup_docs", thought=None,
            args={"skill_name": "toksearch-quickstart"}))
        chat, figs = _apply_event(chat, figs, ToolResult(
            id="t1", output="...", is_error=False))
        self.assertIn("toksearch-quickstart", chat.messages[-1]["content"])

    def test_figure_appends_and_current_tracks_latest(self):
        from toksearch.llm.gui.bridge import _apply_event, Figure
        chat, figs = self._empty()
        chat, figs = _apply_event(chat, figs,
            Figure(kind="plotly", payload={"data": [], "layout": {}}))
        chat, figs = _apply_event(chat, figs,
            Figure(kind="matplotlib", payload="<<mpl fig stub>>"))
        self.assertEqual(len(figs.figures), 2)
        self.assertEqual(figs.current_index, 1)  # newest

    def test_error_appends_red_assistant_message(self):
        from toksearch.llm.gui.bridge import _apply_event, Error
        chat, figs = self._empty()
        chat, figs = _apply_event(chat, figs, Error(
            message="boom", traceback="Traceback..."))
        body = chat.messages[-1]["content"]
        self.assertIn("boom", body)
        self.assertIn("Traceback", body)
```

- [ ] **Step 6: Verify the new tests fail**

Run: `pixi run python -m unittest tests.test_llm_gui_bridge -v`
Expected: ImportError on `ChatState` / `FigState` / `_apply_event`.

- [ ] **Step 7: Implement `ChatState`, `FigState`, and `_apply_event`**

Append to `toksearch/llm/gui/bridge.py`:

```python
from dataclasses import dataclass, field, replace
from typing import Any


@dataclass
class ChatState:
    messages: list[dict]               # [{"role": ..., "content": ...}, ...]
    pending_calls: dict[str, ToolCall] = field(default_factory=dict)


@dataclass
class FigState:
    figures: list[tuple[str, Any]]     # [(kind, payload), ...]
    current_index: int = -1


_ARG_FALLBACKS = ("skill_name", "name", "query")


def _tool_summary(call: ToolCall) -> str:
    detail = (call.thought or "").strip()
    if not detail:
        for key in _ARG_FALLBACKS:
            value = call.args.get(key)
            if value:
                detail = str(value)
                break
    return f"[{call.name}] {detail}".rstrip()


def _ensure_assistant_message(chat: ChatState) -> None:
    if not chat.messages or chat.messages[-1]["role"] != "assistant":
        chat.messages.append({"role": "assistant", "content": ""})


def _apply_event(chat: ChatState, figs: FigState, event):
    """Pure folder: returns the new (chat, fig_state) tuple.

    Mutates the dataclass instances in place for simplicity; callers
    that want immutability can deepcopy before calling.
    """
    if isinstance(event, Text):
        _ensure_assistant_message(chat)
        chat.messages[-1]["content"] += event.text
    elif isinstance(event, ToolCall):
        chat.pending_calls[event.id] = event
    elif isinstance(event, ToolResult):
        call = chat.pending_calls.pop(event.id, None)
        summary = _tool_summary(call) if call else f"[tool {event.id}]"
        body_label = "error" if event.is_error else "output"
        _ensure_assistant_message(chat)
        chat.messages[-1]["content"] += (
            f"\n\n<details><summary>▶ {summary}</summary>\n\n"
            f"```\n{event.output}\n```\n"
            f"</details>\n"
        )
    elif isinstance(event, Figure):
        figs.figures.append((event.kind, event.payload))
        figs.current_index = len(figs.figures) - 1
    elif isinstance(event, Error):
        chat.messages.append({
            "role": "assistant",
            "content": (
                f"⛔ **error:** {event.message}\n\n"
                f"<details><summary>traceback</summary>\n\n"
                f"```\n{event.traceback}\n```\n"
                f"</details>\n"
            ),
        })
    elif isinstance(event, Done):
        pass
    return chat, figs
```

- [ ] **Step 8: Verify all bridge tests pass**

Run: `pixi run python -m unittest tests.test_llm_gui_bridge -v`
Expected: PASS (6 tests).

- [ ] **Step 9: Commit**

```bash
git add toksearch/llm/gui/bridge.py tests/test_llm_gui_bridge.py
git commit -m "Add bridge event types and pure _apply_event folder"
```

---

## Task 5: bridge — `run_turn` generator with worker thread

**Files:**
- Modify: `tests/test_llm_gui_bridge.py`
- Modify: `toksearch/llm/gui/bridge.py`

- [ ] **Step 1: Write the failing test for `run_turn`**

Append to `tests/test_llm_gui_bridge.py`:

```python
class TestRunTurn(unittest.TestCase):
    def test_run_turn_streams_events_in_order(self):
        from toksearch.llm.gui.bridge import run_turn

        # Fake session whose .send() fires callbacks then returns.
        class FakeSession:
            def send(self, prompt, *, on_text, on_tool_call,
                     on_tool_result, **_):
                on_text("Hello ")
                on_text("world")

        session = FakeSession()
        outputs = list(run_turn(session, "hi"))

        # Expect at least one yield per event plus a terminal yield.
        # Last yield's chat state must contain "Hello world".
        last_chat, last_current, last_gallery = outputs[-1]
        self.assertEqual(last_chat.messages[-1]["content"], "Hello world")

    def test_run_turn_surfaces_send_exception_as_error_event(self):
        from toksearch.llm.gui.bridge import run_turn

        class BoomSession:
            def send(self, prompt, **_):
                raise RuntimeError("boom")

        session = BoomSession()
        outputs = list(run_turn(session, "hi"))
        last_chat, _, _ = outputs[-1]
        self.assertIn("boom", last_chat.messages[-1]["content"])
```

- [ ] **Step 2: Verify the test fails**

Run: `pixi run python -m unittest tests.test_llm_gui_bridge.TestRunTurn -v`
Expected: ImportError on `run_turn`.

- [ ] **Step 3: Implement `run_turn`**

Append to `toksearch/llm/gui/bridge.py`:

```python
import queue
import threading
import traceback as _tb


def run_turn(session, prompt: str,
              on_figure_install=None,
              chat: ChatState | None = None,
              figs: FigState | None = None):
    """Generator that drives one Session.send() turn.

    Yields ``(chat, current_figure, gallery_figures)`` tuples whenever
    the UI state should refresh. Designed to be passed directly into
    a Gradio handler.

    ``on_figure_install`` is an optional hook letting callers install
    plotly / matplotlib capture *bound to this turn's queue*; called
    with the queue's ``put`` function before the worker starts.
    """
    if chat is None:
        chat = ChatState(messages=[])
    if figs is None:
        figs = FigState(figures=[])

    # Prime an assistant slot so streaming text has somewhere to land.
    chat.messages.append({"role": "user", "content": prompt})

    q: "queue.Queue" = queue.Queue()

    def emit(event):
        q.put(event)

    if on_figure_install is not None:
        on_figure_install(lambda kind, payload:
                          emit(Figure(kind=kind, payload=payload)))

    def runner():
        try:
            session.send(
                prompt,
                on_text=lambda t: emit(Text(text=t)),
                on_tool_call=lambda c: emit(ToolCall(
                    id=c.id, name=c.name, thought=c.thought,
                    args=dict(c.args))),
                on_tool_result=lambda r: emit(ToolResult(
                    id=r.id, output=r.output, is_error=r.is_error)),
            )
        except Exception as e:
            emit(Error(message=str(e), traceback=_tb.format_exc()))
        finally:
            emit(Done())

    threading.Thread(target=runner, daemon=True).start()

    while True:
        event = q.get()
        if isinstance(event, Done):
            yield chat, _current(figs), _gallery(figs)
            return
        chat, figs = _apply_event(chat, figs, event)
        yield chat, _current(figs), _gallery(figs)


def _current(figs: FigState):
    if figs.current_index < 0:
        return None
    return figs.figures[figs.current_index][1]


def _gallery(figs: FigState):
    # Returns a list of payloads in chronological order for the
    # Gradio gallery. The app layer is responsible for converting
    # each payload to a thumbnail.
    return [payload for (_kind, payload) in figs.figures]
```

- [ ] **Step 4: Verify `run_turn` tests pass**

Run: `pixi run python -m unittest tests.test_llm_gui_bridge -v`
Expected: PASS (8 tests).

- [ ] **Step 5: Commit**

```bash
git add toksearch/llm/gui/bridge.py tests/test_llm_gui_bridge.py
git commit -m "Add bridge.run_turn generator with worker-thread queue drain"
```

---

## Task 6: app — `build_app` returns a Gradio Blocks layout

**Files:**
- Create: `tests/test_llm_gui_app.py`
- Modify: `toksearch/llm/gui/app.py`

- [ ] **Step 1: Write the failing test for component shape**

In `tests/test_llm_gui_app.py`:

```python
# (Apache header)
"""Smoke tests for toksearch.llm.gui.app."""

import unittest
from unittest import mock


class TestBuildApp(unittest.TestCase):
    def test_build_app_returns_blocks_with_expected_components(self):
        from toksearch.llm.gui.app import build_app

        fake_session = mock.Mock()
        blocks = build_app(fake_session)

        # The Blocks instance exposes its children via `.blocks` (a
        # dict mapping component ids to components). We assert by
        # label on a flat walk of the children.
        labels = []
        for child in _walk(blocks):
            label = getattr(child, "label", None)
            if label:
                labels.append(label)

        for required in ("Chat", "Current plot", "History",
                          "Your message", "Reset", "Clear figures"):
            self.assertIn(required, labels,
                          f"missing component with label {required!r}; "
                          f"found {labels}")


def _walk(node):
    """Yield Block + descendants recursively."""
    yield node
    children = getattr(node, "children", None) or []
    for child in children:
        yield from _walk(child)
```

- [ ] **Step 2: Verify the test fails**

Run: `pixi run python -m unittest tests.test_llm_gui_app -v`
Expected: ImportError on `build_app`.

- [ ] **Step 3: Implement `build_app`**

Append to `toksearch/llm/gui/app.py`:

```python
import gradio as gr


def build_app(session, on_submit=None, on_reset=None,
               on_clear_figs=None):
    """Construct the chat GUI Blocks app for a given Session.

    ``on_submit(user_text, chat_state, fig_state) -> generator`` is
    the handler invoked when the user sends a message. If omitted, a
    default handler delegates to :func:`bridge.run_turn`.

    ``on_reset`` and ``on_clear_figs`` are zero-arg callbacks for the
    matching buttons; defaults call ``session.reset()`` / clear the
    in-memory figure list.
    """
    from .bridge import ChatState, FigState, run_turn

    chat_state = gr.State(ChatState(messages=[]))
    fig_state = gr.State(FigState(figures=[]))

    with gr.Blocks(title="toksearch chat") as blocks:
        with gr.Row():
            with gr.Column(scale=2):
                chatbot = gr.Chatbot(label="Chat", height=600,
                                      type="messages")
                user_input = gr.Textbox(label="Your message",
                                          placeholder="Ask a question...",
                                          lines=2)
                with gr.Row():
                    reset_btn = gr.Button("Reset", variant="secondary")
            with gr.Column(scale=1):
                current_plot = gr.Plot(label="Current plot")
                gallery = gr.Gallery(label="History", columns=4,
                                       allow_preview=False)
                clear_btn = gr.Button("Clear figures",
                                        variant="secondary")

        def _default_submit(user_text, chat_st, fig_st):
            if not user_text.strip():
                yield chat_st.messages, None, []
                return
            for chat_st, current, gal in run_turn(session, user_text,
                                                  chat=chat_st,
                                                  figs=fig_st):
                yield chat_st.messages, current, gal

        submit = on_submit or _default_submit
        user_input.submit(
            submit,
            inputs=[user_input, chat_state, fig_state],
            outputs=[chatbot, current_plot, gallery],
        ).then(lambda: "", outputs=user_input)

        def _default_reset():
            session.reset()
            return ChatState(messages=[]), FigState(figures=[]), [], None, []

        reset_handler = on_reset or _default_reset
        reset_btn.click(
            reset_handler,
            outputs=[chat_state, fig_state, chatbot, current_plot, gallery],
        )

        def _default_clear():
            return FigState(figures=[]), None, []

        clear_handler = on_clear_figs or _default_clear
        clear_btn.click(
            clear_handler,
            outputs=[fig_state, current_plot, gallery],
        )

    return blocks
```

- [ ] **Step 4: Verify the smoke test passes**

Run: `pixi run python -m unittest tests.test_llm_gui_app -v`
Expected: PASS (1 test).

- [ ] **Step 5: Commit**

```bash
git add toksearch/llm/gui/app.py tests/test_llm_gui_app.py
git commit -m "Add build_app: split-pane Gradio Blocks for chat + plots"
```

---

## Task 7: `launch_gui` public entry point

**Files:**
- Create: `tests/test_llm_gui_launch.py`
- Modify: `toksearch/llm/gui/__init__.py`
- Modify: `toksearch/llm/gui/__main__.py`

- [ ] **Step 1: Write the failing test**

In `tests/test_llm_gui_launch.py`:

```python
# (Apache header)
"""Tests for the launch_gui() entry point and __main__."""

import unittest
from unittest import mock


class TestLaunchGui(unittest.TestCase):
    def test_launch_gui_builds_session_and_launches_blocks(self):
        from toksearch.llm import gui

        fake_session = mock.Mock()
        fake_blocks = mock.MagicMock()
        with mock.patch.object(gui, "build_session",
                                return_value=fake_session), \
             mock.patch.object(gui, "build_app",
                                return_value=fake_blocks):
            gui.launch_gui(args=mock.Mock(),
                            host="127.0.0.1",
                            port=12345,
                            open_browser=False)
        fake_blocks.launch.assert_called_once()
        kwargs = fake_blocks.launch.call_args.kwargs
        self.assertEqual(kwargs.get("server_name"), "127.0.0.1")
        self.assertEqual(kwargs.get("server_port"), 12345)
        self.assertEqual(kwargs.get("inbrowser"), False)
        self.assertEqual(kwargs.get("share"), False)


class TestMain(unittest.TestCase):
    def test_main_module_calls_launch_gui(self):
        from toksearch.llm.gui import __main__ as main_mod

        with mock.patch.object(main_mod, "launch_gui") as launch:
            main_mod.main(["--port", "9999", "--no-browser"])
        launch.assert_called_once()
        kwargs = launch.call_args.kwargs
        self.assertEqual(kwargs.get("port"), 9999)
        self.assertEqual(kwargs.get("open_browser"), False)
```

- [ ] **Step 2: Verify the test fails**

Run: `pixi run python -m unittest tests.test_llm_gui_launch -v`
Expected: AttributeError on `gui.launch_gui`.

- [ ] **Step 3: Implement `launch_gui` in `toksearch/llm/gui/__init__.py`**

Replace the contents of `toksearch/llm/gui/__init__.py` (keeping the license header) with:

```python
"""Local Gradio chat GUI for toksearch.llm."""

from ..cli import build_session
from .app import build_app


def launch_gui(args, *, host: str = "127.0.0.1",
                port: int | None = None,
                open_browser: bool = True) -> None:
    """Build a Session from CLI args and launch the Gradio app.

    ``args`` is the parsed argparse namespace from ``toksearch chat``
    (or an equivalent shape: must expose ``backend``, ``model``,
    ``max_iterations``, ``packages`` attributes).
    """
    session = build_session(args)
    blocks = build_app(session)
    blocks.launch(
        server_name=host,
        server_port=port,
        inbrowser=open_browser,
        share=False,
    )


__all__ = ["launch_gui"]
```

- [ ] **Step 4: Implement `__main__.py`**

Replace the contents of `toksearch/llm/gui/__main__.py` (keeping the license header) with:

```python
"""Entry point: ``python -m toksearch.llm.gui``."""

import argparse

from . import launch_gui


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m toksearch.llm.gui",
        description="Local Gradio chat GUI for toksearch.llm.",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--no-browser", dest="open_browser",
                         action="store_false", default=True)
    parser.add_argument("--backend", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("-n", "--max-iterations", type=int, default=None)
    parser.add_argument("--package", dest="packages", action="append",
                         default=None)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)
    launch_gui(args,
                host=args.host,
                port=args.port,
                open_browser=args.open_browser)


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Verify launch tests pass**

Run: `pixi run python -m unittest tests.test_llm_gui_launch -v`
Expected: PASS (2 tests).

- [ ] **Step 6: Commit**

```bash
git add toksearch/llm/gui/__init__.py toksearch/llm/gui/__main__.py \
        tests/test_llm_gui_launch.py
git commit -m "Add launch_gui() and python -m toksearch.llm.gui entry point"
```

---

## Task 8: wire `figure_capture` into `launch_gui`

**Files:**
- Modify: `tests/test_llm_gui_launch.py`
- Modify: `toksearch/llm/gui/__init__.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_llm_gui_launch.py`:

```python
class TestFigureCaptureWiring(unittest.TestCase):
    def test_launch_gui_wraps_run_python_handler_and_installs_plotly(self):
        from toksearch.llm import gui
        from toksearch.llm.gui import figure_capture

        fake_session = mock.Mock()
        # Pretend the session has the standard tool registry.
        fake_session._tools_by_name = {"run_python": mock.Mock(handler=mock.Mock())}
        fake_blocks = mock.MagicMock()

        with mock.patch.object(gui, "build_session",
                                return_value=fake_session), \
             mock.patch.object(gui, "build_app",
                                return_value=fake_blocks), \
             mock.patch.object(figure_capture,
                                "wrap_run_python_handler") as wrap, \
             mock.patch.object(figure_capture,
                                "install_plotly_renderer") as install:
            gui.launch_gui(args=mock.Mock(), open_browser=False)
        wrap.assert_called_once()
        install.assert_called_once()
```

- [ ] **Step 2: Verify the test fails**

Run: `pixi run python -m unittest tests.test_llm_gui_launch.TestFigureCaptureWiring -v`
Expected: FAIL (`wrap.assert_called_once()` — capture isn't wired yet).

- [ ] **Step 3: Wire capture in `launch_gui`**

Update `toksearch/llm/gui/__init__.py`:

```python
"""Local Gradio chat GUI for toksearch.llm."""

import queue

from ..cli import build_session
from .app import build_app
from . import figure_capture


def launch_gui(args, *, host: str = "127.0.0.1",
                port: int | None = None,
                open_browser: bool = True) -> None:
    """Build a Session from CLI args and launch the Gradio app.

    Wraps the session's ``run_python`` handler so matplotlib figures
    are auto-captured into the side pane, and installs a custom plotly
    renderer that routes ``fig.show()`` calls into the same channel.
    Both captures push events onto a per-turn ``queue.Queue`` provided
    by the bridge.
    """
    session = build_session(args)

    # The per-turn queue is owned by run_turn; here we install a thin
    # indirection so figure_capture can publish to whichever queue is
    # currently active. The bridge swaps the active queue at the start
    # of each turn via _set_active_figure_emitter().
    from .bridge import _set_active_figure_emitter
    _emit_holder = {"emit": lambda kind, payload: None}
    _set_active_figure_emitter(
        lambda kind, payload: _emit_holder["emit"](kind, payload))

    run_python_spec = session._tools_by_name["run_python"]
    original = run_python_spec.handler

    def on_figure(kind, payload):
        _emit_holder["emit"](kind, payload)

    run_python_spec.handler = figure_capture.wrap_run_python_handler(
        original, on_figure)
    figure_capture.install_plotly_renderer(on_figure)

    blocks = build_app(session)
    blocks.launch(
        server_name=host,
        server_port=port,
        inbrowser=open_browser,
        share=False,
    )


__all__ = ["launch_gui"]
```

- [ ] **Step 4: Add `_set_active_figure_emitter` to `bridge.py`**

In `toksearch/llm/gui/bridge.py`, define near the top (after the dataclasses):

```python
_active_figure_emitter = None


def _set_active_figure_emitter(emit_fn):
    """Install the function figure-capture hooks should call.

    ``run_turn`` swaps this in at the start of each turn and restores
    it afterward; the GUI entry point installs a default that drops
    events on the floor (so capture before any turn is harmless).
    """
    global _active_figure_emitter
    _active_figure_emitter = emit_fn
```

And update `run_turn` to swap it:

```python
def run_turn(session, prompt, chat=None, figs=None):
    ...  # body as before through queue creation

    def emit(event):
        q.put(event)

    previous = _active_figure_emitter
    _set_active_figure_emitter(
        lambda kind, payload: emit(Figure(kind=kind, payload=payload)))

    def runner():
        try:
            session.send(... )
        ...
        finally:
            emit(Done())

    threading.Thread(target=runner, daemon=True).start()

    try:
        while True:
            event = q.get()
            if isinstance(event, Done):
                yield chat, _current(figs), _gallery(figs)
                return
            chat, figs = _apply_event(chat, figs, event)
            yield chat, _current(figs), _gallery(figs)
    finally:
        _set_active_figure_emitter(previous)
```

(Replace the existing `run_turn` body keeping its previous logic; only the active-emitter swap is new.)

- [ ] **Step 5: Update `launch_gui` to use `_active_figure_emitter` directly**

Simplify `launch_gui`'s capture wiring — the holder pattern is no longer needed because the bridge owns the active emitter:

```python
    run_python_spec = session._tools_by_name["run_python"]
    original = run_python_spec.handler

    def on_figure(kind, payload):
        from .bridge import _active_figure_emitter
        if _active_figure_emitter is not None:
            _active_figure_emitter(kind, payload)

    run_python_spec.handler = figure_capture.wrap_run_python_handler(
        original, on_figure)
    figure_capture.install_plotly_renderer(on_figure)
```

(Drop the `_set_active_figure_emitter(lambda: None)` initial install — `_active_figure_emitter` is module-global and `None` by default; the `if` guard handles the pre-turn case.)

- [ ] **Step 6: Verify wiring tests pass**

Run: `pixi run python -m unittest tests.test_llm_gui_launch -v`
Expected: PASS (3 tests).

Also re-run bridge tests to ensure the active-emitter changes didn't regress:

Run: `pixi run python -m unittest tests.test_llm_gui_bridge -v`
Expected: PASS (8 tests).

- [ ] **Step 7: Commit**

```bash
git add toksearch/llm/gui/__init__.py toksearch/llm/gui/bridge.py \
        tests/test_llm_gui_launch.py
git commit -m "Wire figure capture through the bridge's active emitter"
```

---

## Task 9: `toksearch chat --gui` flag

**Files:**
- Modify: `tests/test_llm_cli.py`
- Modify: `toksearch/llm/cli.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_llm_cli.py`:

```python
class TestCliChatGuiFlag(unittest.TestCase):
    def test_chat_gui_flag_delegates_to_launch_gui(self):
        from toksearch.llm import cli

        with mock.patch.object(sys, "argv",
                                ["toksearch", "chat", "--gui"]), \
             mock.patch("toksearch.llm.gui.launch_gui") as launch:
            try:
                cli.main()
            except SystemExit:
                pass
        launch.assert_called_once()
```

- [ ] **Step 2: Verify the test fails**

Run: `pixi run python -m unittest tests.test_llm_cli.TestCliChatGuiFlag -v`
Expected: FAIL (unknown `--gui` flag).

- [ ] **Step 3: Add `--gui` to the `chat` subparser**

In `toksearch/llm/cli.py`, locate the `chat` subparser registration (search for `sub.add_parser("chat"`). Add:

```python
    cp = sub.add_parser("chat", help="Interactive conversation.")
    _add_common(cp)
    cp.add_argument("--gui", action="store_true",
                     help="Launch the local Gradio GUI instead of the "
                          "terminal REPL.")
    cp.add_argument("--no-browser", dest="open_browser",
                     action="store_false", default=True,
                     help="When --gui is set, do not open a browser tab.")
    cp.set_defaults(func=do_chat)
```

Then update `do_chat` to delegate when `--gui` is set. Near the top of `do_chat`:

```python
def do_chat(args):
    if getattr(args, "gui", False):
        from .gui import launch_gui
        launch_gui(args, open_browser=getattr(args, "open_browser", True))
        return
    # ... existing REPL code unchanged ...
```

- [ ] **Step 4: Verify the test passes**

Run: `pixi run python -m unittest tests.test_llm_cli.TestCliChatGuiFlag -v`
Expected: PASS.

- [ ] **Step 5: Re-run the full CLI test suite to catch regressions**

Run: `pixi run python -m unittest tests.test_llm_cli -v`
Expected: PASS (existing tests + the new one).

- [ ] **Step 6: Commit**

```bash
git add toksearch/llm/cli.py tests/test_llm_cli.py
git commit -m "Add --gui flag to toksearch chat for launching the Gradio GUI"
```

---

## Task 10: open PR and verify CI

**Files:** none (process task)

- [ ] **Step 1: Push the branch and open the PR**

```bash
git push -u origin feat/llm-chat-gui
gh pr create --base main --head feat/llm-chat-gui \
  --title "Local Gradio chat GUI for toksearch.llm" \
  --body "$(cat <<'EOF'
## Summary

Adds a local browser-based chat GUI for \`toksearch.llm\`. Split-pane
layout: chat on the left with collapsible tool-call expanders, plot
pane on the right with the current figure + a history gallery.
Plotly is the documented default; matplotlib still supported. Both
capture mechanisms route through the same per-turn queue so figures
appear in the side pane automatically — no new agent API.

New entry points:
- \`python -m toksearch.llm.gui\`
- \`toksearch chat --gui\`

\`fdp chat --gui\` ships in a companion PR against GA-FDP/fdp.

## Test plan

- [x] \`pixi run python -m unittest tests.test_llm_gui_figure_capture\`
- [x] \`pixi run python -m unittest tests.test_llm_gui_bridge\`
- [x] \`pixi run python -m unittest tests.test_llm_gui_app\`
- [x] \`pixi run python -m unittest tests.test_llm_gui_launch\`
- [x] \`pixi run python -m unittest tests.test_llm_cli\`
- [ ] CI: conda build/test
- [ ] Manual: \`pixi run python -m toksearch.llm.gui --backend amsc\` (in toksearch_d3d env once deps are available), ask the agent for a scatter plot, verify it appears in the right pane with hover interactivity.
EOF
)"
```

- [ ] **Step 2: Wait for CI**

Watch the PR's `Conda Package CI/CD` run; iterate on any failures (most likely candidates: the new conda deps causing a solver conflict, or the smoke test in `test_llm_gui_app` failing on the Gradio version installed). Do not merge until green.

- [ ] **Step 3: Run the manual smoke test**

From the **toksearch_d3d** repo's pixi env (which has a real Anthropic-compatible backend wired up via amsc):

```bash
pixi run python -m toksearch.llm.gui --backend amsc
```

- Browser opens.
- Header shows `toksearch chat (backend: anthropic, model: claude-sonnet-4-6)` somewhere in stderr (printed by the underlying `build_session` machinery if it logs).
- Ask "make a scatter plot of x vs sin(x) for x in 0..10". The agent should call `run_python` once; a plotly figure should appear in the side pane within a few seconds; hover should show point values.
- Click `Reset` — chat clears, plot remains (no, plot also clears because the test asks for both?). Confirm against the spec: the spec says `Reset` clears chat history; `Clear figures` clears the plot list. Adjust if behavior diverges.

If everything looks right, request review on the PR.

- [ ] **Step 4: Merge**

Merge via rebase once review is in. (No squash — preserve the per-task history for future bisecting.)

---

## Task 11: `fdp chat --gui` (companion PR in GA-FDP/fdp)

**Files:** `fdp/llm_shims.py`, `fdp/cli.py` (both in the **GA-FDP/fdp** repo, not toksearch)

- [ ] **Step 1: Branch and edit `fdp/llm_shims.py`**

In the `fdp` repo:

```bash
cd /fusion/projects/dt/sammuli/fdp_dev/repos/fdp
git fetch origin main
git checkout -b feat/chat-gui-flag origin/main
```

In `fdp/llm_shims.py`, the existing `do_chat(args, device)` looks like:

```python
def do_chat(args, device: Device) -> None:
    passthrough = _common_passthrough(args)
    cmd = _build_llm_cmd("chat", passthrough, device)
    os.execvpe(cmd[0], cmd, os.environ)
```

Add `--gui` and `--no-browser` to the passthrough list — extend `_common_passthrough` to forward them when set:

```python
def _common_passthrough(args) -> list[str]:
    out: list[str] = []
    if getattr(args, "backend", None):
        out.extend(["--backend", args.backend])
    if getattr(args, "model", None):
        out.extend(["--model", args.model])
    if getattr(args, "max_iterations", None) is not None:
        out.extend(["-n", str(args.max_iterations)])
    if getattr(args, "gui", False):
        out.append("--gui")
    if getattr(args, "open_browser", True) is False:
        out.append("--no-browser")
    return out
```

- [ ] **Step 2: Add the `--gui` / `--no-browser` flags to `fdp chat`**

In `fdp/cli.py`, find `_add_llm_args` (or wherever `--backend` etc. are added to `p_chat`). Add:

```python
    p_chat.add_argument("--gui", action="store_true",
                          help="Launch the local Gradio chat GUI "
                               "instead of the terminal REPL.")
    p_chat.add_argument("--no-browser", dest="open_browser",
                          action="store_false", default=True,
                          help="When --gui is set, do not open a "
                               "browser tab.")
```

Put these only on the `p_chat` parser (NOT on `p_query` — query is one-shot, no GUI sensible).

- [ ] **Step 3: Add a test for the GUI passthrough**

Append to `tests/test_llm_shims.py`:

```python
class TestGuiPassthrough(unittest.TestCase):
    def test_gui_flag_appears_in_cmd(self):
        from fdp.llm_shims import _build_llm_cmd, _common_passthrough
        from fdp.devices import Device

        args = mock.Mock(backend=None, model=None,
                          max_iterations=None,
                          gui=True, open_browser=False)
        passthrough = _common_passthrough(args)
        self.assertIn("--gui", passthrough)
        self.assertIn("--no-browser", passthrough)
```

- [ ] **Step 4: Verify all fdp tests pass**

Run: `pixi run python -m unittest discover -s tests`
Expected: PASS (existing 45 + 1 new).

- [ ] **Step 5: Push, open PR, wait for CI, merge**

```bash
git add fdp/cli.py fdp/llm_shims.py tests/test_llm_shims.py
git commit -m "Add --gui / --no-browser passthrough to fdp chat"
git push -u origin feat/chat-gui-flag
gh pr create --base main --head feat/chat-gui-flag \
  --title "Forward --gui / --no-browser through fdp chat" \
  --body "Companion to GA-FDP/toksearch's chat GUI PR. Adds the two flags to the \`fdp chat\` argparse surface and forwards them through the execvpe call into \`toksearch.llm.cli\`."
```

Once CI is green and toksearch has shipped a release containing the GUI, merge via rebase.

- [ ] **Step 6: Cut releases (operator-run, requires user approval per project policy)**

- Tag `release-2.7.3` on toksearch's main once the GUI PR lands.
- Tag `release-0.1.3` on fdp's main once both that PR and toksearch 2.7.3 are on ga-fdp.
- No toksearch_d3d release required — the pins `toksearch >=2.7.1` / `fdp >=0.1.1` already accept the new versions.

---

## Self-review notes

**Spec coverage.** Walked each spec section:
- Architecture diagram → Task 5 (worker thread + queue) and Task 8 (in-process Session ownership).
- Module layout → Task 1 (skeleton), Tasks 2-7 (per-file population).
- Two-pane split layout → Task 6.
- Plot capture matplotlib → Task 2; plotly → Task 3; namespace + tool description → Task 1.
- Bridge with queue + apply_event → Task 4 (events + folder), Task 5 (run_turn).
- Invocation paths → Task 7 (\_\_main\_\_), Task 9 (toksearch chat --gui), Task 11 (fdp chat --gui).
- Error handling → Task 4's `Error` event + `_apply_event` branch; chat-pane red message tested.
- Testing → unit tests for capture, bridge, app smoke test; manual smoke covered in Task 10.
- Dependencies added → Task 1 (recipe.yaml + pixi.toml).
- Follow-ups (Jupyter, multi-session, pinning, persistence) → explicitly out of scope in the spec, called out in the scope notes here.

**Placeholder scan.** No "TBD" / "add validation" / "similar to Task N" patterns. Every code block is complete.

**Type consistency.** Cross-checked:
- `OnFigure = Callable[[str, object], None]` — used identically in `wrap_run_python_handler` and `install_plotly_renderer`.
- `ChatState.messages` is `list[dict]` with `{"role", "content"}` keys; Gradio's `gr.Chatbot(type="messages")` consumes that exact shape.
- `FigState.figures` is `list[tuple[str, Any]]`; `_current` / `_gallery` access it consistently; `_apply_event(Figure)` appends with the matching shape.
- Event dataclasses (`Text`, `ToolCall`, `ToolResult`, `Figure`, `Error`, `Done`) are referenced by the same names in tests and in `_apply_event`'s isinstance checks.
- `_active_figure_emitter` is the same symbol in `bridge.py` and `__init__.py`'s import.
- `args.gui` / `args.open_browser` are the attribute names used in both `cli.py` and `fdp/cli.py` plus their `__main__.py` argparse counterparts.

No issues found.
