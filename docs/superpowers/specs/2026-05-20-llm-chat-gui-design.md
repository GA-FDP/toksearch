# `toksearch.llm` Chat GUI — Design

**Status:** approved (brainstorming complete)

**Audience:** implementer following with the `superpowers:writing-plans` skill.

## Goal

Add a local browser-based GUI for `toksearch.llm`'s chat session so users can carry on a conversation with the agent *and* see the visualizations it generates. The CLI (`toksearch chat`, `fdp chat`) stays — the GUI is an additional surface that wraps the same `Session` object the CLI uses.

Out of scope for v1: notebook integration (planned as a follow-up project), shared-server / multi-user deployments, persistence across server restarts, side-by-side plot pinning.

## Architecture

```
                 +------------------------+
  browser tab -->| Gradio Blocks app      |
                 |   (one server,         |
                 |    one Session,        |
                 |    one figure history) |
                 +-----------+------------+
                             |
                             v
                 +------------------------+
                 |  toksearch.llm         |
                 |  Session.send(...)     |---> on_text / on_tool_call /
                 |                        |     on_tool_result + figure
                 |                        |     events through a Queue
                 +------------------------+
```

The agent runs in-process on a background worker thread. Gradio handlers drain a `queue.Queue` of `(kind, payload)` events and yield UI state updates back to the browser tab via Gradio's generator protocol. No new agent code — the same `Session.send()` path the CLI uses.

The default is one `Session` per Gradio server: multiple browser tabs see the same conversation. Multi-session can be added later if needed.

## Module layout

A new sub-package under core toksearch:

```
toksearch/llm/gui/
  __init__.py         # public API: launch_gui(host, port, open_browser)
  app.py              # gr.Blocks layout (chat pane + plot pane)
  bridge.py           # Session-to-Gradio adapter (threads, queue, figure events)
  figure_capture.py   # matplotlib + plotly capture mechanisms
```

- `app.py` owns the visual: `gr.Blocks` with a `gr.Chatbot` on the left, a `gr.Plot` (current) + `gr.Gallery` (history) on the right, a `gr.Textbox` for input.
- `bridge.py` owns the streaming dance: kicks off `Session.send()` on a worker thread, drains a `queue.Queue` of typed events, yields Gradio state tuples back to the UI generator.
- `figure_capture.py` houses both capture mechanisms (matplotlib post-exec sweep + plotly custom renderer) and the `wrap_run_python_for_capture()` helper used at app launch.

## Layout

Two-pane split: chat on the left, plot pane on the right.

```
+----------------------+---------------+
| amsc> Querying...    | +-----------+ |
|                      | |           | |
| ▶ [lookup_docs] ...  | |  current  | |
| ▶ [run_python] ...   | |    plot   | |
|                      | |           | |
| Here are results...  | +-----------+ |
|                      |  fig 4 / 4    |
| you> ip vs time [send]| [⬇PNG] [Clear]|
|                      | [thumb thumb] |
+----------------------+---------------+
```

### Chat pane (left)

A `gr.Chatbot` rendering markdown messages. Tool calls are injected directly into the assistant message as collapsible `<details>` blocks — the browser handles expand/collapse with no JS:

```
amsc> Querying d3drdb for last week's plasma shots.

  ▶ [lookup_docs] connecting d3drdb
  ▶ [run_python] Fetch 600 shots
  ▶ [run_python] Compute βN vs NBI

612 plasma shots from May 13–19. See the scatter
plot in the side pane: peak βN vs peak NBI power,
colored by date.
```

`on_tool_call` and the matching `on_tool_result` together produce one `<details>` with the model's `thought` (or a key-arg fallback) as the summary, and the code + output body inside. Streaming `on_text` deltas append to the same message bubble in place.

### Plot pane (right)

Two parts:

- **Current view** (`gr.Plot`): the figure the user has focus on. Defaults to the most recently captured. Plotly figures keep zoom/pan/hover interactivity; matplotlib figures render as static PNG. A `⬇ PNG` button downloads a static export.
- **History gallery** (`gr.Gallery` with `selected_index` bound to a state var): every captured figure in chronological order. Click a thumbnail to swap it into the current view. Plotly figures get a thumbnail via `fig.to_image()` (kaleido); matplotlib figures contribute their existing render. Gallery auto-scrolls to newest on each new figure.

A `Clear figures` button under the gallery resets the figure list (does not touch chat history).

## Plot capture

The `run_python` tool's handler is wrapped at GUI launch (the wrapped handler is registered on the `Session` instance the GUI uses; the CLI's `Session` is unaffected). The wrapper invokes both capture mechanisms after each call.

### Matplotlib

```python
def _figure_capturing_run_python(args, session):
    result = _original_run_python(args, session)
    import matplotlib.pyplot as plt
    for num in plt.get_fignums():
        bridge.fire_figure("matplotlib", plt.figure(num))
    plt.close("all")
    return result
```

Walks `pyplot`'s figure registry after `exec()` returns; emits each figure onto the bridge queue; closes all to clear state for the next call.

### Plotly

Plotly has a renderer plugin system. Register a custom renderer once at GUI startup; set it as the default. When the agent calls `fig.show()`, the renderer fires onto the bridge queue.

```python
import plotly.io as pio
from plotly.io._base_renderers import ExternalRenderer

class _GuiRenderer(ExternalRenderer):
    def render(self, fig_dict):
        bridge.fire_figure("plotly", fig_dict)

pio.renderers["toksearch_gui"] = _GuiRenderer()
pio.renderers.default = "toksearch_gui"
```

### Namespace + tool description

The `run_python` namespace gains `px` (`plotly.express`) and `go` (`plotly.graph_objects`) alongside the existing `plt`, `pd`, `np`, `toksearch`.

The tool description gets one paragraph appended:

> Prefer plotly (`px`, `go`) for visualizations — the GUI renders them as interactive figures with zoom/pan/hover. Use matplotlib (`plt`) only for cases plotly handles poorly (animations, specialized scientific plots). Either way, call `fig.show()` or `plt.show()` to send the figure to the side pane.

Plotly becomes a run-dep of toksearch (already on conda-forge as `plotly`). Kaleido (for static thumbnail export) is also added.

## Bridge

The bridge owns one `queue.Queue` and one worker thread per turn.

```python
def chat_handler(user_text, chat_state, fig_state):
    q = queue.Queue()
    def runner():
        try:
            session.send(
                user_text,
                on_text=lambda t: q.put(("text", t)),
                on_tool_call=lambda c: q.put(("tool_call", c)),
                on_tool_result=lambda r: q.put(("tool_result", r)),
            )
        except Exception as e:
            q.put(("error", e))
        finally:
            q.put(("done", None))

    threading.Thread(target=runner, daemon=True).start()

    pending_calls: dict[str, ToolCall] = {}  # id -> call awaiting result
    while True:
        kind, payload = q.get()
        if kind == "done":
            return
        if kind == "error":
            chat_state = append_error(chat_state, payload)
            yield chat_state, gr.update(), gr.update()
            return
        chat_state, fig_state = _apply_event(chat_state, fig_state,
                                              kind, payload, pending_calls)
        yield chat_state, fig_state.current, fig_state.gallery
```

Figure events from the matplotlib hook and plotly renderer also drop onto `q` — `run_python` runs on the worker thread, so its captures happen on the same thread that owns the queue producer side. One queue, one consumer, no shared-state races.

`_apply_event` is the pure-function update: takes the event kind + current state and returns new state. Unit-testable without any threading.

## Invocation

Three entry points, all thin:

| Command | Behavior |
|---|---|
| `python -m toksearch.llm.gui` | Direct launch via `toksearch.llm.gui.launch_gui()`. |
| `toksearch chat --gui` | New flag on the existing `toksearch chat` subcommand. When set, delegates to `launch_gui(...)` instead of entering the REPL. |
| `fdp chat --gui` | New flag on `fdp chat`. Same `os.execvpe` pattern as today; just appends `--gui` to the toksearch argv. |

`launch_gui(host="127.0.0.1", port=None, open_browser=True)` calls `gr.Blocks.launch(server_name=..., server_port=..., inbrowser=True, share=False)`. Defaults: localhost, auto-picked port, browser opens. A `--no-browser` flag suppresses the auto-open (useful when running in a remote shell with port-forwarding).

The Gradio app uses the same `Session` build path as the CLI (`build_session(args)`) so `--backend`, `--model`, `-n / --max-iterations`, and `--package` flags carry over. The `-v / --verbose` flag is meaningless in GUI mode (the chat pane has always-collapsible expanders) and is silently ignored when `--gui` is set.

## Error handling

- `LLMError` raised inside `session.send()` becomes a red message in the chat showing the first line of the exception. Full traceback lives behind an expander in the same message.
- `fig.to_image()` failing (kaleido missing or otherwise) logs a warning, skips the gallery thumbnail, but still shows the plotly figure in the current view via its HTML representation.
- The Gradio process closes its socket cleanly on `Ctrl-C`; the worker thread is daemonic so it won't keep the interpreter alive after the server stops.

## Testing

| Layer | Test |
|---|---|
| `figure_capture.py` | Unit tests with a `FakeBridge`: call wrapped `run_python` with code that creates one matplotlib figure; assert `fire_figure("matplotlib", ...)` was called with a `Figure` instance. Similarly for plotly: invoke the renderer's `render()` directly and check the event payload. |
| `bridge.py` | Unit test `_apply_event` with hand-built events; assert the returned `(chat_state, fig_state)` matches the expected snapshots. Test the worker-thread integration with a synchronous fake `Session` that fires events in order. |
| `app.py` | Smoke test: instantiate `gr.Blocks`, assert the named components (`chatbot`, `current_plot`, `gallery`, `input`) exist. No browser-driving Selenium test in v1. |

Manual end-to-end is the acceptance criterion: `pixi run fdp chat --gui` (in a toksearch_d3d env updated to the toksearch release that ships the GUI) → browser opens → ask for a scatter plot → plot appears in the side pane with hover interactivity.

## Dependencies added

| Package | Why |
|---|---|
| `gradio >=4` | The whole UI shell. |
| `plotly` | Interactive plot library, made the documented default. |
| `kaleido` | Static thumbnail export for the gallery (plotly only). |

All available on conda-forge. Added to `toksearch`'s conda run-deps under a new `gui` extras group so headless users don't pay the install cost unless they want the GUI.

## Follow-up scope (not in v1)

1. **Jupyter integration.** A `toksearch.llm.notebook` companion that exposes the same `Session` as an ipywidget rendering inline in a notebook. Shares `figure_capture.py` with the GUI. Same architecture; different shell.
2. **Multi-session.** Per-tab `Session` via Gradio state.
3. **Pin a plot side-by-side.** Allow two figures in the right pane for direct comparison.
4. **Persistence.** Save/load chat + figure history across server restarts.
