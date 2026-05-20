# `toksearch.llm` Chat GUI — Design

**Status:** implemented as PR #35 (`feat/llm-chat-gui-v2`)

**Audience:** anyone reviewing or extending the chat GUI.

## Goal

Add a local browser-based GUI for `toksearch.llm`'s chat session so users can carry on a conversation with the agent *and* see the visualizations it generates. The CLI (`toksearch chat`, `fdp chat`) stays — the GUI is an additional surface that wraps the same `Session` object the CLI uses.

Out of scope for v1: notebook integration (planned as a follow-up project), shared-server / multi-user deployments, persistence across server restarts.

## Architecture

```
                 +------------------------+
  browser tab -->| gr.ChatInterface app   |
                 |   (one server,         |
                 |    one Session)        |
                 +-----------+------------+
                             |
                             v
                 +------------------------+
                 |  toksearch.llm         |
                 |  Session.send(...)     |---> on_text / on_tool_call /
                 |                        |     on_tool_result callbacks
                 |                        |     + figure events via
                 |                        |     figure_capture active emitter
                 +------------------------+
```

The agent runs in-process on a background worker thread spawned per chat turn. A `queue.Queue` carries `(kind, payload)` events from the worker to the chat function; the chat function is a generator that yields a list of `gr.ChatMessage` instances representing the assistant's evolving response. `gr.ChatInterface` owns the rest of the UI: input box, submit button, history, and message rendering.

The default is one `Session` per Gradio server: multiple browser tabs share the same conversation. Multi-session can be added later if needed.

## Module layout

A new sub-package under core toksearch:

```
toksearch/llm/gui/
  __init__.py         # public API: launch_gui(args, host, port, open_browser)
  app.py              # build_app(session) + chat_fn streaming generator
  figure_capture.py   # matplotlib + plotly capture + active-emitter dispatch
  __main__.py         # python -m toksearch.llm.gui entry point
```

- `app.py` defines `build_app(session)` which returns a `gr.Blocks` wrapping a single `gr.ChatInterface`. Its inner `chat_fn(message, history)` generator drives one turn: spawns a worker thread that calls `session.send(...)`, drains events from the per-turn queue, and yields a growing `list[gr.ChatMessage]` to the UI.
- `figure_capture.py` owns both capture mechanisms (matplotlib post-exec sweep + plotly custom renderer) plus the `_active_figure_emitter` global the chat function swaps per turn. Capture before any turn is active silently drops figures.
- `__main__.py` provides argparse glue for `python -m toksearch.llm.gui`.

There is no `bridge.py` — Gradio's `ChatInterface` handles state and history, so no separate adapter layer is needed.

## Message rendering

Each chat turn yields a `list[gr.ChatMessage]` that grows as events arrive. Three message kinds, all built from native Gradio primitives:

### Assistant text

```python
gr.ChatMessage(role="assistant", content="Querying d3drdb...")
```

Streaming `on_text` chunks are concatenated into the trailing text bubble (no metadata, content is `str`).

### Tool calls (collapsible)

```python
gr.ChatMessage(
    role="assistant",
    content="(running…)",   # replaced with output when the result arrives
    metadata={
        "title": "🔧 run_python: Fetch 600 shots",
        "status": "pending",  # → "done" | "error"
    },
)
```

Gradio renders these as native collapsible "thinking" sections in the chat. The title prefers the LLM's `thought` field (which `run_python`'s schema requires) and falls back to a key arg like `skill_name` for tools without one. When the matching `tool_result` event arrives, the same message is updated in place with the output body (truncated at 2000 chars) and `status` flipped to `"done"` / `"error"`.

### Inline plots

```python
gr.ChatMessage(role="assistant", content=gr.Plot(fig))
```

`gr.Plot` accepts both matplotlib and plotly figures. Plotly fig-dicts captured from the custom renderer are coerced via `go.Figure(fig_dict)` on the way in. Plots appear in the conversation alongside text and tool-call expanders — same shape as ChatGPT / Claude.ai inline visualizations. No separate plot pane.

## Plot capture

The `run_python` tool's handler is wrapped at GUI launch (the wrapped handler is registered on the `Session` instance the GUI uses; the CLI's `Session` is unaffected). The wrapper invokes both capture mechanisms after each call.

### Matplotlib

```python
def _figure_capturing_run_python(args, session):
    result = _original_run_python(args, session)
    import matplotlib.pyplot as plt
    for num in plt.get_fignums():
        on_figure("matplotlib", plt.figure(num))
    plt.close("all")
    return result
```

Walks `pyplot`'s figure registry after `exec()` returns; emits each figure via `on_figure`; closes all to clear state for the next call.

### Plotly

A custom external renderer registered as the default. When the agent calls `fig.show()`, the renderer fires through `on_figure` with the figure dict.

```python
class _GuiRenderer(ExternalRenderer):
    def render(self, fig_dict):
        on_figure("plotly", fig_dict)

pio.renderers["toksearch_gui"] = _GuiRenderer()
pio.renderers.default = "toksearch_gui"
```

### Active-emitter indirection

Both capture mechanisms call `figure_capture._dispatch_to_active_emitter(kind, payload)`, which looks up the module-level `_active_figure_emitter` at call time (not at closure-binding time) and forwards if set. The chat function installs a per-turn emitter that drops captured figures onto its queue, then restores the previous emitter on exit. Figures captured outside any chat turn are silently dropped.

### Namespace + tool description

The `run_python` namespace gains `px` (`plotly.express`) and `go` (`plotly.graph_objects`) alongside the existing `plt`, `pd`, `np`, `toksearch`.

The tool description gets one paragraph appended:

> Prefer plotly (`px`, `go`) for visualizations — the GUI renders them as interactive figures with zoom/pan/hover. Use matplotlib (`plt`) only for cases plotly handles poorly (animations, specialized scientific plots). Either way, call `fig.show()` or `plt.show()` to send the figure to the side pane in the GUI; in CLI mode the call is captured as text output.

Plotly is a run-dep of toksearch (already on conda-forge). `python-kaleido` is added for static thumbnail export.

## Chat function

```python
def chat_fn(message, history):
    q = queue.Queue()
    def runner():
        try:
            session.send(
                message,
                on_text=lambda t: q.put(("text", t)),
                on_tool_call=lambda c: q.put(("tool_call", c)),
                on_tool_result=lambda r: q.put(("tool_result", r)),
            )
        except Exception as e:
            q.put(("error", (str(e), traceback.format_exc())))
        finally:
            q.put(("done", None))

    previous_emitter = figure_capture._active_figure_emitter
    _set_active_figure_emitter(
        lambda kind, payload: q.put(("figure", (kind, payload))))
    threading.Thread(target=runner, daemon=True).start()

    messages: list[gr.ChatMessage] = []
    pending: dict[str, int] = {}  # tool_use_id -> index into messages

    try:
        while True:
            kind, payload = q.get()
            if kind == "done":
                return
            # ... fold event into messages list ...
            yield messages
    finally:
        _set_active_figure_emitter(previous_emitter)
```

`pending` maps tool-use IDs to indices in the `messages` list so the `tool_result` handler can update the matching collapsible bubble in place when the result arrives.

## Invocation

Three entry points, all thin:

| Command | Behavior |
|---|---|
| `python -m toksearch.llm.gui` | Direct launch via `toksearch.llm.gui.launch_gui()`. |
| `toksearch chat --gui` | New flag on the existing `toksearch chat` subcommand. When set, delegates to `launch_gui(...)` instead of entering the REPL. |
| `fdp chat --gui` | Follow-up PR in GA-FDP/fdp; forwards `--gui` and `--no-browser` through the existing `os.execvpe` shim. |

`launch_gui(host="127.0.0.1", port=None, open_browser=True)` calls `gr.Blocks.launch(server_name=..., server_port=..., inbrowser=True, share=False)`. Defaults: localhost, auto-picked port, browser opens. A `--no-browser` flag suppresses the auto-open (useful when running on a remote host with port-forwarding).

The Gradio app uses the same `Session` build path as the CLI (`build_session(args)`) so `--backend`, `--model`, `-n / --max-iterations`, and `--package` flags carry over. The `-v / --verbose` flag is meaningless in GUI mode (tool-call expanders are always available via the collapsible UI) and is silently ignored when `--gui` is set.

## Error handling

- `Exception` raised inside `session.send()` is caught by the worker thread and emitted as an `error` event. The chat function appends a red ⛔-titled `ChatMessage` with the traceback inside its body.
- Plot capture errors (e.g., `kaleido` missing) log a warning and skip the affected step; chat continues.
- The Gradio process closes its socket cleanly on `Ctrl-C`; the worker thread is daemonic so it won't keep the interpreter alive after the server stops.

## Testing

| Layer | Test |
|---|---|
| `figure_capture.py` | Unit tests with a `FakeBridge`: call wrapped `run_python` with code that creates one matplotlib figure; assert `fire_figure("matplotlib", ...)` was called with a `Figure` instance. Similarly for plotly: invoke the renderer's `render()` directly and check the event payload. |
| `app.py` | Unit tests with a `FakeSession` that fires synchronous callbacks: drain `chat_fn` and assert the yielded message list contains the expected `ChatMessage` shapes — accumulated text bubble, collapsible tool-call expander with `metadata.status == "done"`, key-arg fallback in titles, inline plot bubble from a figure event, red error bubble. |
| `__init__.py` | Mock `build_session` + `build_app`; assert `blocks.launch(...)` kwargs match `launch_gui`'s args. Mock `figure_capture` helpers; assert `wrap_run_python_handler` + `install_plotly_renderer` both called and that the `run_python` ToolSpec ended up with the wrapped handler. |
| `cli.py` | Assert `toksearch chat --gui` short-circuits to `launch_gui` instead of the REPL; assert `--no-browser` passes through. |

Manual end-to-end smoke: `pixi run python -m toksearch.llm.gui --backend amsc` (in a `toksearch_d3d` env with the GUI deps) → browser opens → ask for a scatter plot → plotly figure appears as an inline bubble with hover interactivity.

## Dependencies added

| Package | Why |
|---|---|
| `gradio >=4,<7` | The whole UI shell (tested against 6.x). |
| `plotly` | Interactive plot library, made the documented default. |
| `python-kaleido` | Static thumbnail export for plotly (conda-forge name; pip's `kaleido` differs). |

All available on conda-forge. Added to `toksearch`'s conda run-deps so headless users still get the dependencies — the import cost is small and it keeps the conda surface simple.

## Lessons from the earlier attempt (`feat/llm-chat-gui`)

An earlier branch built a split-pane `gr.Blocks` layout with a manual chatbot + state-management adapter (`bridge.py`) modeled on the brainstorming session's first design. It got stuck on what appeared to be a Gradio websocket / multi-output diff issue — every chat turn dropped the connection after the first yield. We spent significant effort instrumenting the bridge and rewriting the yield contract.

The actual root cause was browser-side: the remote dev host's Firefox couldn't successfully establish a websocket to a local Gradio server. SSH-tunneling to a laptop browser (Chrome) worked fine. v1 would probably have worked too once tunneled, but the diagnostic confusion made the v2 design pivot worth it on its own merits — `gr.ChatInterface` is materially simpler than the manual Blocks plumbing, inline plots match modern chat-UI conventions, and the code is half the size.

The lesson: when seeing "connection lost" with no server-side traceback, **eliminate the browser path before instrumenting code**. Run a stock Gradio chat demo through the same browser; if it also fails, the issue is the transport/browser, not your code.

## Follow-up scope (not in v1)

1. **Jupyter integration.** A `toksearch.llm.notebook` companion that exposes the same `Session` as an ipywidget rendering inline in a notebook. Shares `figure_capture.py` with the GUI.
2. **Multi-session.** Per-tab `Session` via `gr.State` or session keys.
3. **Side pane.** If demand emerges for keeping plots visible while scrolling chat, a `gr.Plot` pane alongside `gr.ChatInterface` is straightforward via `additional_outputs`.
4. **Persistence.** Save/load chat + figure history across server restarts.
