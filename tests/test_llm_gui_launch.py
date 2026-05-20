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
"""Tests for the launch_gui() entry point and __main__."""

import unittest
from unittest import mock


class TestLaunchGui(unittest.TestCase):
    def _fake_session(self):
        from toksearch.llm.tools import ToolSpec
        original_handler = mock.Mock(return_value=mock.Mock(text="ok"))
        original_spec = ToolSpec(
            name="run_python",
            description="x",
            input_schema={},
            handler=original_handler,
        )
        session = mock.Mock()
        session._tools_by_name = {"run_python": original_spec}
        session.tool_specs = [original_spec]
        return session

    def test_launch_gui_builds_session_and_launches_blocks(self):
        from toksearch.llm import gui

        fake_blocks = mock.MagicMock()
        with mock.patch.object(gui, "build_session",
                                return_value=self._fake_session()), \
             mock.patch.object(gui, "build_app",
                                return_value=fake_blocks), \
             mock.patch.object(gui.figure_capture,
                                "install_plotly_renderer"):
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

    def test_launch_gui_wraps_run_python_and_installs_plotly(self):
        from toksearch.llm import gui
        from toksearch.llm.gui import figure_capture

        fake_session = self._fake_session()
        original_handler = fake_session._tools_by_name["run_python"].handler
        fake_blocks = mock.MagicMock()
        with mock.patch.object(gui, "build_session",
                                return_value=fake_session), \
             mock.patch.object(gui, "build_app",
                                return_value=fake_blocks), \
             mock.patch.object(figure_capture,
                                "wrap_run_python_handler") as wrap, \
             mock.patch.object(figure_capture,
                                "install_plotly_renderer") as install:
            wrap.return_value = mock.Mock()
            gui.launch_gui(args=mock.Mock(), open_browser=False)
        wrap.assert_called_once()
        # wrap was called with the original handler + on_figure callback
        self.assertIs(wrap.call_args.args[0], original_handler)
        install.assert_called_once()
        new_spec = fake_session._tools_by_name["run_python"]
        self.assertIs(new_spec.handler, wrap.return_value)


class TestMain(unittest.TestCase):
    def test_main_module_calls_launch_gui(self):
        from toksearch.llm.gui import __main__ as main_mod

        with mock.patch.object(main_mod, "launch_gui") as launch:
            main_mod.main(["--port", "9999", "--no-browser"])
        launch.assert_called_once()
        kwargs = launch.call_args.kwargs
        self.assertEqual(kwargs.get("port"), 9999)
        self.assertEqual(kwargs.get("open_browser"), False)


if __name__ == "__main__":
    unittest.main()
