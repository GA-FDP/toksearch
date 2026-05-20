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
