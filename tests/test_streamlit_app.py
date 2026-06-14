"""Smoke test: the Streamlit app module compiles cleanly.

We compile rather than import because importing executes Streamlit calls that
expect a running Streamlit runtime. Compilation catches syntax/indentation
errors without needing the server.
"""

import py_compile
from pathlib import Path

APP = Path(__file__).resolve().parents[1] / "streamlit_app.py"


def test_streamlit_app_compiles():
    assert APP.exists()
    # Raises py_compile.PyCompileError on failure, failing the test.
    py_compile.compile(str(APP), doraise=True)
