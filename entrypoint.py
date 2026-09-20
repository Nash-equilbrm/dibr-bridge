"""PyInstaller entry point — kept outside the `bridge` package so
`bridge/main.py`'s relative imports (`from .control_server import serve`,
etc.) resolve correctly. Running `bridge/main.py` directly as PyInstaller's
target script breaks those imports (no known parent package); this thin
wrapper imports `bridge` as a real package instead.
"""
from bridge.main import main

if __name__ == "__main__":
    main()
