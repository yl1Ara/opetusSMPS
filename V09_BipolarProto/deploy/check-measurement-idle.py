#!/usr/bin/env python3
"""Exit 0 when the shared GUI is idle, 10 when measuring, and 20 if unknown."""

import sys

import panel  # Register Panel's custom Bokeh models before pulling the document.
from bokeh.client import pull_session


def main() -> int:
    url = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:5006/gui"
    session = None
    try:
        session = pull_session(url=url)
        matches = [
            model
            for model in session.document.models
            if getattr(model, "label", None) == "Start measurement"
            and hasattr(model, "active")
        ]
        if len(matches) != 1:
            print("Could not uniquely locate the Start measurement toggle", file=sys.stderr)
            return 20
        if matches[0].active:
            print("Measurement is active", file=sys.stderr)
            return 10
        return 0
    except Exception as exc:
        print(f"Could not inspect measurement state: {exc}", file=sys.stderr)
        return 20
    finally:
        if session is not None:
            session.close()


if __name__ == "__main__":
    raise SystemExit(main())
