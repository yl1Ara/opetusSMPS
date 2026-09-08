#!/usr/bin/env python3
"""Exit successfully only when the configured local TCP port is available."""

import socket
import sys


def port_available(host, port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            listener.bind((host, int(port)))
        except OSError:
            return False
    return True


def main():
    host = sys.argv[1] if len(sys.argv) > 1 else "127.0.0.1"
    port = int(sys.argv[2]) if len(sys.argv) > 2 else 5006
    if port_available(host, port):
        return 0
    print(
        f"Refusing TDMPS startup: {host}:{port} is already in use; "
        "check for legacy tdmps.service or another Panel process.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
