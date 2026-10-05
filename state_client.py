#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""
CLI client for the training state UNIX socket.

Subcommands:
    ping                     — send {"cmd":"ping"}, print {"type":"pong"}
    get-state                — print the latest full snapshot
    subscribe [--events X,Y] — print the initial snapshot and all subsequent
                               events (optionally filtered) until disconnect

Output is one JSON object per line.
"""

from __future__ import annotations

import argparse
import json
import socket
import sys
from pathlib import Path
from typing import Any, Callable, cast


def _send_request(sock_path: str, request: dict[str, Any]) -> socket.socket:
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.connect(sock_path)
    line = json.dumps(request, separators=(",", ":")) + "\n"
    s.sendall(line.encode("utf-8"))
    return s


def _stream_lines(sock: socket.socket):
    """Yield each newline-delimited JSON payload from the socket as raw str."""
    buf = bytearray()
    while True:
        try:
            chunk = sock.recv(4096)
        except OSError:
            break
        if not chunk:
            if buf:
                yield buf.decode("utf-8", errors="replace")
            break
        buf.extend(chunk)
        while True:
            nl = buf.find(b"\n")
            if nl == -1:
                break
            line = buf[:nl].decode("utf-8", errors="replace")
            del buf[: nl + 1]
            if line:
                yield line


def cmd_ping(args) -> int:
    sock = _send_request(args.socket, {"cmd": "ping"})
    try:
        for line in _stream_lines(sock):
            print(line, flush=True)
            break
    finally:
        sock.close()
    return 0


def cmd_get_state(args) -> int:
    sock = _send_request(args.socket, {"cmd": "get_state"})
    try:
        for line in _stream_lines(sock):
            print(line, flush=True)
            break
    finally:
        sock.close()
    return 0


def cmd_subscribe(args: argparse.Namespace) -> int:
    req: dict[str, Any] = {"cmd": "subscribe"}
    if args.events:
        req["events"] = [e.strip() for e in args.events.split(",") if e.strip()]
    sock = _send_request(args.socket, req)
    try:
        for line in _stream_lines(sock):
            print(line, flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        sock.close()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="state_client",
        description="UNIX-socket client for training state snapshots and events.",
    )
    parser.add_argument(
        "--socket", required=True,
        help="Path to the training state UNIX socket (e.g. <output_dir>/.train.sock)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("ping", help="Ping the server").set_defaults(func=cmd_ping)
    sub.add_parser(
        "get-state", help="Print the latest full state snapshot",
    ).set_defaults(func=cmd_get_state)

    p_sub = sub.add_parser(
        "subscribe",
        help="Subscribe to event stream (initial snapshot, then events)",
    )
    p_sub.add_argument(
        "--events", default=None,
        help="Comma-separated event kinds to filter (e.g. epoch_done,new_best). "
             "Default: all events.",
    )
    p_sub.set_defaults(func=cmd_subscribe)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    sock_path = Path(args.socket)
    if not sock_path.exists():
        print(
            json.dumps({"type": "error", "message": f"socket not found: {sock_path}"}),
            flush=True,
        )
        return 2
    try:
        func = cast(Callable[[argparse.Namespace], int], args.func)
        return func(args)
    except ConnectionRefusedError as e:
        print(json.dumps({"type": "error", "message": f"connection refused: {e}"}))
        return 2
    except FileNotFoundError as e:
        print(json.dumps({"type": "error", "message": f"socket missing: {e}"}))
        return 2


if __name__ == "__main__":
    sys.exit(main())
