"""Keep chroma_db/ single-owner: a direct (PersistentClient) script and a `chroma run` server must never overlap.

claim_chroma_port() does both directions at once:
  - server already running  -> the port is taken, so we refuse to start;
  - we start first          -> we hold the port, so `chroma run` fails with "address already in use".
The OS frees the port when the process ends, even after a crash, so there is no stale lock file.
"""
import socket
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # make `src` importable

from src.config import CHROMA_SERVER_HOST, CHROMA_SERVER_PORT

_held = []   # keep the sockets referenced for the life of the process


def _addresses(host, port):
    # `localhost` can mean ::1 and/or 127.0.0.1 (Chroma binds ::1 on Windows), so claim every one of them
    return sorted({(af, sa) for af, _, _, _, sa in socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)})


def claim_chroma_port(host=CHROMA_SERVER_HOST, port=CHROMA_SERVER_PORT):
    """Exit if a Chroma server (or another ingest) holds the port; otherwise hold it until this process exits."""
    for family, sockaddr in _addresses(host, port):
        s = socket.socket(family, socket.SOCK_STREAM)
        if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):            # Windows: nobody else may share this port
            s.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        try:
            s.bind(sockaddr)
        except OSError:
            s.close()
            sys.exit(
                f"Port {host}:{port} is already in use: a Chroma server (or another ingest) is probably running.\n"
                "Stop it first (Ctrl+C in its terminal), then re-run. Two processes on chroma_db/ "
                "can lock or corrupt it.\n"
                "If your server uses another port, update CHROMA_SERVER_PORT in src/config.py."
            )
        _held.append(s)   # bound but not listening: connections are refused, and the port stays ours
