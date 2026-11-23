"""Stateful websocket connection to a real MemoryAgent policy server."""

from __future__ import annotations

from typing import Any, Mapping
from urllib.parse import urlsplit, urlunsplit

from .protocol import packb, unpackb


class MemoryAgentProtocolError(RuntimeError):
    pass


class MemoryAgentConnection:
    """Handle the OpenPI metadata handshake and per-episode reset marker."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int | None = 8000,
        *,
        api_key: str | None = None,
        open_timeout: float = 60.0,
        request_timeout: float | None = 300.0,
    ) -> None:
        import websockets.sync.client

        self._request_timeout = request_timeout
        self._socket = None
        uri = self._uri(host, port)
        headers = {"Authorization": f"Api-Key {api_key}"} if api_key else None
        kwargs = dict(
            compression=None,
            max_size=None,
            open_timeout=float(open_timeout),
            ping_interval=None,
        )
        if headers:
            kwargs["additional_headers"] = headers
        try:
            self._socket = websockets.sync.client.connect(uri, **kwargs)
            metadata = self._receive()
            self.metadata = dict(metadata)
            self._pending_reset = False
        except BaseException:
            self.close()
            raise

    @staticmethod
    def _uri(host: str, port: int | None) -> str:
        raw = str(host).strip()
        if not raw.startswith(("ws://", "wss://")):
            raw = f"ws://{raw}"
        parsed = urlsplit(raw)
        if not parsed.hostname:
            raise ValueError(f"invalid websocket host/URI {host!r}")
        if parsed.port is not None:
            return raw
        if port is None:
            return raw
        hostname = parsed.hostname
        if ":" in hostname and not hostname.startswith("["):
            hostname = f"[{hostname}]"
        return urlunsplit(
            (parsed.scheme, f"{hostname}:{int(port)}", parsed.path, parsed.query, parsed.fragment)
        )

    def _receive(self) -> Any:
        raw = self._socket.recv(timeout=self._request_timeout)
        if isinstance(raw, str):
            raise MemoryAgentProtocolError(f"server returned a text error:\n{raw}")
        return unpackb(raw)

    def reset(self) -> None:
        self._pending_reset = True

    def _request(self, observation: Mapping[str, Any], *, observe_only: bool) -> dict:
        request = dict(observation)
        if observe_only:
            request["observe_only"] = True
        else:
            request.pop("observe_only", None)
        pending = self._pending_reset
        if pending:
            request["reset"] = True
        self._socket.send(packb(request))
        response = self._receive()
        if not isinstance(response, dict):
            raise MemoryAgentProtocolError("policy response must be a dict")
        if pending:
            self._pending_reset = False
        return dict(response)

    def infer(self, observation: Mapping[str, Any]) -> dict:
        return self._request(observation, observe_only=False)

    def observe(self, observation: Mapping[str, Any]) -> dict:
        return self._request(observation, observe_only=True)

    def close(self) -> None:
        socket = self._socket
        if socket is not None:
            try:
                socket.close()
            finally:
                self._socket = None

    def __enter__(self):
        return self

    def __exit__(self, *_args) -> None:
        self.close()
