import logging
import time
from typing import Any, Dict, Mapping, Optional, Tuple

from typing_extensions import override
import websockets.sync.client

from openpi_client import base_policy as _base_policy
from openpi_client import msgpack_numpy


class WebsocketClientPolicy(_base_policy.BasePolicy):
    """Implements the Policy interface by communicating with a server over websocket.

    See WebsocketPolicyServer for a corresponding server implementation.
    """

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: Optional[int] = None,
        api_key: Optional[str] = None,
        *,
        connect_kwargs: Optional[Mapping[str, Any]] = None,
    ) -> None:
        if host.startswith("ws"):
            self._uri = host
        else:
            self._uri = f"ws://{host}"
        if port is not None:
            self._uri += f":{port}"
        self._packer = msgpack_numpy.Packer()
        self._api_key = api_key
        # Keep connection tuning local to this client. In particular, callers
        # that need to disable keepalive pings must not monkey-patch the module-
        # level websockets connect function used by every other client.
        self._connect_kwargs = dict(connect_kwargs or {})
        self._pending_reset = False
        self._ws, self._server_metadata = self._wait_for_server()
        self.supports_observe = bool(self._server_metadata.get("supports_observe", False))

    def get_server_metadata(self) -> Dict:
        return self._server_metadata

    def _wait_for_server(self) -> Tuple[websockets.sync.client.ClientConnection, Dict]:
        logging.info(f"Waiting for server at {self._uri}...")
        while True:
            try:
                headers = {"Authorization": f"Api-Key {self._api_key}"} if self._api_key else None
                connect_kwargs = {
                    "compression": None,
                    "max_size": None,
                    "additional_headers": headers,
                }
                connect_kwargs.update(self._connect_kwargs)
                conn = websockets.sync.client.connect(self._uri, **connect_kwargs)
                metadata = msgpack_numpy.unpackb(conn.recv())
                return conn, metadata
            except ConnectionRefusedError:
                logging.info("Still waiting for server...")
                time.sleep(5)

    @override
    def infer(self, obs: Dict) -> Dict:  # noqa: UP006
        return self._request(obs, observe_only=False)

    @override
    def observe(self, obs: Dict) -> None:  # noqa: UP006
        """Send an observation-only update when the server advertises support.

        Ordinary OpenPI servers do not advertise ``supports_observe``. In that
        case this remains a true no-op and, importantly, does not consume a
        pending episode reset.
        """
        if not self.supports_observe:
            return
        self._request(obs, observe_only=True)

    def _request(self, obs: Dict, *, observe_only: bool) -> Dict:  # noqa: UP006
        request = dict(obs)
        # ``observe_only`` is a protocol control key, not an observation field.
        # Force it for observe() and strip any stale/user-provided value for infer().
        if observe_only:
            request["observe_only"] = True
        else:
            request.pop("observe_only", None)

        pending_reset = self._pending_reset
        if pending_reset:
            # Stateful servers mark episode boundaries on the first request. An
            # observe-only update can legitimately be that first request.
            request["reset"] = True

        data = self._packer.pack(request)
        self._ws.send(data)
        response = self._ws.recv()
        if isinstance(response, str):
            # we're expecting bytes; if the server sends a string, it's an error.
            raise RuntimeError(f"Error in inference server:\n{response}")
        result = msgpack_numpy.unpackb(response)
        # Consume reset only after a complete request/response. On transport or
        # server failure, retrying with reset is safer than silently losing it.
        if pending_reset:
            self._pending_reset = False
        return result

    @override
    def reset(self) -> None:
        # The websocket protocol is stateless request/response: there is no standalone reset
        # message. Instead, tag the next transmitted infer/observe request so a stateful server
        # resets its per-episode state before consuming the new episode's first observation.
        self._pending_reset = True
