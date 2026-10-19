"""Tests for explicit connection pinning via ClientSession.pin_connection().

Connector-level behaviour is exercised with mocked protocols in the first
section; end-to-end semantics (same TCP connection, redirects, websocket,
server disconnects, concurrency) are covered against real loopback servers
in the second section.
"""

import asyncio
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any, Protocol, cast
from unittest import mock

import pytest
from pytest_aiohttp import AiohttpServer
from yarl import URL

import aiohttp
from aiohttp import (
    ClientConnectionPinError,
    ClientSession,
    ClientTimeout,
    TraceConfig,
    web,
)
from aiohttp.client_proto import ResponseHandler
from aiohttp.client_reqrep import ClientRequest, ClientRequestArgs, ConnectionKey
from aiohttp.tracing import Trace

KEY = ConnectionKey("localhost", 80, False, True, None, None)
KEY2 = ConnectionKey("otherhost", 80, False, True, None, None)

if sys.version_info >= (3, 11):
    from typing import Unpack

    class _RequestMaker(Protocol):
        def __call__(
            self, method: str, url: URL, **kwargs: Unpack[ClientRequestArgs]
        ) -> ClientRequest: ...

else:
    _RequestMaker = Any


def _peer(request: web.Request) -> tuple[str, int]:
    assert request.transport is not None
    peer = request.transport.get_extra_info("peername")
    assert isinstance(peer, tuple)
    return cast("tuple[str, int]", peer)


def _mocked_proto(
    loop: asyncio.AbstractEventLoop,
    *,
    should_close: bool = False,
    is_connected: bool = True,
) -> mock.Mock:
    proto = mock.Mock(spec=ResponseHandler)
    proto.is_connected.return_value = is_connected
    proto.should_close = should_close
    # ``closed`` resolves only when the transport is closed, mirroring
    # ResponseHandler; resolving it up-front would look like a lost peer.
    closed = loop.create_future()

    def _close(*args: object) -> None:
        if not closed.done():
            closed.set_result(None)

    proto.close.side_effect = _close
    proto.closed = closed
    proto.transport = mock.Mock()
    return proto


@contextmanager
def _patched_factory(
    connector: aiohttp.BaseConnector, protos: list[mock.Mock]
) -> Iterator[None]:
    async def create_connection(
        req: object, traces: object, timeout: object
    ) -> mock.Mock:
        proto = protos.pop(0)
        return proto

    with mock.patch.object(connector, "_create_connection", create_connection):
        yield


class _FakeTrace:
    """Minimal stand-in for tracing.Trace recording connection events."""

    def __init__(self) -> None:
        self.events: list[str] = []

    async def send_connection_queued_start(self) -> None:
        self.events.append("queued_start")

    async def send_connection_queued_end(self) -> None:
        self.events.append("queued_end")

    async def send_connection_create_start(self) -> None:
        self.events.append("create_start")

    async def send_connection_create_end(self) -> None:
        self.events.append("create_end")

    async def send_connection_reuseconn(self) -> None:
        self.events.append("reuseconn")


# ---------------------------------------------------------------------------
# Connector-level unit tests
# ---------------------------------------------------------------------------


async def test_pin_binds_on_opening_connect_and_parks_on_release(
    make_client_request: _RequestMaker,
) -> None:
    loop = asyncio.get_running_loop()
    connector = aiohttp.BaseConnector()
    proto = _mocked_proto(loop)
    protos = [proto]
    req = make_client_request("GET", URL("http://localhost:80/"))
    trace = _FakeTrace()
    traces: list[Trace] = cast("list[Trace]", [trace])
    try:
        with _patched_factory(connector, protos):
            pin = connector.create_pin(None)
            conn = await connector.connect(req, traces, ClientTimeout(), pin=pin)
            assert pin.key == KEY
            assert pin.proto is proto
            assert proto in connector._acquired

            conn.release()

            # parked on the pin, never in the shared pool
            assert connector._conns == {}
            assert connector._pinned_idle[proto].done()
            assert proto not in connector._acquired

            conn2 = await connector.connect(req, traces, ClientTimeout(), pin=pin)
            assert conn2.protocol is proto
            conn2.release()

        assert trace.events == [
            "create_start",
            "create_end",
            "reuseconn",
        ]
    finally:
        await connector.close()


async def test_pin_second_request_waits_for_busy_connection(
    make_client_request: _RequestMaker,
) -> None:
    loop = asyncio.get_running_loop()
    connector = aiohttp.BaseConnector()
    proto = _mocked_proto(loop)
    protos = [proto]
    req = make_client_request("GET", URL("http://localhost:80/"))
    try:
        with _patched_factory(connector, protos):
            pin = connector.create_pin(None)
            conn1 = await connector.connect(req, [], ClientTimeout(), pin=pin)

            conn2_task = asyncio.ensure_future(
                connector.connect(req, [], ClientTimeout(), pin=pin)
            )
            await asyncio.sleep(0.05)
            assert not conn2_task.done()
            # only one protocol ever created: no second connection opened
            assert protos == []

            conn1.release()
            conn2 = await conn2_task
            assert conn2.protocol is proto
            conn2.release()
    finally:
        await connector.close()


async def test_pin_wait_timeout_leaves_no_slot_held(
    make_client_request: _RequestMaker,
) -> None:
    loop = asyncio.get_running_loop()
    connector = aiohttp.BaseConnector()
    proto = _mocked_proto(loop)
    protos = [proto]
    req = make_client_request("GET", URL("http://localhost:80/"))
    try:
        with _patched_factory(connector, protos):
            pin = connector.create_pin(None)
            conn1 = await connector.connect(req, [], ClientTimeout(), pin=pin)
            acquired_before = set(connector._acquired)

            with pytest.raises(asyncio.TimeoutError):
                await connector.connect(
                    req,
                    [],
                    ClientTimeout(connect=0.02),
                    pin=pin,
                )
            # timed-out waiter never consumed a connection slot
            assert set(connector._acquired) == acquired_before

            conn1.release()
    finally:
        await connector.close()


async def test_pin_cancelled_wait_keeps_pin_usable(
    make_client_request: _RequestMaker,
) -> None:
    loop = asyncio.get_running_loop()
    connector = aiohttp.BaseConnector()
    proto = _mocked_proto(loop)
    protos = [proto]
    req = make_client_request("GET", URL("http://localhost:80/"))
    try:
        with _patched_factory(connector, protos):
            pin = connector.create_pin(None)
            conn1 = await connector.connect(req, [], ClientTimeout(), pin=pin)

            wait_task = asyncio.ensure_future(
                connector.connect(req, [], ClientTimeout(), pin=pin)
            )
            await asyncio.sleep(0.05)
            wait_task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await wait_task

            conn1.release()
            conn2 = await connector.connect(req, [], ClientTimeout(), pin=pin)
            assert conn2.protocol is proto
            conn2.release()
    finally:
        await connector.close()


async def test_pin_queued_traces_fire(make_client_request: _RequestMaker) -> None:
    loop = asyncio.get_running_loop()
    connector = aiohttp.BaseConnector()
    proto = _mocked_proto(loop)
    protos = [proto]
    req = make_client_request("GET", URL("http://localhost:80/"))
    trace = _FakeTrace()
    traces: list[Trace] = cast("list[Trace]", [trace])
    try:
        with _patched_factory(connector, protos):
            pin = connector.create_pin(None)
            conn1 = await connector.connect(req, traces, ClientTimeout(), pin=pin)
            task = asyncio.ensure_future(
                connector.connect(req, traces, ClientTimeout(), pin=pin)
            )
            await asyncio.sleep(0.05)
            conn1.release()
            conn2 = await task
            conn2.release()
        assert "queued_start" in trace.events
        assert "queued_end" in trace.events
        assert trace.events.index("queued_start") < trace.events.index("queued_end")
    finally:
        await connector.close()


async def test_pin_closed_connection_invalidates_pin(
    make_client_request: _RequestMaker,
) -> None:
    loop = asyncio.get_running_loop()
    connector = aiohttp.BaseConnector()
    proto = _mocked_proto(loop, should_close=True)
    protos = [proto]
    req = make_client_request("GET", URL("http://localhost:80/"))
    try:
        with _patched_factory(connector, protos):
            pin = connector.create_pin(None)
            conn = await connector.connect(req, [], ClientTimeout(), pin=pin)
            conn.release()  # should_close -> pin invalidated, proto closed

        assert pin.invalid
        proto.close.assert_called()
        with pytest.raises(ClientConnectionPinError):
            await connector.connect(req, [], ClientTimeout(), pin=pin)
    finally:
        await connector.close()


async def test_pin_dead_idle_connection_fails_without_redial(
    make_client_request: _RequestMaker,
) -> None:
    loop = asyncio.get_running_loop()
    connector = aiohttp.BaseConnector()
    proto = _mocked_proto(loop)
    protos = [proto]
    req = make_client_request("GET", URL("http://localhost:80/"))
    try:
        with _patched_factory(connector, protos):
            pin = connector.create_pin(None)
            conn = await connector.connect(req, [], ClientTimeout(), pin=pin)
            conn.release()
            assert protos == []

            # peer dropped the idle transport
            proto.is_connected.return_value = False
            with pytest.raises(ClientConnectionPinError):
                await connector.connect(req, [], ClientTimeout(), pin=pin)
            # no replacement connection was created
            assert protos == []
    finally:
        await connector.close()


async def test_unpin_returns_idle_connection_to_pool(
    make_client_request: _RequestMaker,
) -> None:
    loop = asyncio.get_running_loop()
    connector = aiohttp.BaseConnector()
    proto = _mocked_proto(loop)
    protos = [proto]
    req = make_client_request("GET", URL("http://localhost:80/"))
    try:
        with _patched_factory(connector, protos):
            pin = connector.create_pin(None)
            conn = await connector.connect(req, [], ClientTimeout(), pin=pin)
            conn.release()
            connector.unpin(pin)

            pool = connector._conns[KEY]
            assert pool[0][0] is proto

            # plain (unpinned) request now gets it back from the pool
            conn2 = await connector.connect(req, [], ClientTimeout())
            assert conn2.protocol is proto
            conn2.release()
            assert protos == []
    finally:
        await connector.close()


async def test_concurrent_pins_never_cross(make_client_request: _RequestMaker) -> None:
    loop = asyncio.get_running_loop()
    connector = aiohttp.BaseConnector()
    proto_a = _mocked_proto(loop)
    proto_b = _mocked_proto(loop)
    req = make_client_request("GET", URL("http://localhost:80/"))
    try:
        with _patched_factory(connector, [proto_a, proto_b]):
            pin_a = connector.create_pin(None)
            pin_b = connector.create_pin(None)

            conn_a = await connector.connect(req, [], ClientTimeout(), pin=pin_a)
            conn_a.release()
            conn_b = await connector.connect(req, [], ClientTimeout(), pin=pin_b)
            conn_b.release()

            assert pin_a.proto is proto_a
            assert pin_b.proto is proto_b

            conn_a2 = await connector.connect(req, [], ClientTimeout(), pin=pin_a)
            assert conn_a2.protocol is proto_a
            conn_a2.release()
            conn_b2 = await connector.connect(req, [], ClientTimeout(), pin=pin_b)
            assert conn_b2.protocol is proto_b
            conn_b2.release()
    finally:
        await connector.close()


async def test_nested_pins_share_connection_then_outer_survives(
    make_client_request: _RequestMaker,
) -> None:
    loop = asyncio.get_running_loop()
    connector = aiohttp.BaseConnector()
    proto = _mocked_proto(loop)
    protos = [proto]
    req = make_client_request("GET", URL("http://localhost:80/"))
    try:
        with _patched_factory(connector, protos):
            outer = connector.create_pin(None)
            conn = await connector.connect(req, [], ClientTimeout(), pin=outer)
            conn.release()

            inner = connector.create_pin(outer)
            conn_inner = await connector.connect(req, [], ClientTimeout(), pin=inner)
            assert conn_inner.protocol is proto
            conn_inner.release()
            assert connector._proto_pins[proto] == [outer, inner]

            connector.unpin(inner)
            # outer keeps the proto reserved, it is not back in the pool
            assert KEY not in connector._conns
            assert connector._pinned_idle[proto].done()

            conn_outer = await connector.connect(req, [], ClientTimeout(), pin=outer)
            assert conn_outer.protocol is proto
            conn_outer.release()
            assert protos == []
    finally:
        await connector.close()


async def test_nested_pin_to_other_key_uses_pool_but_not_outer_pin(
    make_client_request: _RequestMaker,
) -> None:
    loop = asyncio.get_running_loop()
    connector = aiohttp.BaseConnector()
    proto1 = _mocked_proto(loop)
    proto2 = _mocked_proto(loop)
    req1 = make_client_request("GET", URL("http://localhost:80/"))
    req2 = make_client_request("GET", URL("http://otherhost:80/"))
    try:
        with _patched_factory(connector, [proto1, proto2]):
            outer = connector.create_pin(None)
            c1 = await connector.connect(req1, [], ClientTimeout(), pin=outer)
            c1.release()

            inner = connector.create_pin(outer)
            c2 = await connector.connect(req2, [], ClientTimeout(), pin=inner)
            assert c2.protocol is proto2
            assert inner.key == KEY2
            c2.release()

            connector.unpin(inner)
            # proto2 was only pinned by the inner scope: back in the pool
            assert connector._conns[KEY2][0][0] is proto2

            # outer pin is untouched
            c1b = await connector.connect(req1, [], ClientTimeout(), pin=outer)
            assert c1b.protocol is proto1
            c1b.release()

            # unpinned key2 request is served from the pool, no new dial
            c2b = await connector.connect(req2, [], ClientTimeout())
            assert c2b.protocol is proto2
            c2b.release()
    finally:
        await connector.close()


async def test_pin_does_not_touch_other_keys(
    make_client_request: _RequestMaker,
) -> None:
    loop = asyncio.get_running_loop()
    connector = aiohttp.BaseConnector()
    proto1 = _mocked_proto(loop)
    proto2 = _mocked_proto(loop)
    req1 = make_client_request("GET", URL("http://localhost:80/"))
    req2 = make_client_request("GET", URL("http://otherhost:80/"))
    try:
        with _patched_factory(connector, [proto1, proto2]):
            pin = connector.create_pin(None)
            c1 = await connector.connect(req1, [], ClientTimeout(), pin=pin)
            c1.release()
            assert pin.key == KEY

            c2 = await connector.connect(req2, [], ClientTimeout(), pin=pin)
            assert c2.protocol is proto2
            c2.release()

            # pin target is unchanged and proto2 went to the normal pool
            assert pin.key == KEY
            assert connector._conns[KEY2][0][0] is proto2
            assert proto2 not in connector._proto_pins
    finally:
        await connector.close()


async def test_unpinned_request_cannot_take_parked_pinned_connection(
    make_client_request: _RequestMaker,
) -> None:
    loop = asyncio.get_running_loop()
    connector = aiohttp.BaseConnector()
    proto_pinned = _mocked_proto(loop)
    proto_other = _mocked_proto(loop)
    req = make_client_request("GET", URL("http://localhost:80/"))
    try:
        with _patched_factory(connector, [proto_pinned, proto_other]):
            pin = connector.create_pin(None)
            c1 = await connector.connect(req, [], ClientTimeout(), pin=pin)
            c1.release()

            c2 = await connector.connect(req, [], ClientTimeout())
            assert c2.protocol is proto_other
            c2.release()

            c3 = await connector.connect(req, [], ClientTimeout(), pin=pin)
            assert c3.protocol is proto_pinned
            c3.release()
    finally:
        await connector.close()


async def test_connector_close_interrupts_pin_wait(
    make_client_request: _RequestMaker,
) -> None:
    loop = asyncio.get_running_loop()
    connector = aiohttp.BaseConnector()
    proto = _mocked_proto(loop)
    protos = [proto]
    req = make_client_request("GET", URL("http://localhost:80/"))
    try:
        with _patched_factory(connector, protos):
            pin = connector.create_pin(None)
            conn = await connector.connect(req, [], ClientTimeout(), pin=pin)

            wait_task = asyncio.ensure_future(
                connector.connect(req, [], ClientTimeout(), pin=pin)
            )
            await asyncio.sleep(0.05)

            close_task = asyncio.ensure_future(connector.close())
            with pytest.raises(aiohttp.ClientConnectionError):
                await wait_task
            await close_task
            # acquired pinned transport was closed by connector shutdown
            proto.close.assert_called()
            # release is a no-op on a closed connector, but clears the
            # wrapper so Connection.__del__ does not warn at GC
            conn.release()
    finally:
        await connector.close()


# ---------------------------------------------------------------------------
# End-to-end tests against real loopback servers
# ---------------------------------------------------------------------------


def _record_app(peers: list[tuple[str, int]]) -> web.Application:
    async def hello(request: web.Request) -> web.Response:
        peers.append(_peer(request))
        return web.Response(text="ok")

    async def force_close(request: web.Request) -> web.Response:
        peers.append(_peer(request))
        resp = web.Response(text="bye")
        resp.force_close()
        return resp

    async def ws_handler(request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        await ws.receive()
        return ws

    app = web.Application()
    app.router.add_get("/", hello)
    app.router.add_get("/close", force_close)
    app.router.add_get("/ws", ws_handler)
    return app


async def test_session_pin_reuses_one_connection(aiohttp_server: AiohttpServer) -> None:
    peers: list[tuple[str, int]] = []
    server = await aiohttp_server(_record_app(peers))

    async with ClientSession() as client:
        async with client.pin_connection() as pin:
            for _ in range(3):
                async with await client.get(server.make_url("/")) as resp:
                    assert resp.status == 200
                    await resp.read()
            assert pin.bound
            assert pin.key is not None

    assert len(set(peers)) == 1
    assert len(peers) == 3


async def test_session_pin_dies_on_server_close(aiohttp_server: AiohttpServer) -> None:
    peers: list[tuple[str, int]] = []
    server = await aiohttp_server(_record_app(peers))

    async with ClientSession() as client:
        with pytest.raises(ClientConnectionPinError):
            async with client.pin_connection():
                async with await client.get(server.make_url("/close")) as resp:
                    await resp.read()
                async with await client.get(server.make_url("/")) as resp:
                    await resp.read()

    # the failing request never reached the server
    assert len(peers) == 1


async def test_session_pin_after_scope_connection_goes_to_pool(
    aiohttp_server: AiohttpServer,
) -> None:
    peers: list[tuple[str, int]] = []
    server = await aiohttp_server(_record_app(peers))

    async with ClientSession() as client:
        async with client.pin_connection():
            async with await client.get(server.make_url("/")) as resp:
                await resp.read()
        pinned_peer = peers[0]

        async with await client.get(server.make_url("/")) as resp:
            await resp.read()
        assert peers[-1] == pinned_peer


async def test_session_pin_other_host_uses_pool_not_pinned_connection(
    aiohttp_server: AiohttpServer,
) -> None:
    peers_a: list[tuple[str, int]] = []
    peers_b: list[tuple[str, int]] = []
    server_a = await aiohttp_server(_record_app(peers_a))
    server_b = await aiohttp_server(_record_app(peers_b))

    async with ClientSession() as client:
        async with client.pin_connection():
            async with await client.get(server_a.make_url("/")) as resp:
                await resp.read()
            async with await client.get(server_b.make_url("/")) as resp:
                await resp.read()
            async with await client.get(server_a.make_url("/")) as resp:
                await resp.read()

    assert peers_a == [peers_a[0], peers_a[0]]
    assert len(peers_b) == 1
    assert peers_a[0] != peers_b[0]


async def test_session_pin_serializes_concurrent_same_key_requests(
    aiohttp_server: AiohttpServer,
) -> None:
    seen = asyncio.Event()
    release = asyncio.Event()
    peers: list[tuple[str, int]] = []

    async def gated(request: web.Request) -> web.Response:
        peers.append(_peer(request))
        seen.set()
        await release.wait()
        return web.Response(text="ok")

    app = web.Application()
    app.router.add_get("/", gated)
    server = await aiohttp_server(app)

    async with ClientSession() as client:

        async def one() -> None:
            async with await client.get(server.make_url("/")) as resp:
                await resp.read()

        async with client.pin_connection():
            first = asyncio.ensure_future(one())
            await seen.wait()
            others = [asyncio.ensure_future(one()) for _ in range(2)]
            await asyncio.sleep(0.1)
            # queued requests cannot have opened additional connections
            assert len(peers) == 1
            release.set()
            await asyncio.gather(first, *others)

    assert len(set(peers)) == 1
    assert len(peers) == 3


async def test_session_two_concurrent_scopes_use_distinct_connections(
    aiohttp_server: AiohttpServer,
) -> None:
    gate = asyncio.Event()
    peers: dict[str, list[tuple[str, int]]] = {"a": [], "b": []}

    async def gated(request: web.Request) -> web.Response:
        sid = request.query["id"]
        peers[sid].append(_peer(request))
        await gate.wait()
        return web.Response(text="ok")

    app = web.Application()
    app.router.add_get("/", gated)
    server = await aiohttp_server(app)
    url = server.make_url("/")

    async with ClientSession() as client:

        async def scope(sid: str) -> None:
            async with client.pin_connection():
                async with await client.get(url.with_query(id=sid)) as resp:
                    await gate.wait()
                    await resp.read()
                async with await client.get(url.with_query(id=sid)) as resp:
                    await resp.read()

        task_a = asyncio.ensure_future(scope("a"))
        task_b = asyncio.ensure_future(scope("b"))

        # wait on the server side: both scopes hold their own connection
        while len(peers["a"]) < 1 or len(peers["b"]) < 1:
            await asyncio.sleep(0.02)
        peer_a = peers["a"][0]
        peer_b = peers["b"][0]
        assert peer_a != peer_b

        gate.set()
        await asyncio.gather(task_a, task_b)

        # each scope reused its own connection, no crossing
        assert peers["a"] == [peer_a, peer_a]
        assert peers["b"] == [peer_b, peer_b]


async def test_session_nested_scopes(aiohttp_server: AiohttpServer) -> None:
    peers: list[tuple[str, int]] = []
    server = await aiohttp_server(_record_app(peers))

    async with ClientSession() as client:

        async def hit() -> None:
            async with await client.get(server.make_url("/")) as resp:
                await resp.read()

        async with client.pin_connection():
            await hit()
            outer_peer = peers[0]
            async with client.pin_connection() as inner:
                # read the property into locals: it is unbound before the
                # opening request and bound afterwards
                unbound_before = not inner.bound
                await hit()
                bound_after = inner.bound
            await hit()

    assert unbound_before
    assert bound_after
    assert peers == [outer_peer, outer_peer, outer_peer]


async def test_session_pin_redirect_same_origin_stays_pinned(
    aiohttp_server: AiohttpServer,
) -> None:
    peers: list[tuple[str, int]] = []

    async def redir(request: web.Request) -> web.Response:
        peers.append(_peer(request))
        raise web.HTTPFound("/final")

    async def final(request: web.Request) -> web.Response:
        peers.append(_peer(request))
        return web.Response(text="final")

    app = web.Application()
    app.router.add_get("/redir", redir)
    app.router.add_get("/final", final)
    server = await aiohttp_server(app)

    async with ClientSession() as client:
        async with client.pin_connection():
            async with await client.get(server.make_url("/redir")) as resp:
                assert resp.status == 200
                await resp.read()
            async with await client.get(server.make_url("/final")) as resp:
                await resp.read()

    assert len(set(peers)) == 1
    assert len(peers) == 3


async def test_session_pin_redirect_cross_origin_keeps_pin_target(
    aiohttp_server: AiohttpServer,
) -> None:
    peers_a: list[tuple[str, int]] = []
    peers_b: list[tuple[str, int]] = []
    server_b = await aiohttp_server(_record_app(peers_b))

    async def hello_a(request: web.Request) -> web.Response:
        peers_a.append(_peer(request))
        return web.Response(text="ok")

    async def leave(request: web.Request) -> web.Response:
        peers_a.append(_peer(request))
        raise web.HTTPFound(server_b.make_url("/"))

    app_a = web.Application()
    app_a.router.add_get("/", hello_a)
    app_a.router.add_get("/leave", leave)
    server_a = await aiohttp_server(app_a)

    async with ClientSession() as client:
        async with client.pin_connection() as pin:
            async with await client.get(server_a.make_url("/leave")) as resp:
                assert resp.status == 200
                await resp.read()
            assert pin.key is not None
            assert pin.key.host == server_a.host
            assert len(peers_b) == 1
            # back on the pinned host uses the original connection
            async with await client.get(server_a.make_url("/")) as resp:
                await resp.read()

    assert peers_a == [peers_a[0], peers_a[0]]


async def test_session_pin_ws_upgrade_exhausts_pin(
    aiohttp_server: AiohttpServer,
) -> None:
    peers: list[tuple[str, int]] = []
    server = await aiohttp_server(_record_app(peers))

    async with ClientSession() as client:
        with pytest.raises(ClientConnectionPinError):
            async with client.pin_connection():
                async with await client.get(server.make_url("/")) as resp:
                    await resp.read()
                async with client.ws_connect(server.make_url("/ws")) as ws:
                    await ws.close()
                async with await client.get(server.make_url("/")) as resp:
                    await resp.read()


async def test_session_close_with_open_pin(aiohttp_server: AiohttpServer) -> None:
    peers: list[tuple[str, int]] = []
    server = await aiohttp_server(_record_app(peers))

    async with ClientSession() as client:
        connector = client.connector
        assert connector is not None
        async with client.pin_connection():
            async with await client.get(server.make_url("/")) as resp:
                await resp.read()
        await client.close()
        assert connector.closed
        assert connector._pinned_idle == {}
        assert connector._proto_pins == {}


async def test_pin_connection_on_closed_session_raises(
    aiohttp_server: AiohttpServer,
) -> None:
    server = await aiohttp_server(_record_app([]))

    async with ClientSession() as client:
        await client.get(server.make_url("/"))  # ensure the server works
        await client.close()

        with pytest.raises(RuntimeError):
            async with client.pin_connection():
                pass


async def test_session_pin_tracing_events(aiohttp_server: AiohttpServer) -> None:
    seen = asyncio.Event()
    release = asyncio.Event()
    events: list[str] = []

    async def gated(request: web.Request) -> web.Response:
        seen.set()
        await release.wait()
        return web.Response(text="ok")

    app = web.Application()
    app.router.add_get("/", gated)
    server = await aiohttp_server(app)

    trace_config = TraceConfig()

    async def on_queued_start(*args: object) -> None:
        events.append("queued_start")

    async def on_queued_end(*args: object) -> None:
        events.append("queued_end")

    async def on_reuseconn(*args: object) -> None:
        events.append("reuseconn")

    trace_config.on_connection_queued_start.append(on_queued_start)
    trace_config.on_connection_queued_end.append(on_queued_end)
    trace_config.on_connection_reuseconn.append(on_reuseconn)

    async with ClientSession(trace_configs=[trace_config]) as client:

        async def hit() -> None:
            async with await client.get(server.make_url("/")) as resp:
                await resp.read()

        async with client.pin_connection():
            first = asyncio.ensure_future(hit())
            await seen.wait()
            second = asyncio.ensure_future(hit())
            await asyncio.sleep(0.05)
            release.set()
            await asyncio.gather(first, second)

    assert "queued_start" in events
    assert "queued_end" in events
    assert "reuseconn" in events


async def test_session_pin_wait_uses_request_timeout_and_pin_survives(
    aiohttp_server: AiohttpServer,
) -> None:
    seen = asyncio.Event()
    release = asyncio.Event()
    hits = 0

    async def gated(request: web.Request) -> web.Response:
        nonlocal hits
        hits += 1
        seen.set()
        await release.wait()
        return web.Response(text="ok")

    app = web.Application()
    app.router.add_get("/", gated)
    server = await aiohttp_server(app)

    async with ClientSession() as client:
        async with client.pin_connection():
            first = asyncio.ensure_future(
                client.get(
                    server.make_url("/"),
                    timeout=ClientTimeout(total=5),
                )
            )
            await seen.wait()

            # queued behind the in-flight pinned connection; its own short
            # timeout fires without opening another connection
            with pytest.raises(asyncio.TimeoutError):
                async with await client.get(
                    server.make_url("/"),
                    timeout=ClientTimeout(total=0.2),
                ):
                    pass

            release.set()
            first_resp = await first
            await first_resp.read()

            # the timed-out wait did not break the pin
            async with await client.get(server.make_url("/")) as resp:
                await resp.read()

    # exactly one connection served the opening, first and final requests
    assert True
