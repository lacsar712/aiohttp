"""Tests for the task-local outbound egress profile stack.

The stack is implemented in ``aiohttp.egress`` and consumed by
``ClientSession._request``, ``ClientSession._ws_connect`` and the
``BaseConnector`` connection lending paths.  Tests assert observable
outcomes (next hop not sent, jar not mutated, headers absent, timeout
raised, connection not lent / not pooled), timing and cross-task /
cross-session isolation.
"""

import asyncio
import contextlib
import platform
import socket
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator
from typing import Any

import pytest

import aiohttp
from aiohttp import hdrs, web
from aiohttp.client import ClientTimeout
from aiohttp.cookiejar import CookieJar, DummyCookieJar
from aiohttp.egress import (
    ConnectorAdmit,
    CookieSurface,
    EgressProfile,
    RedirectSurface,
    TimeoutOverlay,
    current_egress_view,
    effective_egress,
    egress_profile,
    pop_egress_profile,
    push_egress_profile,
    resolve_request_timeout,
)
from aiohttp.tracing import TraceConfig

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def unix_server(
    event_loop: asyncio.AbstractEventLoop, unix_sockname: str
) -> Iterator[Callable[[web.Application], Awaitable[None]]]:
    runners: list[web.AppRunner] = []

    async def go(app: web.Application) -> None:
        runner = web.AppRunner(app)
        runners.append(runner)
        await runner.setup()
        site = web.UnixSite(runner, unix_sockname)
        await site.start()

    yield go

    for runner in runners:
        event_loop.run_until_complete(runner.cleanup())


@pytest.fixture
async def named_pipe_server(
    pipe_name: str,
) -> AsyncIterator[Callable[[web.Application], Awaitable[None]]]:
    runners: list[web.AppRunner] = []

    async def go(app: web.Application) -> None:
        runner = web.AppRunner(app)
        runners.append(runner)
        await runner.setup()
        site = web.NamedPipeSite(runner, pipe_name)
        await site.start()

    yield go

    for runner in runners:
        await runner.cleanup()


def _redirect_chain_app(hits: list[str]) -> web.Application:
    app = web.Application()

    async def r1(request: web.Request) -> web.StreamResponse:
        hits.append(request.path)
        raise web.HTTPFound("/r2")

    async def r2(request: web.Request) -> web.StreamResponse:
        hits.append(request.path)
        raise web.HTTPFound("/final")

    async def final(request: web.Request) -> web.Response:
        hits.append(request.path)
        return web.Response(text="ok")

    app.router.add_get("/r1", r1)
    app.router.add_get("/r2", r2)
    app.router.add_get("/final", final)
    return app


def _cookie_app(seen_cookies: list[str]) -> web.Application:
    async def handler(request: web.Request) -> web.Response:
        seen_cookies.append(request.headers.get(hdrs.COOKIE, ""))
        resp = web.Response(text="ok")
        resp.set_cookie("new", "2")
        return resp

    app = web.Application()
    app.router.add_get("/cookie", handler)
    return app


@contextlib.asynccontextmanager
async def _silent_tcp_server() -> AsyncIterator[str]:
    """A TCP server that accepts, drains the request and never replies."""

    async def handle(
        reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        try:
            await reader.read(1)
            # Block until the client goes away so server teardown is instant.
            await reader.read()
        finally:
            writer.close()
            try:
                await writer.wait_closed()
            except (ConnectionError, RuntimeError, OSError):
                pass

    server = await asyncio.start_server(handle, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    try:
        yield f"http://127.0.0.1:{port}/"
    finally:
        server.close()
        await server.wait_closed()


def _ws_echo_app() -> web.Application:
    async def handler(request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        async for msg in ws:
            await ws.send_str(str(msg.data))
        return ws

    app = web.Application()
    app.router.add_get("/ws", handler)
    return app


# ---------------------------------------------------------------------------
# Stack mechanics and read-only views
# ---------------------------------------------------------------------------


async def test_empty_stack_is_keep() -> None:
    view = current_egress_view()
    assert view.depth == 0
    assert view.profile is None
    assert not view.active
    assert view.redirect_surface is RedirectSurface.KEEP
    assert view.cookie_surface is CookieSurface.KEEP
    assert view.timeout_overlay is TimeoutOverlay.KEEP
    assert view.connector_admit is ConnectorAdmit.KEEP


async def test_frame_is_popped_on_exception() -> None:
    with pytest.raises(RuntimeError, match="boom"):
        with egress_profile(redirect_surface=RedirectSurface.DENY_REDIRECT):
            assert current_egress_view().depth == 1
            raise RuntimeError("boom")
    assert current_egress_view().depth == 0


async def test_pop_rejects_foreign_stack_top() -> None:
    outer = EgressProfile(redirect_surface=RedirectSurface.DENY_REDIRECT)
    inner = EgressProfile(redirect_surface=RedirectSurface.CAP_ONE)
    outer_token = push_egress_profile(outer)
    inner_token = push_egress_profile(inner)
    # The inner frame cannot pop the outer one.
    with pytest.raises(RuntimeError):
        pop_egress_profile(outer_token, outer)
    pop_egress_profile(inner_token, inner)
    pop_egress_profile(outer_token, outer)
    assert current_egress_view().depth == 0


async def test_nested_frames_restore_outer_on_pop() -> None:
    with egress_profile(redirect_surface=RedirectSurface.DENY_REDIRECT):
        assert current_egress_view().redirect_surface is RedirectSurface.DENY_REDIRECT
        with egress_profile(redirect_surface=RedirectSurface.CAP_ONE):
            assert current_egress_view().redirect_surface is RedirectSurface.CAP_ONE
        assert current_egress_view().redirect_surface is RedirectSurface.DENY_REDIRECT
    assert current_egress_view().profile is None


def test_fail_fast_requires_timeout() -> None:
    with pytest.raises(ValueError, match="fail_fast_timeout"):
        EgressProfile(timeout_overlay=TimeoutOverlay.FAIL_FAST)


def test_enum_dimensions_are_independent() -> None:
    profile = EgressProfile(
        redirect_surface=RedirectSurface.DENY_REDIRECT,
        cookie_surface=CookieSurface.FREEZE_JAR,
        timeout_overlay=TimeoutOverlay.FREEZE_BASE,
        connector_admit=ConnectorAdmit.REJECT_ACQUIRE,
        fail_fast_timeout=ClientTimeout(total=1),
    )
    assert profile.redirect_surface is RedirectSurface.DENY_REDIRECT
    assert profile.cookie_surface is CookieSurface.FREEZE_JAR
    assert profile.timeout_overlay is TimeoutOverlay.FREEZE_BASE
    assert profile.connector_admit is ConnectorAdmit.REJECT_ACQUIRE


async def test_effective_view_merges_session_runtime() -> None:
    session = aiohttp.ClientSession(timeout=ClientTimeout(total=11))
    try:
        with egress_profile(
            redirect_surface=RedirectSurface.CAP_ONE,
            connector_admit=ConnectorAdmit.FORCE_DETACH,
        ):
            eff = effective_egress(session, max_redirects=7)
            assert eff.max_redirects == 1
            assert eff.connector_admit is ConnectorAdmit.FORCE_DETACH
            assert eff.raise_for_status == session.raise_for_status
            # The view observes the actual session objects, never copies.
            assert eff.cookie_jar is session.cookie_jar
            assert eff.connector is session.connector
            assert eff.timeout is session.timeout
            assert eff.depth == 1
    finally:
        await session.close()


async def test_fail_fast_timeout_merge_is_shorter_per_field() -> None:
    baseline = ClientTimeout(total=10, sock_read=8)
    with egress_profile(
        timeout_overlay=TimeoutOverlay.FAIL_FAST,
        fail_fast_timeout=ClientTimeout(total=0.3),
    ):
        merged = resolve_request_timeout(ClientTimeout(total=9, sock_read=8), baseline)
        assert merged.total == pytest.approx(0.3)
        assert merged.sock_read == pytest.approx(0.3)

    with egress_profile(
        timeout_overlay=TimeoutOverlay.FAIL_FAST,
        fail_fast_timeout=ClientTimeout(total=10, sock_read=0.4),
    ):
        merged = resolve_request_timeout(baseline, baseline)
        assert merged.total == pytest.approx(10)
        assert merged.sock_read == pytest.approx(0.4)

    # FREEZE_BASE keeps the session baseline even with a shorter call timeout.
    with egress_profile(timeout_overlay=TimeoutOverlay.FREEZE_BASE):
        merged = resolve_request_timeout(
            ClientTimeout(total=0.2), ClientTimeout(total=12)
        )
        assert merged.total == pytest.approx(12)


# ---------------------------------------------------------------------------
# redirect_surface
# ---------------------------------------------------------------------------


async def test_deny_redirect_does_not_send_next_hop(
    aiohttp_server: Callable[[web.Application], Awaitable[Any]],
) -> None:
    hits: list[str] = []
    server = await aiohttp_server(_redirect_chain_app(hits))
    async with aiohttp.ClientSession() as session:
        with egress_profile(redirect_surface=RedirectSurface.DENY_REDIRECT):
            with pytest.raises(aiohttp.EgressRedirectDenied):
                await session.get(server.make_url("/r1"))
    assert hits == ["/r1"]


async def test_cap_one_allows_one_hop_then_fails(
    aiohttp_server: Callable[[web.Application], Awaitable[Any]],
) -> None:
    hits: list[str] = []
    server = await aiohttp_server(_redirect_chain_app(hits))
    async with aiohttp.ClientSession() as session:
        with egress_profile(redirect_surface=RedirectSurface.CAP_ONE):
            with pytest.raises(aiohttp.EgressRedirectCapExceeded):
                await session.get(server.make_url("/r1"))
    assert hits == ["/r1", "/r2"]


async def test_redirect_keep_frame_follows_as_usual(
    aiohttp_server: Callable[[web.Application], Awaitable[Any]],
) -> None:
    hits: list[str] = []
    server = await aiohttp_server(_redirect_chain_app(hits))
    async with aiohttp.ClientSession() as session:
        with egress_profile(redirect_surface=RedirectSurface.KEEP):
            async with session.get(server.make_url("/r1")) as resp:
                assert resp.status == 200
                await resp.read()
    assert hits == ["/r1", "/r2", "/final"]


async def test_behavior_restored_after_frame_pop(
    aiohttp_server: Callable[[web.Application], Awaitable[Any]],
) -> None:
    hits: list[str] = []
    server = await aiohttp_server(_redirect_chain_app(hits))
    async with aiohttp.ClientSession() as session:
        with egress_profile(redirect_surface=RedirectSurface.DENY_REDIRECT):
            with pytest.raises(aiohttp.EgressRedirectDenied):
                await session.get(server.make_url("/r1"))
        # Empty stack again: redirects follow with session semantics.
        async with session.get(server.make_url("/r1")) as resp:
            assert resp.status == 200
            await resp.read()
    assert hits == ["/r1", "/r1", "/r2", "/final"]


async def test_module_level_request_obeys_deny(
    aiohttp_server: Callable[[web.Application], Awaitable[Any]],
) -> None:
    hits: list[str] = []
    server = await aiohttp_server(_redirect_chain_app(hits))
    with egress_profile(redirect_surface=RedirectSurface.DENY_REDIRECT):
        with pytest.raises(aiohttp.EgressRedirectDenied):
            async with aiohttp.request("GET", server.make_url("/r1")):
                pass
    assert hits == ["/r1"]


async def test_redirect_pushed_after_first_hop_is_recomputed(
    aiohttp_server: Callable[[web.Application], Awaitable[Any]],
) -> None:
    """Pushing the frame from the redirect callback (after first-hop bytes)
    must be re-read at the redirect decision, not ignored via an entry
    snapshot.
    """
    hits: list[str] = []
    state: dict[str, Any] = {}
    profile = EgressProfile(redirect_surface=RedirectSurface.DENY_REDIRECT)

    async def on_redirect(
        session: aiohttp.ClientSession,
        ctx: Any,
        params: Any,
    ) -> None:
        state["token"] = push_egress_profile(profile)

    trace = TraceConfig()
    trace.on_request_redirect.append(on_redirect)
    server = await aiohttp_server(_redirect_chain_app(hits))
    async with aiohttp.ClientSession(trace_configs=[trace]) as session:
        try:
            with pytest.raises(aiohttp.EgressRedirectDenied):
                await session.get(server.make_url("/r1"))
        finally:
            if "token" in state:
                pop_egress_profile(state["token"], profile)
    assert hits == ["/r1"]


# ---------------------------------------------------------------------------
# cookie_surface
# ---------------------------------------------------------------------------


async def test_freeze_jar_blocks_writes_but_keeps_outbound(
    aiohttp_server: Callable[[web.Application], Awaitable[Any]],
) -> None:
    seen: list[str] = []
    server = await aiohttp_server(_cookie_app(seen))
    url = server.make_url("/cookie")
    # The default jar rejects cookies for IP-literal hosts; unsafe=True is
    # only about accepting them, not about the cookie surface under test.
    async with aiohttp.ClientSession(cookie_jar=CookieJar(unsafe=True)) as session:
        session.cookie_jar.update_cookies({"old": "1"}, url)
        with egress_profile(cookie_surface=CookieSurface.FREEZE_JAR):
            async with session.get(url) as resp:
                assert resp.status == 200
                await resp.read()
            sent = session.cookie_jar.filter_cookies(url)
            assert "old" in sent
            assert "new" not in sent
        # After pop, Set-Cookie is processed again.
        async with session.get(url) as resp:
            await resp.read()
        assert "new" in session.cookie_jar.filter_cookies(url)
    assert "old=1" in seen[0]


async def test_omit_outbound_strips_jar_cookies_but_writes_responses(
    aiohttp_server: Callable[[web.Application], Awaitable[Any]],
) -> None:
    seen: list[str] = []
    server = await aiohttp_server(_cookie_app(seen))
    url = server.make_url("/cookie")
    # The default jar rejects cookies for IP-literal hosts; unsafe=True is
    # only about accepting them, not about the cookie surface under test.
    async with aiohttp.ClientSession(cookie_jar=CookieJar(unsafe=True)) as session:
        session.cookie_jar.update_cookies({"old": "1"}, url)
        with egress_profile(cookie_surface=CookieSurface.OMIT_OUTBOUND):
            async with session.get(url) as resp:
                await resp.read()
            assert seen[-1] == ""
            assert "new" in session.cookie_jar.filter_cookies(url)
            # Explicit per-request cookies are not jar cookies: still sent.
            async with session.get(url, cookies={"explicit": "3"}) as resp:
                await resp.read()
            assert "explicit=3" in seen[-1]
            assert "old=1" not in seen[-1]


async def test_dummy_cookie_jar_surface_fork(
    aiohttp_server: Callable[[web.Application], Awaitable[Any]],
) -> None:
    seen: list[str] = []
    server = await aiohttp_server(_cookie_app(seen))
    url = server.make_url("/cookie")
    async with aiohttp.ClientSession(cookie_jar=DummyCookieJar()) as session:
        assert len(session.cookie_jar) == 0
        for surface in (
            CookieSurface.FREEZE_JAR,
            CookieSurface.OMIT_OUTBOUND,
        ):
            with egress_profile(cookie_surface=surface):
                async with session.get(url) as resp:
                    await resp.read()
            assert len(session.cookie_jar) == 0
            assert seen[-1] == ""


async def test_freeze_pushed_after_first_hop_applies_to_next_hop(
    aiohttp_server: Callable[[web.Application], Awaitable[Any]],
) -> None:
    state: dict[str, Any] = {}
    profile = EgressProfile(cookie_surface=CookieSurface.FREEZE_JAR)

    async def r1(request: web.Request) -> web.StreamResponse:
        resp = web.Response(status=302, headers={hdrs.LOCATION: "/final"})
        resp.set_cookie("c1", "1")
        return resp

    async def final(request: web.Request) -> web.Response:
        resp = web.Response(text="ok")
        resp.set_cookie("c2", "2")
        return resp

    async def on_redirect(
        session: aiohttp.ClientSession, ctx: Any, params: Any
    ) -> None:
        state["token"] = push_egress_profile(profile)

    trace = TraceConfig()
    trace.on_request_redirect.append(on_redirect)
    app = web.Application()
    app.router.add_get("/r1", r1)
    app.router.add_get("/final", final)
    server = await aiohttp_server(app)
    async with aiohttp.ClientSession(
        cookie_jar=CookieJar(unsafe=True), trace_configs=[trace]
    ) as session:
        try:
            async with session.get(server.make_url("/r1")) as resp:
                assert resp.status == 200
                await resp.read()
        finally:
            if "token" in state:
                pop_egress_profile(state["token"], profile)
        cookies = session.cookie_jar.filter_cookies(server.make_url("/final"))
        assert "c1" in cookies
        assert "c2" not in cookies


# ---------------------------------------------------------------------------
# timeout_overlay
# ---------------------------------------------------------------------------


async def test_fail_fast_total_timeout_fires() -> None:
    async with aiohttp.ClientSession(timeout=ClientTimeout(total=30)) as session:
        async with _silent_tcp_server() as url:
            loop = asyncio.get_running_loop()
            start = loop.time()
            with egress_profile(
                timeout_overlay=TimeoutOverlay.FAIL_FAST,
                fail_fast_timeout=ClientTimeout(total=0.4),
            ):
                with pytest.raises(asyncio.TimeoutError):
                    await session.get(url)
            assert loop.time() - start < 5


async def test_fail_fast_sock_read_fires(
    aiohttp_server: Callable[[web.Application], Awaitable[Any]],
) -> None:
    async def slow_body(request: web.Request) -> web.StreamResponse:
        resp = web.StreamResponse()
        await resp.prepare(request)
        await asyncio.sleep(1)
        await resp.write(b"x")
        return resp

    app = web.Application()
    app.router.add_get("/slow", slow_body)
    server = await aiohttp_server(app)
    async with aiohttp.ClientSession(timeout=ClientTimeout(total=30)) as session:
        loop = asyncio.get_running_loop()
        start = loop.time()
        with egress_profile(
            timeout_overlay=TimeoutOverlay.FAIL_FAST,
            fail_fast_timeout=ClientTimeout(total=10, sock_read=0.3),
        ):
            async with session.get(server.make_url("/slow")) as resp:
                with pytest.raises(asyncio.TimeoutError):
                    await resp.read()
        assert loop.time() - start < 5


async def test_fail_fast_beats_longer_per_call_timeout() -> None:
    async with aiohttp.ClientSession(timeout=ClientTimeout(total=30)) as session:
        async with _silent_tcp_server() as url:
            with egress_profile(
                timeout_overlay=TimeoutOverlay.FAIL_FAST,
                fail_fast_timeout=ClientTimeout(total=0.4),
            ):
                with pytest.raises(asyncio.TimeoutError):
                    await session.get(url, timeout=ClientTimeout(total=20))


async def test_freeze_base_ignores_shorter_per_call_timeout(
    aiohttp_server: Callable[[web.Application], Awaitable[Any]],
) -> None:
    async def slow(request: web.Request) -> web.Response:
        await asyncio.sleep(0.7)
        return web.Response(text="late")

    app = web.Application()
    app.router.add_get("/slow", slow)
    server = await aiohttp_server(app)
    async with aiohttp.ClientSession(timeout=ClientTimeout(total=5)) as session:
        loop = asyncio.get_running_loop()
        start = loop.time()
        with egress_profile(timeout_overlay=TimeoutOverlay.FREEZE_BASE):
            async with session.get(
                server.make_url("/slow"), timeout=ClientTimeout(total=0.2)
            ) as resp:
                assert resp.status == 200
                await resp.read()
        assert loop.time() - start > 0.5


async def test_freeze_base_cancelled_task_still_fails(
    aiohttp_server: Callable[[web.Application], Awaitable[Any]],
) -> None:
    async def slow(request: web.Request) -> web.Response:
        await asyncio.sleep(5)
        return web.Response(text="late")

    app = web.Application()
    app.router.add_get("/slow", slow)
    server = await aiohttp_server(app)
    async with aiohttp.ClientSession(timeout=ClientTimeout(total=30)) as session:

        async def call() -> None:
            with egress_profile(timeout_overlay=TimeoutOverlay.FREEZE_BASE):
                async with session.get(
                    server.make_url("/slow"), timeout=ClientTimeout(total=0.1)
                ) as resp:
                    await resp.read()

        task = asyncio.create_task(call())
        await asyncio.sleep(0.3)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task


async def test_raise_for_status_still_works_under_fail_fast(
    aiohttp_server: Callable[[web.Application], Awaitable[Any]],
) -> None:
    async def boom(request: web.Request) -> web.Response:
        return web.Response(status=500, text="err")

    app = web.Application()
    app.router.add_get("/boom", boom)
    server = await aiohttp_server(app)
    async with aiohttp.ClientSession(
        timeout=ClientTimeout(total=30), raise_for_status=True
    ) as session:
        with egress_profile(
            timeout_overlay=TimeoutOverlay.FAIL_FAST,
            fail_fast_timeout=ClientTimeout(total=10),
        ):
            with pytest.raises(aiohttp.ClientResponseError) as exc_info:
                await session.get(server.make_url("/boom"))
            assert exc_info.value.status == 500


# ---------------------------------------------------------------------------
# connector_admit
# ---------------------------------------------------------------------------


async def test_reject_acquire_request_lends_nothing(
    aiohttp_server: Callable[[web.Application], Awaitable[Any]],
) -> None:
    hits: list[str] = []
    app = web.Application()

    async def handler(request: web.Request) -> web.Response:
        hits.append(request.path)
        return web.Response()

    app.router.add_get("/", handler)
    server = await aiohttp_server(app)
    connector = aiohttp.TCPConnector()
    async with aiohttp.ClientSession(connector=connector) as session:
        with egress_profile(connector_admit=ConnectorAdmit.REJECT_ACQUIRE):
            with pytest.raises(aiohttp.EgressAcquireDenied):
                await session.get(server.make_url("/"))
        assert connector._acquired == set()
    assert hits == []
    await connector.close()


async def test_reject_acquire_ws_connect(
    aiohttp_server: Callable[[web.Application], Awaitable[Any]],
) -> None:
    hits: list[str] = []

    async def handler(request: web.Request) -> web.WebSocketResponse:
        hits.append("ws")
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        return ws

    app = web.Application()
    app.router.add_get("/ws", handler)
    server = await aiohttp_server(app)
    async with aiohttp.ClientSession() as session:
        with egress_profile(connector_admit=ConnectorAdmit.REJECT_ACQUIRE):
            with pytest.raises(aiohttp.EgressAcquireDenied):
                await session.ws_connect(server.make_url("/ws"))
    assert hits == []


async def test_ws_connect_not_affected_by_deny_redirect(
    aiohttp_server: Callable[[web.Application], Awaitable[Any]],
) -> None:
    server = await aiohttp_server(_ws_echo_app())
    async with aiohttp.ClientSession() as session:
        with egress_profile(redirect_surface=RedirectSurface.DENY_REDIRECT):
            ws = await session.ws_connect(server.make_url("/ws"))
            await ws.send_str("ping")
            msg = await ws.receive()
            assert msg.data == "ping"
            await ws.close()


async def test_ws_connect_fail_fast_total(
    aiohttp_server: Callable[[web.Application], Awaitable[Any]],
) -> None:
    async with aiohttp.ClientSession(timeout=ClientTimeout(total=30)) as session:
        async with _silent_tcp_server() as url:
            with egress_profile(
                timeout_overlay=TimeoutOverlay.FAIL_FAST,
                fail_fast_timeout=ClientTimeout(total=0.4),
            ):
                with pytest.raises(asyncio.TimeoutError):
                    await session.ws_connect(url)


async def test_ws_reject_pushed_during_handshake_is_denied(
    aiohttp_server: Callable[[web.Application], Awaitable[Any]],
) -> None:
    state: dict[str, Any] = {}
    profile = EgressProfile(connector_admit=ConnectorAdmit.REJECT_ACQUIRE)

    async def on_headers_sent(
        session: aiohttp.ClientSession, ctx: Any, params: Any
    ) -> None:
        # Borrow already happened; push after the first handshake bytes.
        state["token"] = push_egress_profile(profile)

    trace = TraceConfig()
    trace.on_request_headers_sent.append(on_headers_sent)
    server = await aiohttp_server(_ws_echo_app())
    async with aiohttp.ClientSession(trace_configs=[trace]) as session:
        try:
            with pytest.raises(aiohttp.EgressAcquireDenied):
                await session.ws_connect(server.make_url("/ws"))
        finally:
            if "token" in state:
                pop_egress_profile(state["token"], profile)


async def test_force_detach_closes_tcp_connection_on_release(
    aiohttp_server: Callable[[web.Application], Awaitable[Any]],
) -> None:
    transports: list[object] = []

    async def handler(request: web.Request) -> web.Response:
        transports.append(request.transport)
        return web.Response(text="ok")

    app = web.Application()
    app.router.add_get("/", handler)
    server = await aiohttp_server(app)
    connector = aiohttp.TCPConnector()
    async with aiohttp.ClientSession(connector=connector) as session:
        with egress_profile(connector_admit=ConnectorAdmit.FORCE_DETACH):
            for _ in range(2):
                async with session.get(server.make_url("/")) as resp:
                    await resp.read()
            assert not connector._conns
        assert len({id(t) for t in transports}) == 2
        # Outside the frame: the next request creates and pools one conn.
        async with session.get(server.make_url("/")) as resp:
            await resp.read()
        assert len({id(t) for t in transports}) == 3
        async with session.get(server.make_url("/")) as resp:
            await resp.read()
        assert len({id(t) for t in transports}) == 3
    await connector.close()


async def test_force_detach_marks_borrow_release_outside_frame(
    aiohttp_server: Callable[[web.Application], Awaitable[Any]],
) -> None:
    async def handler(request: web.Request) -> web.Response:
        return web.Response(text="ok")

    app = web.Application()
    app.router.add_get("/", handler)
    server = await aiohttp_server(app)
    connector = aiohttp.TCPConnector()
    async with aiohttp.ClientSession(connector=connector) as session:
        with egress_profile(connector_admit=ConnectorAdmit.FORCE_DETACH):
            resp = await session.get(server.make_url("/"))
            await resp.read()
        # Released after the frame popped: the borrowed conn is still tainted.
        resp.release()
        assert not connector._conns
    await connector.close()


async def test_reject_pushed_during_limit_wait_is_denied(
    aiohttp_server: Callable[[web.Application], Awaitable[Any]],
) -> None:
    hold = asyncio.Event()
    hits: list[str] = []

    async def hold_handler(request: web.Request) -> web.StreamResponse:
        resp = web.StreamResponse()
        await resp.prepare(request)
        await hold.wait()
        return resp

    async def final_handler(request: web.Request) -> web.Response:
        hits.append(request.path)
        return web.Response(text="ok")

    app = web.Application()
    app.router.add_get("/hold", hold_handler)
    app.router.add_get("/final", final_handler)
    server = await aiohttp_server(app)

    state: dict[str, Any] = {}
    profile = EgressProfile(connector_admit=ConnectorAdmit.REJECT_ACQUIRE)

    async def on_queued_start(
        session: aiohttp.ClientSession, ctx: Any, params: Any
    ) -> None:
        state["token"] = push_egress_profile(profile)

    trace = TraceConfig()
    trace.on_connection_queued_start.append(on_queued_start)
    connector = aiohttp.TCPConnector(limit=1)
    async with aiohttp.ClientSession(
        connector=connector, trace_configs=[trace]
    ) as session:
        holder = await session.get(server.make_url("/hold"))
        try:

            async def blocked() -> None:
                try:
                    await session.get(server.make_url("/final"))
                finally:
                    if "token" in state:
                        pop_egress_profile(state["token"], profile)

            task = asyncio.create_task(blocked())
            await asyncio.sleep(0.5)
            holder.release()
            with pytest.raises(aiohttp.EgressAcquireDenied):
                await task
            assert hits == []
        finally:
            hold.set()
            holder.close()
    await connector.close()


@pytest.mark.skipif(not hasattr(socket, "AF_UNIX"), reason="requires UNIX sockets")
async def test_reject_acquire_unix_connector(
    unix_server: Callable[[web.Application], Awaitable[None]],
    unix_sockname: str,
) -> None:
    hits: list[str] = []

    async def handler(request: web.Request) -> web.Response:
        hits.append(request.path)
        return web.Response()

    app = web.Application()
    app.router.add_get("/", handler)
    await unix_server(app)

    connector = aiohttp.UnixConnector(unix_sockname)
    async with aiohttp.ClientSession(connector=connector) as session:
        with egress_profile(connector_admit=ConnectorAdmit.REJECT_ACQUIRE):
            with pytest.raises(aiohttp.EgressAcquireDenied):
                await session.get("http://127.0.0.1/")
    assert hits == []
    await connector.close()


@pytest.mark.skipif(
    platform.system() != "Windows", reason="Proactor Event loop present only in Windows"
)
@pytest.mark.asyncio(loop_factories=("proactor",))
async def test_reject_acquire_named_pipe_connector(
    named_pipe_server: Callable[[web.Application], Awaitable[None]],
    pipe_name: str,
) -> None:
    hits: list[str] = []

    async def handler(request: web.Request) -> web.Response:
        hits.append(request.path)
        return web.Response()

    app = web.Application()
    app.router.add_get("/", handler)
    await named_pipe_server(app)

    connector = aiohttp.NamedPipeConnector(pipe_name)
    async with aiohttp.ClientSession(connector=connector) as session:
        with egress_profile(connector_admit=ConnectorAdmit.REJECT_ACQUIRE):
            with pytest.raises(aiohttp.EgressAcquireDenied):
                await session.get("http://this-does-not-matter.com/")
    assert hits == []
    await connector.close()


# ---------------------------------------------------------------------------
# Task / session isolation
# ---------------------------------------------------------------------------


async def test_two_tasks_redirect_surfaces_do_not_cross(
    aiohttp_server: Callable[[web.Application], Awaitable[Any]],
) -> None:
    hits: list[str] = []
    server = await aiohttp_server(_redirect_chain_app(hits))
    url = server.make_url("/r1")
    async with aiohttp.ClientSession() as session:

        async def denied() -> None:
            with egress_profile(redirect_surface=RedirectSurface.DENY_REDIRECT):
                with pytest.raises(aiohttp.EgressRedirectDenied):
                    await session.get(url)

        async def following() -> None:
            async with session.get(url) as resp:
                assert resp.status == 200
                await resp.read()

        await asyncio.gather(denied(), following())
    assert hits.count("/r1") == 2
    assert hits.count("/r2") == 1
    assert hits.count("/final") == 1


async def test_two_tasks_connector_admit_does_not_cross(
    aiohttp_server: Callable[[web.Application], Awaitable[Any]],
) -> None:
    hits: list[str] = []

    async def handler(request: web.Request) -> web.Response:
        hits.append(request.path)
        return web.Response(text="ok")

    app = web.Application()
    app.router.add_get("/", handler)
    server = await aiohttp_server(app)
    url = server.make_url("/")
    async with aiohttp.ClientSession() as session:

        async def rejected() -> None:
            with egress_profile(connector_admit=ConnectorAdmit.REJECT_ACQUIRE):
                with pytest.raises(aiohttp.EgressAcquireDenied):
                    await session.get(url)

        async def admitted() -> None:
            async with session.get(url) as resp:
                assert resp.status == 200
                await resp.read()

        await asyncio.gather(rejected(), admitted())
    assert hits == ["/"]


async def test_two_sessions_same_task_nested_frames_do_not_mix(
    aiohttp_server: Callable[[web.Application], Awaitable[Any]],
) -> None:
    hits: list[str] = []
    server = await aiohttp_server(_redirect_chain_app(hits))
    url = server.make_url("/r1")
    async with (
        aiohttp.ClientSession() as session_a,
        aiohttp.ClientSession() as session_b,
    ):
        with egress_profile(redirect_surface=RedirectSurface.DENY_REDIRECT):
            assert current_egress_view().depth == 1
            with pytest.raises(aiohttp.EgressRedirectDenied):
                await session_a.get(url)
            with egress_profile(redirect_surface=RedirectSurface.KEEP):
                assert current_egress_view().depth == 2
                async with session_b.get(url) as resp:
                    assert resp.status == 200
                    await resp.read()
            assert current_egress_view().depth == 1
            # Session A still governed by the outer deny frame.
            with pytest.raises(aiohttp.EgressRedirectDenied):
                await session_a.get(url)
        assert current_egress_view().depth == 0


async def test_spawned_task_does_not_inherit_live_frame_mutability() -> None:
    with egress_profile(redirect_surface=RedirectSurface.DENY_REDIRECT):
        views: list[int] = []

        async def child() -> None:
            # Context is copied: child sees the same frame depth but has its
            # own stack; pushing here never mutates the parent stack.
            views.append(current_egress_view().depth)
            child_frame = EgressProfile(redirect_surface=RedirectSurface.CAP_ONE)
            token = push_egress_profile(child_frame)
            try:
                views.append(current_egress_view().depth)
            finally:
                pop_egress_profile(token, child_frame)

        await asyncio.create_task(child())
        assert views == [1, 2]
        assert current_egress_view().depth == 1
