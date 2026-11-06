"""Task-local outbound egress profile stack for the HTTP client.

The egress profile stack is a per-task stack (backed by
:class:`contextvars.ContextVar`) of :class:`EgressProfile` frames.  Each
frame governs four independent outbound dimensions of client calls that
run while the frame is on top of the stack::

    * ``redirect_surface`` -- whether redirects are followed
    * ``cookie_surface``   -- whether response cookies are written to the
                              jar and/or jar cookies are sent outbound
    * ``timeout_overlay``  -- how the effective request timeout is merged
    * ``connector_admit``  -- whether connections may be borrowed and how
                              they are returned to the connector

Frames never replace the :class:`~aiohttp.ClientSession`,
:class:`~aiohttp.BaseConnector` or cookie jar a call actually uses: they
only gate operations on the session / connector / jar involved in the
current call.  An empty stack is equivalent to no profile at all.
"""

from __future__ import annotations

import contextlib
import enum
from collections.abc import Callable, Iterator
from contextvars import ContextVar, Token
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from .client_exceptions import (
    EgressAcquireDenied,
    EgressRedirectCapExceeded,
    EgressRedirectDenied,
)
from .helpers import _SENTINEL, sentinel

if TYPE_CHECKING:
    from .abc import AbstractCookieJar
    from .client_reqrep import ClientTimeout
    from .connector import BaseConnector

__all__ = (
    "RedirectSurface",
    "CookieSurface",
    "TimeoutOverlay",
    "ConnectorAdmit",
    "EgressProfile",
    "EgressView",
    "EffectiveEgress",
    "egress_profile",
    "push_egress_profile",
    "pop_egress_profile",
    "current_egress_profile",
    "current_egress_view",
    "effective_egress",
    "resolve_request_timeout",
)


class RedirectSurface(enum.Enum):
    """Redirect policy of an egress frame.

    * ``KEEP`` -- follow per the call/session ``max_redirects``.
    * ``DENY_REDIRECT`` -- if a 3xx response requires following, raise
      :class:`~aiohttp.EgressRedirectDenied`; the next hop is never sent.
      Websocket handshakes are not governed by this dimension.
    * ``CAP_ONE`` -- follow at most one hop.  When a second hop would be
      required, raise :class:`~aiohttp.EgressRedirectCapExceeded`.
    """

    KEEP = "keep"
    DENY_REDIRECT = "deny_redirect"
    CAP_ONE = "cap_one"


class CookieSurface(enum.Enum):
    """Cookie jar policy of an egress frame.

    * ``KEEP`` -- responses update the jar, requests send jar cookies.
    * ``FREEZE_JAR`` -- responses never add cookies to the jar, while
      requests still carry cookies already present in the jar.
    * ``OMIT_OUTBOUND`` -- requests never carry jar cookies, while
      responses may still write cookies into the jar.
    """

    KEEP = "keep"
    FREEZE_JAR = "freeze_jar"
    OMIT_OUTBOUND = "omit_outbound"


class TimeoutOverlay(enum.Enum):
    """Timeout policy of an egress frame.

    * ``KEEP`` -- use the per-call timeout or, absent one, the session
      default :class:`~aiohttp.ClientTimeout`.
    * ``FAIL_FAST`` -- clamp the effective timeout to the (shorter)
      :class:`~aiohttp.ClientTimeout` carried by the frame.
    * ``FREEZE_BASE`` -- ignore a shorter per-call timeout and keep the
      session default baseline; cancelled tasks still fail.
    """

    KEEP = "keep"
    FAIL_FAST = "fail_fast"
    FREEZE_BASE = "freeze_base"


class ConnectorAdmit(enum.Enum):
    """Connector admission policy of an egress frame.

    * ``KEEP`` -- borrow and release connections unchanged.
    * ``REJECT_ACQUIRE`` -- borrowing a connection fails with
      :class:`~aiohttp.EgressAcquireDenied` before a protocol connection
      is lent, both for regular requests and ``ws_connect``.
    * ``FORCE_DETACH`` -- connections borrowed by the frame are closed
      when released and can never be pooled for reuse.
    """

    KEEP = "keep"
    REJECT_ACQUIRE = "reject_acquire"
    FORCE_DETACH = "force_detach"


@dataclass(frozen=True, slots=True)
class EgressProfile:
    """An immutable frame of the outbound egress profile stack.

    The four dimensions are independent enums and never collapse into a
    single boolean.  ``fail_fast_timeout`` is only consulted when
    ``timeout_overlay`` is :attr:`TimeoutOverlay.FAIL_FAST`.
    """

    redirect_surface: RedirectSurface = RedirectSurface.KEEP
    cookie_surface: CookieSurface = CookieSurface.KEEP
    timeout_overlay: TimeoutOverlay = TimeoutOverlay.KEEP
    connector_admit: ConnectorAdmit = ConnectorAdmit.KEEP
    fail_fast_timeout: ClientTimeout | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.redirect_surface, RedirectSurface):
            raise TypeError("redirect_surface must be a RedirectSurface member")
        if not isinstance(self.cookie_surface, CookieSurface):
            raise TypeError("cookie_surface must be a CookieSurface member")
        if not isinstance(self.timeout_overlay, TimeoutOverlay):
            raise TypeError("timeout_overlay must be a TimeoutOverlay member")
        if not isinstance(self.connector_admit, ConnectorAdmit):
            raise TypeError("connector_admit must be a ConnectorAdmit member")
        if (
            self.timeout_overlay is TimeoutOverlay.FAIL_FAST
            and self.fail_fast_timeout is None
        ):
            raise ValueError(
                "fail_fast_timeout is required when "
                "timeout_overlay=TimeoutOverlay.FAIL_FAST"
            )


@dataclass(frozen=True, slots=True)
class EgressView:
    """Read-only snapshot of the egress stack state in the current task."""

    depth: int
    profile: EgressProfile | None
    redirect_surface: RedirectSurface
    cookie_surface: CookieSurface
    timeout_overlay: TimeoutOverlay
    connector_admit: ConnectorAdmit

    @property
    def active(self) -> bool:
        """Whether at least one frame is on the stack."""
        return self.profile is not None

    def effective_max_redirects(self, session_max_redirects: int) -> int:
        """Merge the frame with a ``max_redirects`` baseline."""
        if self.redirect_surface is RedirectSurface.DENY_REDIRECT:
            return 0
        if self.redirect_surface is RedirectSurface.CAP_ONE:
            return 1
        return session_max_redirects

    def resolve_timeout(
        self,
        call_timeout: "ClientTimeout | _SENTINEL | None",
        session_timeout: "ClientTimeout",
    ) -> "ClientTimeout":
        """Merge the frame with a per-call and a session timeout."""
        return resolve_request_timeout(call_timeout, session_timeout)

    @property
    def jar_frozen(self) -> bool:
        """True when responses must not write cookies into the jar."""
        return self.cookie_surface is CookieSurface.FREEZE_JAR

    @property
    def outbound_cookies_omitted(self) -> bool:
        """True when requests must not carry jar cookies."""
        return self.cookie_surface is CookieSurface.OMIT_OUTBOUND

    @property
    def acquire_rejected(self) -> bool:
        """True when connection borrowing must fail."""
        return self.connector_admit is ConnectorAdmit.REJECT_ACQUIRE

    @property
    def detach_forced(self) -> bool:
        """True when borrowed connections must be closed on release."""
        return self.connector_admit is ConnectorAdmit.FORCE_DETACH


@dataclass(frozen=True, slots=True)
class EffectiveEgress:
    """Read-only effective view merged with session runtime state."""

    redirect_surface: RedirectSurface
    cookie_surface: CookieSurface
    timeout_overlay: TimeoutOverlay
    connector_admit: ConnectorAdmit
    max_redirects: int
    raise_for_status: bool | Callable[[Any], Any]
    timeout: "ClientTimeout"
    cookie_jar: "AbstractCookieJar"
    connector: "BaseConnector | None"
    depth: int
    profile: EgressProfile | None


# The stack is intentionally a ContextVar holding an immutable tuple.
# asyncio tasks copy the context on creation, so two concurrent tasks
# never see each other's frames.  The frames themselves carry no session,
# connector or jar references -- enforcement always acts on the objects
# the current call actually uses, so a frame never redirects another
# session's traffic onto this session's jar/connector.
_egress_stack: ContextVar[tuple[EgressProfile, ...]] = ContextVar(
    "aiohttp_egress_profile_stack",
    default=(),
)


def current_egress_profile() -> EgressProfile | None:
    """Return the top frame of the current task's stack, or ``None``."""
    stack = _egress_stack.get()
    return stack[-1] if stack else None


def current_egress_view() -> EgressView:
    """Return a read-only view of the current task's stack top."""
    stack = _egress_stack.get()
    profile = stack[-1] if stack else None
    if profile is None:
        return EgressView(
            depth=0,
            profile=None,
            redirect_surface=RedirectSurface.KEEP,
            cookie_surface=CookieSurface.KEEP,
            timeout_overlay=TimeoutOverlay.KEEP,
            connector_admit=ConnectorAdmit.KEEP,
        )
    return EgressView(
        depth=len(stack),
        profile=profile,
        redirect_surface=profile.redirect_surface,
        cookie_surface=profile.cookie_surface,
        timeout_overlay=profile.timeout_overlay,
        connector_admit=profile.connector_admit,
    )


def effective_egress(session: Any, *, max_redirects: int = 10) -> EffectiveEgress:
    """Return the read-only effective profile merged with a live session.

    The merge uses the session's current default timeout,
    ``raise_for_status``, cookie jar and connector; it does not copy or
    replace them.
    """
    view = current_egress_view()
    return EffectiveEgress(
        redirect_surface=view.redirect_surface,
        cookie_surface=view.cookie_surface,
        timeout_overlay=view.timeout_overlay,
        connector_admit=view.connector_admit,
        max_redirects=view.effective_max_redirects(max_redirects),
        raise_for_status=session.raise_for_status,
        timeout=_resolve_timeout(sentinel, session.timeout),
        cookie_jar=session.cookie_jar,
        connector=session.connector,
        depth=view.depth,
        profile=view.profile,
    )


def push_egress_profile(profile: EgressProfile) -> Token[tuple[EgressProfile, ...]]:
    """Push a frame onto the current task's stack.

    Pair every push with :func:`pop_egress_profile`; prefer
    :func:`egress_profile` for automatic cleanup on exceptions.
    """
    stack = _egress_stack.get()
    return _egress_stack.set(stack + (profile,))


def pop_egress_profile(
    token: Token[tuple[EgressProfile, ...]], profile: EgressProfile
) -> None:
    """Pop a frame previously pushed by :func:`push_egress_profile`.

    Raises ``RuntimeError`` if the given frame is not the current stack
    top, so frames pushed by another (nested) section can never be
    silently consumed.
    """
    stack = _egress_stack.get()
    if not stack or stack[-1] is not profile:
        raise RuntimeError(
            "egress profile stack top does not match the frame being popped"
        )
    _egress_stack.reset(token)


@contextlib.contextmanager
def egress_profile(
    *,
    redirect_surface: RedirectSurface = RedirectSurface.KEEP,
    cookie_surface: CookieSurface = CookieSurface.KEEP,
    timeout_overlay: TimeoutOverlay = TimeoutOverlay.KEEP,
    connector_admit: ConnectorAdmit = ConnectorAdmit.KEEP,
    fail_fast_timeout: "ClientTimeout | None" = None,
) -> Iterator[EgressProfile]:
    """Push an :class:`EgressProfile` frame for the duration of the block.

    The frame is bound to the current task and is popped even if the block
    exits with an exception.
    """
    profile = EgressProfile(
        redirect_surface=redirect_surface,
        cookie_surface=cookie_surface,
        timeout_overlay=timeout_overlay,
        connector_admit=connector_admit,
        fail_fast_timeout=fail_fast_timeout,
    )
    token = push_egress_profile(profile)
    try:
        yield profile
    finally:
        pop_egress_profile(token, profile)


def guard_redirect_follow(
    followed: int, *, ignore_redirect_surface: bool = False
) -> None:
    """Enforce ``redirect_surface`` before sending the next hop.

    ``followed`` is the number of hops already followed (1 when the first
    hop is about to be taken).  Websocket handshakes pass
    ``ignore_redirect_surface=True``.
    """
    if ignore_redirect_surface:
        return
    profile = current_egress_profile()
    if profile is None:
        return
    surface = profile.redirect_surface
    if surface is RedirectSurface.DENY_REDIRECT:
        raise EgressRedirectDenied(
            "redirect_surface=deny_redirect forbids following the redirect"
        )
    if surface is RedirectSurface.CAP_ONE and followed > 1:
        raise EgressRedirectCapExceeded(
            "redirect_surface=cap_one forbids following more than one redirect"
        )


def jar_is_frozen() -> bool:
    """Return True when the current frame forbids jar writes."""
    profile = current_egress_profile()
    return profile is not None and profile.cookie_surface is CookieSurface.FREEZE_JAR


def outbound_cookies_omitted() -> bool:
    """Return True when the current frame forbids outbound jar cookies."""
    profile = current_egress_profile()
    return profile is not None and profile.cookie_surface is CookieSurface.OMIT_OUTBOUND


def guard_connection_acquire() -> None:
    """Raise before a connector lends any protocol connection.

    Invoked by ``BaseConnector.connect`` (including after the limit-slot
    wait) and by every connector ``_create_connection`` implementation.
    """
    profile = current_egress_profile()
    if profile is not None and profile.connector_admit is ConnectorAdmit.REJECT_ACQUIRE:
        raise EgressAcquireDenied(
            "connector_admit=reject_acquire forbids borrowing a connection"
        )


def acquire_is_rejected() -> bool:
    """Return True when :func:`guard_connection_acquire` would raise."""
    profile = current_egress_profile()
    return (
        profile is not None and profile.connector_admit is ConnectorAdmit.REJECT_ACQUIRE
    )


def force_detach_active() -> bool:
    """Return True when the current frame forces connection detachment."""
    profile = current_egress_profile()
    return (
        profile is not None and profile.connector_admit is ConnectorAdmit.FORCE_DETACH
    )


def _min_timeout(a: float | None, b: float | None) -> float | None:
    if a is None:
        return b
    if b is None:
        return a
    return min(a, b)


def _cap(value: float | None, ceiling: float | None) -> float | None:
    if value is None or ceiling is None:
        return value
    return min(value, ceiling)


def resolve_request_timeout(
    call_timeout: "ClientTimeout | _SENTINEL | None",
    session_timeout: "ClientTimeout",
) -> "ClientTimeout":
    """Resolve the effective timeout for an outgoing request.

    Same semantics as :meth:`EgressView.resolve_timeout`, callable
    without first taking a view snapshot.
    """
    return _resolve_timeout(call_timeout, session_timeout)


def _resolve_timeout(
    call_timeout: "ClientTimeout | _SENTINEL | None",
    session_timeout: "ClientTimeout",
) -> "ClientTimeout":
    # Imported lazily at call time to keep the module import-light.
    from .client_reqrep import ClientTimeout

    baseline = (
        session_timeout
        if call_timeout is sentinel or call_timeout is None
        else call_timeout
    )
    profile = current_egress_profile()
    if profile is None or profile.timeout_overlay is TimeoutOverlay.KEEP:
        return baseline
    if profile.timeout_overlay is TimeoutOverlay.FREEZE_BASE:
        # The per-call timeout is ignored; the session baseline governs.
        return session_timeout

    # FAIL_FAST: clamp each deadline with the frame's shorter timeout.
    overlay = profile.fail_fast_timeout
    assert overlay is not None  # checked in EgressProfile.__post_init__
    total = _min_timeout(baseline.total, overlay.total)
    connect = _cap(_min_timeout(baseline.connect, overlay.connect), total)
    sock_connect = _cap(
        _min_timeout(baseline.sock_connect, overlay.sock_connect), total
    )
    sock_read = _cap(_min_timeout(baseline.sock_read, overlay.sock_read), total)
    if (
        total == baseline.total
        and connect == baseline.connect
        and sock_connect == baseline.sock_connect
        and sock_read == baseline.sock_read
    ):
        return baseline
    return ClientTimeout(
        total=total,
        connect=connect,
        sock_connect=sock_connect,
        sock_read=sock_read,
        ceil_threshold=baseline.ceil_threshold,
    )
