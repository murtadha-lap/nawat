"""Errors the researcher is meant to read.

Every message states the cause and the corrective action on one line
(PRD 9.8 Voice, NFR-3.3). Exit codes are stable so shell scripts can branch
on the failure class without parsing text.
"""

from __future__ import annotations


class NawatError(Exception):
    """Base for every failure the platform reports to a human."""

    exit_code = 1

    def __init__(self, cause: str, remedy: str | None = None) -> None:
        self.cause = cause.strip()
        self.remedy = remedy.strip() if remedy else None
        super().__init__(f"{self.cause} {self.remedy}" if self.remedy else self.cause)


class InvalidKey(NawatError):
    """A key does not name an addressable artifact."""

    exit_code = 2


class NotFound(NawatError):
    """The artifact is not in the cache, the object store, or upstream."""

    exit_code = 3


class StoreUnavailable(NawatError):
    """Object storage could not be reached or answered with an error.

    Raised rather than swallowed: with the store unreachable we cannot verify a
    replica, and an unverified artifact is never evicted (PRD principle 2).
    """

    exit_code = 4


class VerificationFailed(NawatError):
    """A local copy and its replica disagree. The local copy is preserved."""

    exit_code = 5


class InsufficientSpace(NawatError):
    """Space could not be freed safely. Nothing was deleted beyond what is listed."""

    exit_code = 6

    def __init__(self, cause: str, remedy: str | None = None, held: list[str] | None = None) -> None:
        super().__init__(cause, remedy)
        self.held = held or []


class Offline(NawatError):
    """An upstream fetch was required but outbound access is disabled."""

    exit_code = 7


class Protected(NawatError):
    """The artifact is pinned or leased and will not be removed."""

    exit_code = 8
