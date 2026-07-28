"""Upstream hub fetch — used at most once per artifact, ever.

The cache writes through to object storage on the first fetch (FR-1.3), so this
path is taken once in an artifact's life and never again, including after the
local copy is evicted. When the platform is offline, reaching here is an error
rather than a silent download.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from .config import Config
from .errors import NotFound, Offline
from .keys import Key


class Hub(ABC):
    """Somewhere an artifact can come from when neither cache nor store has it."""

    name = "upstream hub"

    @abstractmethod
    def size(self, key: Key) -> int | None:
        """Total download size if it can be known cheaply, else None."""

    @abstractmethod
    def fetch(self, key: Key, dest: Path) -> None: ...


class OfflineHub(Hub):
    """The configured behaviour when outbound access is disabled (NFR-3.4)."""

    name = "offline"

    def __init__(self, reason: str = "outbound access is disabled") -> None:
        self.reason = reason

    def size(self, key: Key) -> int | None:
        return None

    def fetch(self, key: Key, dest: Path) -> None:
        raise Offline(
            f"{key} is not in object storage and {self.reason}.",
            "Seed it into object storage first, or unset NAWAT_OFFLINE for this fetch.",
        )


class HuggingFaceHub(Hub):
    """Fetches models and datasets from the Hugging Face Hub."""

    name = "Hugging Face Hub"

    def __init__(self, token: str | None = None) -> None:
        self.token = token

    def _api(self):
        try:
            from huggingface_hub import HfApi
        except ImportError as exc:
            raise Offline(
                "huggingface_hub is not installed, so the upstream hub cannot be reached.",
                "Install it with: pip install huggingface_hub — or seed the artifact into object storage directly.",
            ) from exc
        return HfApi(token=self.token)

    def _repo_type(self, key: Key) -> str:
        return "dataset" if key.kind == "datasets" else "model"

    def size(self, key: Key) -> int | None:
        repo_id = key.hub_repo_id
        if not repo_id:
            return None
        try:
            info = self._api().repo_info(repo_id, repo_type=self._repo_type(key), files_metadata=True)
        except Offline:
            raise
        except Exception:
            return None
        total = 0
        for sibling in getattr(info, "siblings", None) or ():
            total += getattr(sibling, "size", None) or 0
        return total or None

    def fetch(self, key: Key, dest: Path) -> None:
        repo_id = key.hub_repo_id
        if not repo_id:
            raise NotFound(
                f"{key} is not in object storage and its kind has no upstream source.",
                "Publish it from a run, or seed it into object storage directly.",
            )
        try:
            from huggingface_hub import snapshot_download
        except ImportError as exc:
            raise Offline(
                "huggingface_hub is not installed, so the upstream hub cannot be reached.",
                "Install it with: pip install huggingface_hub — or seed the artifact into object storage directly.",
            ) from exc
        dest.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id=repo_id,
            repo_type=self._repo_type(key),
            local_dir=str(dest),
            token=self.token,
            max_workers=8,
        )


def build_hub(config: Config) -> Hub:
    if config.offline:
        return OfflineHub("this host is configured offline (NAWAT_OFFLINE)")
    return HuggingFaceHub(token=config.hub_token)
