"""Bounded worker-local and fingerprinted session caching for CARLA metadata."""

from __future__ import annotations

import lzma
import os
import pickle
from collections import OrderedDict
from collections.abc import MutableMapping
from typing import Any

_CACHE_VERSION = "lead.metadata.v1"


def _file_key(path: str) -> tuple:
    stat = os.stat(path)
    return (
        _CACHE_VERSION,
        path,
        stat.st_dev,
        stat.st_ino,
        stat.st_size,
        stat.st_mtime_ns,
        stat.st_ctime_ns,
    )


def _read_metadata_file(path: str) -> dict[str, Any]:
    with lzma.open(path, "rb") as stream:
        return pickle.load(stream)


class MetadataCache:
    """Cache decompressed metadata without sharing mutable samples with callers.

    Files are stat-ed on every request so replacing or editing a recording
    invalidates both cache tiers. The session namespace deliberately ignores
    the old path-only entries, which cannot establish whether a file changed.
    Each DataLoader worker retains at most ``max_entries`` serialized records.
    Uncompressed pickle bytes deserialize faster than deep-copying these large
    nested records and give each caller independent lists, dicts and arrays.
    """

    def __init__(
        self,
        session_cache: MutableMapping | None = None,
        max_entries: int = 128,
    ) -> None:
        if max_entries < 0:
            raise ValueError("metadata_cache_max_entries must be nonnegative")
        self.session_cache = session_cache
        self.max_entries = max_entries
        self._memory: OrderedDict[tuple, bytes] = OrderedDict()
        self._pid = os.getpid()

    def __getstate__(self) -> dict:
        # Spawned workers must not receive the parent's potentially warm LRU.
        return {**self.__dict__, "_memory": OrderedDict(), "_pid": None}

    def _remember(self, key: tuple, payload: bytes) -> None:
        if self.max_entries:
            self._memory[key] = payload
            self._memory.move_to_end(key)
            while len(self._memory) > self.max_entries:
                self._memory.popitem(last=False)

    def load(self, path: str | os.PathLike[str]) -> dict[str, Any]:
        """Read one metadata record, returning an independent mutable copy."""
        if self._pid != os.getpid():
            # Forked workers inherit objects without invoking __getstate__.
            self._memory.clear()
            self._pid = os.getpid()
        path = os.path.abspath(os.fspath(path))

        # A source file can be replaced while it is being decoded. Do not
        # publish that value under the fingerprint of a different file.
        for _ in range(3):
            key = _file_key(path)
            if key in self._memory:
                self._memory.move_to_end(key)
                return pickle.loads(self._memory[key])

            payload = (
                self.session_cache.get(key) if self.session_cache is not None else None
            )
            if payload is None:
                metadata = _read_metadata_file(path)
                if _file_key(path) != key:
                    continue
                if not self.max_entries and self.session_cache is None:
                    return metadata
                payload = pickle.dumps(metadata, protocol=pickle.HIGHEST_PROTOCOL)
                if self.session_cache is not None:
                    self.session_cache[key] = payload
                self._remember(key, payload)
                # This object came directly from the file and is not retained.
                return metadata

            self._remember(key, payload)
            return pickle.loads(payload)

        raise RuntimeError(f"Metadata file changed repeatedly while reading: {path}")
