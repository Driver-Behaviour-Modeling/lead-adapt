"""Metadata cache behavior at the file, session and worker boundaries."""

import lzma
import os
import pickle
from unittest.mock import patch

import diskcache
import numpy as np
import pytest

from lead.data_loader import metadata_cache
from lead.data_loader.metadata_cache import MetadataCache


def write_metadata(path, value):
    with lzma.open(path, "wb") as stream:
        pickle.dump(value, stream)


def test_memory_hit_avoids_source_read_and_returns_independent_nested_data(tmp_path):
    path = tmp_path / "frame.pkl"
    expected = {"route": [[1, 2], [3, 4]], "nested": {"values": [5]}}
    write_metadata(path, expected)
    cache = MetadataCache()
    with patch.object(
        metadata_cache,
        "_read_metadata_file",
        wraps=metadata_cache._read_metadata_file,
    ) as reader:
        first = cache.load(path)
        first["route"][0][0] = 99
        first["nested"]["values"].append(6)
        second = cache.load(path)
    assert second == expected
    assert reader.call_count == 1


def test_session_hit_survives_new_reader_and_ignores_legacy_path_entry(tmp_path):
    path = tmp_path / "frame.pkl"
    expected = {"speed": 7, "route": [[1, 2]]}
    write_metadata(path, expected)
    with diskcache.Cache(str(tmp_path / "session")) as session:
        session[str(path)] = {"speed": -1}
        first_cache = MetadataCache(session)
        first_cache.load(path)["route"][0][0] = -100
        second_cache = MetadataCache(session, max_entries=0)
        with patch.object(
            metadata_cache,
            "_read_metadata_file",
            side_effect=AssertionError("cache miss"),
        ):
            assert second_cache.load(path) == expected
            assert second_cache.load(path) == expected
        assert not second_cache._memory


def test_array_mutations_do_not_leak_into_memory_or_session(tmp_path):
    path = tmp_path / "frame.pkl"
    expected = np.arange(6, dtype=np.float32).reshape(3, 2)
    write_metadata(path, {"route": expected})
    session = {}
    cache = MetadataCache(session)
    cache.load(path)["route"][:] = -1
    np.testing.assert_array_equal(cache.load(path)["route"], expected)
    np.testing.assert_array_equal(MetadataCache(session).load(path)["route"], expected)


def test_file_replacement_invalidates_memory_and_session_even_with_same_mtime(tmp_path):
    path = tmp_path / "frame.pkl"
    write_metadata(path, {"speed": 1})
    session = {}
    cache = MetadataCache(session)
    assert cache.load(path) == {"speed": 1}
    previous_stat = path.stat()
    replacement = tmp_path / "replacement.pkl"
    write_metadata(replacement, {"speed": 2})
    os.utime(replacement, ns=(previous_stat.st_atime_ns, previous_stat.st_mtime_ns))
    replacement.replace(path)
    assert cache.load(path) == {"speed": 2}
    assert MetadataCache(session).load(path) == {"speed": 2}


def test_in_place_rewrite_invalidates_cached_metadata(tmp_path):
    path = tmp_path / "frame.pkl"
    write_metadata(path, {"route": [1]})
    cache = MetadataCache({})
    cache.load(path)
    write_metadata(path, {"route": [1, 2, 3]})
    assert cache.load(path) == {"route": [1, 2, 3]}


def test_lru_is_bounded_and_recent_reads_keep_their_entries(tmp_path):
    paths = [tmp_path / f"{index}.pkl" for index in range(3)]
    for index, path in enumerate(paths):
        write_metadata(path, {"index": index})
    cache = MetadataCache(max_entries=2)
    with patch.object(
        metadata_cache,
        "_read_metadata_file",
        wraps=metadata_cache._read_metadata_file,
    ) as reader:
        for index in (0, 1, 0, 2, 0):
            assert cache.load(paths[index]) == {"index": index}
        assert reader.call_count == 3
        assert len(cache._memory) == 2
        assert cache.load(paths[1]) == {"index": 1}
        assert reader.call_count == 4
        assert len(cache._memory) == 2


def test_zero_capacity_disables_memory_and_negative_capacity_is_rejected(tmp_path):
    path = tmp_path / "frame.pkl"
    write_metadata(path, {"speed": 1})
    cache = MetadataCache(max_entries=0)
    with patch.object(
        metadata_cache,
        "_read_metadata_file",
        wraps=metadata_cache._read_metadata_file,
    ) as reader:
        cache.load(path)
        cache.load(path)
    assert reader.call_count == 2
    assert not cache._memory
    with pytest.raises(ValueError, match="nonnegative"):
        MetadataCache(max_entries=-1)


def test_worker_boundaries_clear_inherited_memory(tmp_path):
    path = tmp_path / "frame.pkl"
    write_metadata(path, {"speed": 1})
    cache = MetadataCache()
    cache.load(path)
    spawned_copy = pickle.loads(pickle.dumps(cache))
    assert not spawned_copy._memory
    with (
        patch.object(metadata_cache.os, "getpid", return_value=cache._pid + 1),
        patch.object(
            metadata_cache,
            "_read_metadata_file",
            wraps=metadata_cache._read_metadata_file,
        ) as reader,
    ):
        assert cache.load(path) == {"speed": 1}
    assert reader.call_count == 1


def test_file_change_during_read_retries_before_publishing_session_value(tmp_path):
    path = tmp_path / "frame.pkl"
    write_metadata(path, {"speed": 1})
    original_reader = metadata_cache._read_metadata_file
    reads = []

    def read_then_replace(path):
        result = original_reader(path)
        reads.append(result)
        if len(reads) == 1:
            write_metadata(path, {"speed": 2, "new": True})
        return result

    session = {}
    with patch.object(
        metadata_cache,
        "_read_metadata_file",
        side_effect=read_then_replace,
    ):
        assert MetadataCache(session).load(path) == {"speed": 2, "new": True}
    assert len(reads) == 2
    assert [pickle.loads(value) for value in session.values()] == [
        {"speed": 2, "new": True},
    ]


def test_deleted_source_does_not_return_stale_cached_metadata(tmp_path):
    path = tmp_path / "frame.pkl"
    write_metadata(path, {"speed": 1})
    cache = MetadataCache({})
    cache.load(path)
    path.unlink()
    with pytest.raises(FileNotFoundError):
        cache.load(path)
