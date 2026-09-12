"""Optional batch transfer and one-batch CUDA lookahead in the training process.

The iterable must not mutate a yielded host batch while its upload is pending.
DataLoader's pinned batches satisfy that ownership contract. Use the iterator as
a context manager so an early exit also releases outstanding pinned batches.
This module never initializes CUDA at import time or inside a DataLoader worker.
"""

from __future__ import annotations

import copy
from collections import deque
from collections.abc import Callable, Iterable, Iterator, Mapping, MutableMapping
from dataclasses import fields, is_dataclass
from typing import Any

import torch


def _map_tensors(value: Any, operation: Callable[[torch.Tensor], Any]) -> Any:
    """Transform tensor leaves without changing ordinary batch containers."""
    if isinstance(value, torch.Tensor):
        return operation(value)
    if isinstance(value, Mapping):
        mapped = {key: _map_tensors(item, operation) for key, item in value.items()}
        if isinstance(value, MutableMapping):
            result = copy.copy(value)
            result.clear()
            result.update(mapped)
            return result
        return type(value)(mapped)
    if isinstance(value, tuple) and hasattr(value, "_fields"):
        return type(value)(*(_map_tensors(item, operation) for item in value))
    if isinstance(value, tuple):
        return type(value)(_map_tensors(item, operation) for item in value)
    if isinstance(value, list):
        result = copy.copy(value)
        result[:] = [_map_tensors(item, operation) for item in value]
        return result
    if is_dataclass(value) and not isinstance(value, type):
        result = copy.copy(value)
        for field in fields(value):
            object.__setattr__(
                result,
                field.name,
                _map_tensors(getattr(value, field.name), operation),
            )
        return result
    return value


def move_batch_to_device(
    batch: Any,
    device: torch.device | str | int,
    *,
    non_blocking: bool = False,
) -> Any:
    """Move tensor leaves; preserve dtypes, layouts and non-tensor metadata."""
    return _map_tensors(
        batch,
        lambda tensor: tensor.to(device=device, non_blocking=non_blocking),
    )


def record_batch_stream(batch: Any, stream: torch.cuda.Stream) -> None:
    """Keep CUDA storage live until this consumer stream has finished with it."""
    if isinstance(batch, torch.Tensor):
        if batch.is_cuda and batch.device == stream.device:
            batch.record_stream(stream)
    elif isinstance(batch, Mapping):
        for value in batch.values():
            record_batch_stream(value, stream)
    elif isinstance(batch, tuple | list):
        for value in batch:
            record_batch_stream(value, stream)
    elif is_dataclass(batch) and not isinstance(batch, type):
        for field in fields(batch):
            record_batch_stream(getattr(batch, field.name), stream)


class DeviceBatchIterator(Iterator):
    """Transfer batches eagerly, or prefetch one batch on a separate CUDA stream.

    ``enabled=False`` performs ordinary blocking transfers. CPU targets also use
    that path, without querying CUDA. A CUDA target still requires working CUDA;
    unavailable accelerators are not silently replaced with a different device.

    The CUDA path fences the consumer on each batch's copy event, recursively
    records its tensors on the consumer stream, and retains the source batch
    until DMA completes. At most two consumed host batches are retained; one
    additional host batch may belong to the prefetched item. CUDA construction
    happens lazily in the process consuming the iterator, never in workers.
    """

    def __init__(
        self,
        batches: Iterable,
        device: torch.device | str | int,
        *,
        enabled: bool = False,
    ) -> None:
        self.device = (
            torch.device("cuda", device)
            if isinstance(device, int)
            else torch.device(device)
        )
        self.enabled = enabled and self.device.type == "cuda"
        self._source = batches
        self._iterator: Iterator | None = None
        self._stream: torch.cuda.Stream | None = None
        self._pending: tuple[Any, Any, torch.cuda.Event] | None = None
        self._host_inflight: deque[tuple[Any, torch.cuda.Event]] = deque()
        self._closed = False

    def __enter__(self) -> DeviceBatchIterator:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def __iter__(self) -> DeviceBatchIterator:
        return self

    def _start(self) -> None:
        if (
            self.device.type == "cuda"
            and torch.utils.data.get_worker_info() is not None
        ):
            raise RuntimeError("CUDA batch transfer must run in the training process.")
        # Start the DataLoader iterator before creating our CUDA stream.
        self._iterator = iter(self._source)
        if self.enabled:
            self._stream = torch.cuda.Stream(device=self.device)
            # Resolve an unindexed 'cuda' to the actual device of our stream.
            self.device = self._stream.device
            self._preload()

    def _release_completed_hosts(self) -> None:
        while self._host_inflight and self._host_inflight[0][1].query():
            self._host_inflight.popleft()
        # Bound retained pinned memory even if the CPU outruns DMA submission.
        while len(self._host_inflight) > 2:
            self._host_inflight[0][1].synchronize()
            self._host_inflight.popleft()

    def _preload(self) -> None:
        assert self._iterator is not None and self._stream is not None
        try:
            host_batch = next(self._iterator)
        except StopIteration:
            self._pending = None
            return
        copy_event = torch.cuda.Event()
        # This also makes already-on-device leaves safe when callers supply a
        # mixed host/device batch produced by the current stream.
        self._stream.wait_stream(torch.cuda.current_stream(self.device))
        try:
            with torch.cuda.stream(self._stream):
                moved_batch = move_batch_to_device(
                    host_batch,
                    self.device,
                    non_blocking=True,
                )
                record_batch_stream(host_batch, self._stream)
                copy_event.record(self._stream)
        except BaseException:
            # A failed recursive move can leave earlier uploads in flight.
            # Keep host_batch alive until those uploads have completed.
            self._stream.synchronize()
            raise
        self._pending = (host_batch, moved_batch, copy_event)
        self._release_completed_hosts()

    def __next__(self) -> Any:
        if self._closed:
            raise StopIteration
        try:
            if self._iterator is None:
                self._start()
            assert self._iterator is not None
            if not self.enabled:
                return move_batch_to_device(next(self._iterator), self.device)
            if self._pending is None:
                raise StopIteration

            host_batch, moved_batch, copy_event = self._pending
            self._pending = None
            compute_stream = torch.cuda.current_stream(self.device)
            compute_stream.wait_event(copy_event)
            record_batch_stream(moved_batch, compute_stream)
            self._host_inflight.append((host_batch, copy_event))
            self._preload()
            return moved_batch
        except BaseException:
            self.close()
            raise

    def close(self) -> None:
        """Drain only the copy stream before releasing outstanding host batches."""
        if self._closed:
            return
        if self._stream is not None:
            self._stream.synchronize()
        self._pending = None
        self._host_inflight.clear()
        self._iterator = None
        self._source = ()
        self._closed = True
