"""Batch-transfer contracts, including actual cross-stream CUDA consumption."""

import gc
import os
import weakref
from collections import OrderedDict, defaultdict, namedtuple
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np
import pytest
import torch

from lead.training.batch_pipeline import DeviceBatchIterator, move_batch_to_device

Pair = namedtuple("Pair", ["value", "label"])


@dataclass(frozen=True)
class Sample:
    tensor: torch.Tensor
    metadata: object


def test_recursive_transfer_preserves_containers_and_metadata():
    metadata = np.array(["Town12", "Town13"])
    source = OrderedDict(
        nested=[Pair(torch.ones(2, dtype=torch.int64), "label"), None],
        sample=Sample(torch.zeros(3, dtype=torch.float64), metadata),
        defaults=defaultdict(list, image=torch.zeros(3, 4, 5, dtype=torch.uint8)),
        readonly=MappingProxyType({"tensor": torch.ones(1)}),
        count=7,
    )
    moved = move_batch_to_device(source, "meta")
    assert type(moved) is OrderedDict
    assert type(moved["nested"]) is list
    assert type(moved["nested"][0]) is Pair
    assert moved["nested"][0].value.device.type == "meta"
    assert moved["nested"][0].value.dtype == torch.int64
    assert moved["nested"][0].label == "label"
    assert moved["nested"][1] is None
    assert type(moved["sample"]) is Sample
    assert moved["sample"].tensor.dtype == torch.float64
    assert moved["sample"].tensor.device.type == "meta"
    assert moved["sample"].metadata is metadata
    assert isinstance(moved["defaults"], defaultdict)
    assert moved["defaults"].default_factory is list
    assert moved["defaults"]["image"].dtype == torch.uint8
    assert isinstance(moved["readonly"], MappingProxyType)
    assert moved["readonly"]["tensor"].device.type == "meta"
    assert moved["count"] == 7
    assert source["sample"].tensor.device.type == "cpu"


@pytest.mark.parametrize("enabled", [False, True])
def test_cpu_iteration_does_not_query_cuda_or_prefetch(monkeypatch, enabled):
    def no_cuda(*args, **kwargs):
        raise AssertionError("CPU fallback touched CUDA")

    monkeypatch.setattr(torch.cuda, "Stream", no_cuda)
    monkeypatch.setattr(torch.cuda, "is_available", no_cuda)
    monkeypatch.setattr(torch.cuda, "current_stream", no_cuda)
    consumed = []

    def batches():
        for index in range(3):
            consumed.append(index)
            yield {"x": torch.tensor(index), "label": f"sample-{index}"}

    with DeviceBatchIterator(batches(), "cpu", enabled=enabled) as iterator:
        assert consumed == []
        first = next(iterator)
        assert consumed == [0]
        assert first["x"].item() == 0
        assert first["label"] == "sample-0"
        assert [batch["x"].item() for batch in iterator] == [1, 2]
    assert list(iterator) == []


@pytest.mark.parametrize("enabled", [False, True])
def test_cuda_transfer_is_rejected_inside_a_worker_before_cuda_init(
    monkeypatch,
    enabled,
):
    monkeypatch.setattr(torch.utils.data, "get_worker_info", lambda: object())

    def no_stream(*args, **kwargs):
        raise AssertionError("worker initialized CUDA")

    monkeypatch.setattr(torch.cuda, "Stream", no_stream)
    iterator = DeviceBatchIterator([torch.ones(1)], "cuda", enabled=enabled)
    with pytest.raises(RuntimeError, match="training process"):
        next(iterator)


def test_empty_iterable_and_explicit_close():
    with DeviceBatchIterator([], "cpu", enabled=True) as iterator:
        assert list(iterator) == []
    iterator.close()
    assert list(iterator) == []


@pytest.fixture
def cuda_device():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for stream lifetime checks")
    return torch.device(os.environ.get("LEAD_TEST_CUDA_DEVICE", "cuda:0"))


def test_cuda_nested_batches_survive_allocator_reuse_on_consumer_stream(cuda_device):
    count = 32

    def batches():
        for index in range(count):
            yield {
                "nested": [
                    Pair(
                        torch.full((128, 256), index, pin_memory=True),
                        str(index),
                    ),
                    Sample(torch.full((128, 256), index + 1, pin_memory=True), None),
                ],
            }

    consumer = torch.cuda.Stream(device=cuda_device)
    results = []
    with torch.cuda.stream(consumer):
        with DeviceBatchIterator(batches(), cuda_device, enabled=True) as iterator:
            for index, batch in enumerate(iterator):
                left = batch["nested"][0].value
                right = batch["nested"][1].tensor
                assert left.device == cuda_device
                assert batch["nested"][0].label == str(index)
                # Delay reads so the copy stream can reuse freed allocations.
                torch.cuda._sleep(100_000)
                results.append((left + right).sum())
                del batch, left, right
    consumer.synchronize()
    actual = torch.stack(results).cpu()
    expected = (2 * torch.arange(count) + 1) * (128 * 256)
    torch.testing.assert_close(actual, expected)


def test_cuda_early_close_retains_pinned_sources_until_upload_completes(
    cuda_device,
    monkeypatch,
):
    references = []

    def batches():
        for index in range(4):
            host = torch.full((256, 256), float(index), pin_memory=True)
            references.append(weakref.ref(host))
            yield {"x": host}
            del host

    # Force the nonblocking poll to report "pending" even on fast devices, so
    # this lifetime assertion does not depend on DMA versus Python timing.
    monkeypatch.setattr(torch.cuda.Event, "query", lambda self: False)
    consumer = torch.cuda.Stream(device=cuda_device)
    with torch.cuda.stream(consumer):
        with DeviceBatchIterator(batches(), cuda_device, enabled=True) as iterator:
            first = next(iterator)
            assert len(references) == 2  # one current and one prefetched batch
            assert references[0]() is not None
            assert references[1]() is not None
            result = first["x"].sum()
        gc.collect()
        assert all(reference() is None for reference in references)
    consumer.synchronize()
    assert result.item() == 0.0


@pytest.mark.parametrize("enabled", [False, True])
def test_cuda_eager_and_prefetch_produce_identical_values(cuda_device, enabled):
    host = torch.arange(48, dtype=torch.float64).reshape(2, 3, 8).pin_memory()
    with DeviceBatchIterator([{"x": host}], cuda_device, enabled=enabled) as iterator:
        result = next(iterator)["x"]
    torch.testing.assert_close(result.cpu(), host)


def test_source_exception_closes_pending_cuda_transfers(cuda_device):
    def batches():
        yield torch.ones(16, pin_memory=True)
        raise RuntimeError("loader failed")

    iterator = DeviceBatchIterator(batches(), cuda_device, enabled=True)
    with pytest.raises(RuntimeError, match="loader failed"):
        next(iterator)
    assert iterator._closed
    assert list(iterator) == []
