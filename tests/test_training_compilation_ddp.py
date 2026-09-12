"""CPU distributed regression: compiled training still synchronizes gradients."""

import io
import json
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.nn.parallel import DistributedDataParallel

from lead.training.compilation import configure_compilation


class DistributedBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 8)

    def forward(self, inputs):
        return self.linear(inputs).relu()


class DistributedPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = DistributedBackbone()
        self.head = nn.Linear(8, 2)

    def forward(self, inputs):
        return self.head(self.backbone(inputs))


def distributed_worker(rank, rendezvous, result_path, scope):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=f"file://{rendezvous}",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=60),
    )
    try:
        torch.manual_seed(42)
        model = DistributedPolicy()
        reference = DistributedPolicy()
        reference.load_state_dict(model.state_dict())
        wrapped = DistributedDataParallel(model)
        compiled_graphs = []

        def counting_backend(graph, example_inputs, **kwargs):
            compiled_graphs.append(graph)
            return graph.forward

        configure_compilation(
            wrapped,
            SimpleNamespace(
                compile=True,
                compile_scope=scope,
                compile_backend=counting_backend,
                compile_mode="default",
                compile_fullgraph=False,
                compile_dynamic=False,
            ),
        )
        optimizer = torch.optim.SGD(wrapped.parameters(), lr=0.01)
        reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.01)
        for step in range(2):
            # Ranks deliberately see different samples. Their averaged gradient
            # must equal eager training on the concatenation of both batches.
            torch.manual_seed(100 + rank + 10 * step)
            inputs = torch.randn(3, 4)
            all_inputs = [torch.empty_like(inputs) for _ in range(2)]
            dist.all_gather(all_inputs, inputs)
            assert not torch.equal(all_inputs[0], all_inputs[1])
            reference(torch.cat(all_inputs)).square().mean().backward()
            wrapped(inputs).square().mean().backward()
            for actual, expected in zip(
                model.parameters(),
                reference.parameters(),
                strict=True,
            ):
                torch.testing.assert_close(actual.grad, expected.grad)
            optimizer.step()
            reference_optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            reference_optimizer.zero_grad(set_to_none=True)

        assert compiled_graphs, "DDP training never invoked the compiler backend"
        assert set(model.state_dict()) == set(reference.state_dict())
        for actual, expected in zip(
            model.parameters(),
            reference.parameters(),
            strict=True,
        ):
            torch.testing.assert_close(actual, expected)

        # Trainer saves the bare module, so compiled/DDP implementation prefixes
        # must not leak into a checkpoint that existing inference loads strictly.
        checkpoint = io.BytesIO()
        torch.save(wrapped.module.state_dict(), checkpoint)
        checkpoint.seek(0)
        restored = DistributedPolicy()
        restored.load_state_dict(torch.load(checkpoint, weights_only=True), strict=True)
        torch.testing.assert_close(restored(inputs), reference(inputs))
        if rank == 0:
            Path(result_path).write_text(
                json.dumps(
                    {
                        "scope": scope,
                        "compiled_graphs": len(compiled_graphs),
                    },
                ),
            )
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(
    not dist.is_available() or not dist.is_gloo_available(),
    reason="requires the CPU gloo distributed backend",
)
@pytest.mark.parametrize("scope", ["backbone", "model"])
def test_ddp_compilation_synchronizes_and_preserves_checkpoints(tmp_path, scope):
    result = tmp_path / "result.json"
    # Spawn avoids inheriting another test's CUDA context or compiler state.
    mp.spawn(
        distributed_worker,
        args=(str(tmp_path / "rendezvous"), str(result), scope),
        nprocs=2,
        join=True,
    )
    assert json.loads(result.read_text())["compiled_graphs"] > 0
