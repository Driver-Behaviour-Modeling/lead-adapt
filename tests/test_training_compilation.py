"""Compilation must run, preserve gradients and keep old checkpoint keys."""

import copy
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from lead.training.compilation import configure_compilation


class TinyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 8)

    def forward(self, data):
        return self.linear(data).relu()


class TinyPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = TinyBackbone()
        self.head = nn.Linear(8, 2)

    def forward(self, data):
        return self.head(self.backbone(data))


@pytest.mark.parametrize("scope", ["backbone", "model"])
def test_compiled_call_runs_and_preserves_checkpoint_and_gradients(scope):
    torch.manual_seed(42)
    eager = TinyPolicy()
    compiled = copy.deepcopy(eager)
    calls = []

    def counting_backend(graph, example_inputs, **kwargs):
        calls.append(graph)
        return graph.forward

    config = SimpleNamespace(
        compile=True,
        compile_scope=scope,
        compile_backend=counting_backend,
        compile_mode="default",
        compile_fullgraph=True,
        compile_dynamic=False,
    )
    keys = set(eager.state_dict())
    configure_compilation(compiled, config)
    x = torch.randn(3, 4)
    expected, actual = eager(x), compiled(x)
    torch.testing.assert_close(actual, expected)
    expected.square().mean().backward()
    actual.square().mean().backward()
    assert calls, "The compiled graph was never called"
    assert set(compiled.state_dict()) == keys
    for a, b in zip(eager.parameters(), compiled.parameters(), strict=True):
        torch.testing.assert_close(a.grad, b.grad)
    restored = TinyPolicy()
    restored.load_state_dict(compiled.state_dict(), strict=True)
    torch.testing.assert_close(restored(x), actual)


def test_disabled_compilation_does_not_wrap_model():
    model = TinyPolicy()
    configure_compilation(model, SimpleNamespace(compile=False))
    assert model._compiled_call_impl is None
    assert model.backbone._compiled_call_impl is None


def test_compile_switch_can_be_disabled_in_legacy_config(monkeypatch):
    from lead.training.config_training import TrainingConfig

    monkeypatch.delenv("LEAD_TRAINING_CONFIG", raising=False)
    monkeypatch.setattr("sys.argv", ["test"])
    config = TrainingConfig({"compile": False})
    assert config.compile is False
    assert config.compile_scope == "backbone"
