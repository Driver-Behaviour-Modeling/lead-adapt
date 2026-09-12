"""Eager validation and full-graph compilation must coexist in the backbone."""

import importlib

import jaxtyping as jt
import pytest
import torch
from beartype.roar import (
    BeartypeCallHintParamViolation,
    BeartypeCallHintReturnViolation,
)

from lead.common.runtime_typing import runtime_beartype, runtime_jaxtyped


@runtime_beartype
def typed_scale(inputs: jt.Float[torch.Tensor, "B 3"], scale: float):
    return inputs * scale


@runtime_jaxtyped
def typed_pair(
    left: jt.Float[torch.Tensor, "B 3"],
    right: jt.Float[torch.Tensor, "B 3"],
) -> jt.Float[torch.Tensor, "B 3"]:
    return left * right + left


def test_eager_argument_and_shared_axis_checks_remain_active():
    with pytest.raises(BeartypeCallHintParamViolation):
        typed_scale(torch.randn(2, 4), 2.0)
    with pytest.raises(BeartypeCallHintParamViolation):
        typed_scale(torch.randn(2, 3), "bad")
    # Broadcasting would work, but eager jaxtyping requires equal batch axes.
    with pytest.raises(jt.TypeCheckError):
        typed_pair(torch.randn(2, 3), torch.randn(1, 3))


@pytest.mark.parametrize("decorator", [runtime_beartype, runtime_jaxtyped])
def test_eager_return_checks_remain_active(decorator):
    @decorator
    def wrong_return(
        inputs: jt.Float[torch.Tensor, "B 3"],
    ) -> jt.Float[torch.Tensor, "B 3"]:
        return inputs[:, :2]

    with pytest.raises((jt.TypeCheckError, BeartypeCallHintReturnViolation)):
        wrong_return(torch.randn(2, 3))


def counting_compiler(function):
    graphs = []

    def backend(graph, example_inputs, **kwargs):
        graphs.append(graph)
        return graph.forward

    return torch.compile(function, backend=backend, fullgraph=True), graphs


def test_typed_functions_compile_as_one_graph_and_preserve_gradients():
    def operation(left, right):
        return typed_scale(typed_pair(left, right), 2.0)

    compiled, graphs = counting_compiler(operation)
    left = torch.randn(2, 3, requires_grad=True)
    right = torch.randn(2, 3, requires_grad=True)
    expected = operation(left, right)
    actual = compiled(left, right)
    torch.testing.assert_close(actual, expected)
    expected_gradients = torch.autograd.grad(expected.sum(), (left, right))
    actual_gradients = torch.autograd.grad(actual.sum(), (left, right))
    for actual_gradient, expected_gradient in zip(
        actual_gradients,
        expected_gradients,
        strict=True,
    ):
        torch.testing.assert_close(actual_gradient, expected_gradient)
    assert len(graphs) == 1
    # Capturing the raw path must not disable checks on later eager calls.
    with pytest.raises(jt.TypeCheckError):
        typed_pair(torch.randn(2, 3), torch.randn(1, 3))


@pytest.mark.parametrize("package", ["adapt", "tfv6"])
def test_actual_imagenet_normalization_compiles_and_rejects_bad_eager_shape(package):
    utils = importlib.import_module(f"lead.{package}.transfuser_utils")
    with pytest.raises(BeartypeCallHintParamViolation):
        utils.normalize_imagenet(torch.randn(1, 4, 4, 6))
    compiled, graphs = counting_compiler(utils.normalize_imagenet)
    inputs = torch.randn(1, 3, 4, 6, requires_grad=True)
    expected = utils.normalize_imagenet(inputs)
    actual = compiled(inputs)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(
        torch.autograd.grad(actual.sum(), inputs)[0],
        torch.autograd.grad(expected.sum(), inputs)[0],
    )
    assert len(graphs) == 1


@pytest.mark.parametrize("package", ["adapt", "tfv6"])
def test_actual_fusion_block_compiles_with_fp32_norms_and_unchanged_state(package):
    backbone = importlib.import_module(f"lead.{package}.transfuser_backbone")
    utils = importlib.import_module(f"lead.{package}.transfuser_utils")
    eager = backbone.Block(8, 2, 2, 0.0, 0.0)
    compiled_module = backbone.Block(8, 2, 2, 0.0, 0.0)
    compiled_module.load_state_dict(eager.state_dict())
    utils.patch_norm_fp32(eager)
    utils.patch_norm_fp32(compiled_module)
    compiled, graphs = counting_compiler(compiled_module)
    inputs = torch.randn(2, 5, 8, requires_grad=True)
    with pytest.raises(jt.TypeCheckError):
        eager(torch.randn(2, 8))
    expected = eager(inputs)
    actual = compiled(inputs)
    torch.testing.assert_close(actual, expected)
    expected.square().mean().backward()
    actual.square().mean().backward()
    for actual_parameter, expected_parameter in zip(
        compiled_module.parameters(),
        eager.parameters(),
        strict=True,
    ):
        torch.testing.assert_close(actual_parameter.grad, expected_parameter.grad)
    assert len(graphs) == 1
    assert set(compiled_module.state_dict()) == set(eager.state_dict())
    restored = backbone.Block(8, 2, 2, 0.0, 0.0)
    restored.load_state_dict(compiled_module.state_dict(), strict=True)
