"""Keep eager runtime type checks out of torch.compile's traced tensor code.

These decorators preserve the original beartype/jaxtyping checks for ordinary
Python calls. During compiler capture, they call the undecorated function so
shape-memo bookkeeping and Python type introspection are not traced. Compiled
execution relies on PyTorch's tensor operations and compiler guards instead.
They do not unwrap or bypass other decorators, such as FP32 normalization.
"""

from collections.abc import Callable
from functools import wraps
from typing import ParamSpec, TypeVar

import torch
from beartype import beartype
from jaxtyping import jaxtyped

P = ParamSpec("P")
R = TypeVar("R")


def _compiler_aware_check(
    function: Callable[P, R],
    checked_function: Callable[P, R],
) -> Callable[P, R]:
    @wraps(function)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        if torch.compiler.is_compiling():
            return function(*args, **kwargs)
        return checked_function(*args, **kwargs)

    return wrapper


def runtime_beartype(function: Callable[P, R]) -> Callable[P, R]:
    """Apply ordinary beartype checks outside compiler capture."""
    return _compiler_aware_check(function, beartype(function))


def runtime_jaxtyped(function: Callable[P, R]) -> Callable[P, R]:
    """Apply jaxtyping's shared-axis checks outside compiler capture."""
    return _compiler_aware_check(function, jaxtyped(typechecker=beartype)(function))
