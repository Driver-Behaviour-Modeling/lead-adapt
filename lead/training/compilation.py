"""Compile the module that training calls without changing checkpoint keys."""

import logging

from torch import nn
from torch.nn.parallel import DistributedDataParallel

LOG = logging.getLogger(__name__)


def configure_compilation(model_wrapper: nn.Module, config) -> None:
    """Enable compilation in place, preserving the model and DDP interfaces.

    ``torch.compile(model)`` returns another module. Discarding that return value
    leaves eager training active; replacing the model also prefixes state keys.
    ``Module.compile`` installs the compiled call on the existing module instead.
    """
    if not config.compile:
        LOG.info("Training compilation disabled")
        return

    scope = config.compile_scope
    if scope == "backbone":
        model = (
            model_wrapper.module
            if isinstance(model_wrapper, DistributedDataParallel)
            else model_wrapper
        )
        target = model.backbone
    elif scope == "model":
        target = model_wrapper
    else:
        raise ValueError(f"Unknown compile_scope {scope!r}; use 'backbone' or 'model'")

    target.compile(
        backend=config.compile_backend,
        mode=config.compile_mode,
        fullgraph=config.compile_fullgraph,
        dynamic=config.compile_dynamic,
    )
    LOG.info(
        "Training compilation enabled: scope=%s backend=%s mode=%s fullgraph=%s",
        scope,
        config.compile_backend,
        config.compile_mode,
        config.compile_fullgraph,
    )
