"""JSON-compatible configuration for Zirium's shared registry reader."""

from typing import Literal

from pydantic import BaseModel, ConfigDict


class OperationShapeConfig(BaseModel):
    """Assign an existing custom grammar to an exact operation name."""

    model_config = ConfigDict(strict=True, extra="forbid")

    name: str
    shape: Literal["func_like", "call_like"]


class RegistryConfig(BaseModel):
    """A complete registry; registration conflicts are checked by from_config."""

    model_config = ConfigDict(strict=True, extra="forbid")

    builtins: list[str]
    operation_shapes: list[OperationShapeConfig]
