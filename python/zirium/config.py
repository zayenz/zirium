"""JSON-compatible configuration for Zirium's shared registry reader."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class OperationShapeConfig(BaseModel):
    """Assign an existing custom grammar to an exact operation name."""

    model_config = ConfigDict(strict=True, extra="forbid")

    name: str
    shape: Literal[
        "func_like",
        "call_like",
        "binary_operands",
        "optional_typed_operands",
        "unary_operand",
        "variadic_operands",
        "literal_attribute",
        "operand_clauses",
        "region_clauses",
    ]


class OperationFormatConfig(BaseModel):
    """Assign a validated format description to an exact operation name."""

    model_config = ConfigDict(strict=True, extra="forbid")

    name: str
    format: str


class RegistryConfig(BaseModel):
    """A complete registry composed from presets and explicit entries."""

    model_config = ConfigDict(strict=True, extra="forbid")

    presets: list[str] = Field(default_factory=list)
    builtins: list[str]
    operation_shapes: list[OperationShapeConfig]
    operation_formats: list[OperationFormatConfig] = Field(default_factory=list)
