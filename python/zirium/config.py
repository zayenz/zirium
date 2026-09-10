"""JSON-compatible configuration for Zirium's shared registry reader."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class OperationShapeConfig(BaseModel):
    """Assign an existing custom grammar to an exact operation name."""

    model_config = ConfigDict(strict=True, extra="forbid")

    name: str
    shape: Literal[
        "func_like",
        "call_like",
        "binary_operands",
        "optional_typed_operands",
        "attr_first_optional_typed_operands",
        "unary_operand",
        "variadic_operands",
        "literal_attribute",
        "operand_clauses",
        "region_clauses",
    ]
    callee_attribute: str | None = Field(
        default=None, exclude_if=lambda value: value is None
    )

    @model_validator(mode="after")
    def validate_callee_attribute(self):
        if self.callee_attribute is not None and self.shape != "call_like":
            raise ValueError("callee_attribute requires a call_like shape")
        return self


class OperationFormatConfig(BaseModel):
    """Assign a validated format description to an exact operation name."""

    model_config = ConfigDict(strict=True, extra="forbid")

    name: str
    format: str


class OperationGrammarConfig(BaseModel):
    """Select exactly one shape or format for an operation alternative."""

    model_config = ConfigDict(strict=True, extra="forbid")

    shape: (
        Literal[
            "func_like",
            "call_like",
            "binary_operands",
            "optional_typed_operands",
            "attr_first_optional_typed_operands",
            "unary_operand",
            "variadic_operands",
            "literal_attribute",
            "operand_clauses",
            "region_clauses",
        ]
        | None
    ) = None
    format: str | None = None

    @model_validator(mode="after")
    def select_one_grammar(self):
        if (self.shape is None) == (self.format is None):
            raise ValueError("an alternative requires exactly one shape or format")
        return self


class OperationAlternativesConfig(BaseModel):
    """Assign two or more ordered syntax alternatives to one operation."""

    model_config = ConfigDict(strict=True, extra="forbid")

    name: str
    alternatives: list[OperationGrammarConfig]

    @model_validator(mode="after")
    def require_multiple_grammars(self):
        if len(self.alternatives) < 2:
            raise ValueError("operation alternatives require at least two grammars")
        return self


class RegistryConfig(BaseModel):
    """A complete registry composed from presets and explicit entries."""

    model_config = ConfigDict(strict=True, extra="forbid")

    presets: list[str] = Field(default_factory=list)
    builtins: list[str]
    operation_shapes: list[OperationShapeConfig]
    operation_formats: list[OperationFormatConfig] = Field(default_factory=list)
    operation_alternatives: list[OperationAlternativesConfig] = Field(
        default_factory=list
    )
