"""Immutable queries evaluated in Rust by Document.query().

Use ops() for the whole document and input() for a nested query's input.
"""

from abc import ABC, abstractmethod
from typing import Generic, Never, TypeAlias, TypeVar

from ._zirium import (  # ty: ignore[unresolved-import]
    AttributeQuery,
    CaptureQuery,
    ChangePredicate,
    ChangeQuery,
    CountQuery,
    DiffCountQuery,
    DiffOpQuery,
    DiffStringQuery,
    MapQuery,
    MatchBuilder,
    MatchPredicate,
    OpQuery,
    Predicate,
    ReportExpr,
    ReportGenerator,
    ReportQuery,
    ScalarStringQuery,
    SsaQuery,
    StringQuery,
    TypeQuery,
    always,
    bound_ops,
    bound_ssa,
    capture_ops,
    capture_ssa,
    change,
    change_input,
    changed,
    changes,
    dialect,
    diff_op_input,
    has_attr,
    input,
    matches,
    op,
    op_expr,
    ops,
    report,
    require,
    result_type,
    same_value,
    string_attr_eq,
)

ReportValue: TypeAlias = (
    bool | int | float | str | list["ReportValue"] | dict[str, "ReportValue"]
)
ReportInput: TypeAlias = (
    ReportExpr
    | ReportQuery
    | StringQuery
    | ScalarStringQuery
    | CountQuery
    | bool
    | int
    | float
    | str
    | list["ReportInput"]
    | dict[str, "ReportInput"]
)

__all__ = [
    "AttributeQuery",
    "CaptureQuery",
    "ChangePredicate",
    "ChangeQuery",
    "CountQuery",
    "DiffCountQuery",
    "DiffOpQuery",
    "DiffStringQuery",
    "MapQuery",
    "MatchBuilder",
    "MatchPredicate",
    "OpQuery",
    "Predicate",
    "QueryExpr",
    "ReportExpr",
    "ReportGenerator",
    "ReportInput",
    "ReportQuery",
    "ReportValue",
    "ScalarStringQuery",
    "SsaQuery",
    "StringQuery",
    "TypeQuery",
    "always",
    "bound_ops",
    "bound_ssa",
    "capture_ops",
    "capture_ssa",
    "change",
    "change_input",
    "changed",
    "changes",
    "dialect",
    "diff_op_input",
    "has_attr",
    "input",
    "matches",
    "op",
    "op_expr",
    "ops",
    "report",
    "require",
    "result_type",
    "same_value",
    "string_attr_eq",
]

_T = TypeVar("_T")


class QueryExpr(ABC, Generic[_T]):
    """Common annotation and virtual base class for native query expressions."""

    @abstractmethod
    def __bool__(self) -> Never:
        raise TypeError("evaluate a query with document.query(expression)")


for _query_type in (
    SsaQuery,
    CaptureQuery,
    OpQuery,
    ReportQuery,
    StringQuery,
    CountQuery,
    ScalarStringQuery,
    TypeQuery,
    AttributeQuery,
    MapQuery,
    ChangeQuery,
    DiffOpQuery,
    DiffStringQuery,
    DiffCountQuery,
):
    QueryExpr.register(_query_type)
