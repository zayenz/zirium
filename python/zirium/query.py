"""Immutable queries evaluated in Rust by Document.query().

Use ops() for the whole document and input() for a nested query's input.
"""

from abc import ABC, abstractmethod
from typing import Generic, Never, TypeAlias, TypeVar

from ._zirium import (  # ty: ignore[unresolved-import]
    AttributeQuery,
    ChangePredicate,
    ChangeQuery,
    CountQuery,
    DiffCountQuery,
    DiffOpQuery,
    DiffStringQuery,
    MapQuery,
    OpQuery,
    Predicate,
    ReportExpr,
    ReportGenerator,
    ReportQuery,
    ScalarStringQuery,
    StringQuery,
    TypeQuery,
    always,
    bound_ops,
    change,
    change_input,
    changed,
    changes,
    dialect,
    diff_op_input,
    has_attr,
    input,
    op,
    op_expr,
    ops,
    report,
    require,
    result_type,
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
    "ChangePredicate",
    "ChangeQuery",
    "CountQuery",
    "DiffCountQuery",
    "DiffOpQuery",
    "DiffStringQuery",
    "MapQuery",
    "OpQuery",
    "Predicate",
    "QueryExpr",
    "ReportExpr",
    "ReportGenerator",
    "ReportInput",
    "ReportQuery",
    "ReportValue",
    "ScalarStringQuery",
    "StringQuery",
    "TypeQuery",
    "always",
    "bound_ops",
    "change",
    "change_input",
    "changed",
    "changes",
    "dialect",
    "diff_op_input",
    "has_attr",
    "input",
    "op",
    "op_expr",
    "ops",
    "report",
    "require",
    "result_type",
    "string_attr_eq",
]

_T = TypeVar("_T")


class QueryExpr(ABC, Generic[_T]):
    """Common annotation and virtual base class for native query expressions."""

    @abstractmethod
    def __bool__(self) -> Never:
        raise TypeError("evaluate a query with document.query(expression)")


for _query_type in (
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
