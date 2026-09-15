"""Immutable queries evaluated in Rust by Document.query().

Use ops() for the whole document and input() for a nested query's input.
"""

from abc import ABC, abstractmethod
from typing import Generic, Never, TypeVar

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
    ScalarStringQuery,
    StringQuery,
    TypeQuery,
    always,
    change,
    change_input,
    changed,
    changes,
    dialect,
    diff_op_input,
    has_attr,
    input,
    op,
    ops,
    result_type,
    string_attr_eq,
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
    "ScalarStringQuery",
    "StringQuery",
    "TypeQuery",
    "always",
    "change",
    "change_input",
    "changed",
    "changes",
    "dialect",
    "diff_op_input",
    "has_attr",
    "input",
    "op",
    "ops",
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
