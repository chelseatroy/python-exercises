"""Operation definitions and registry for the calculator."""
from typing import Callable


class Operation:
    """Represents a calculator operation."""

    def __init__(self,
                 keyword: str,
                 nested_keyword: str,
                 func: Callable[[float, float], float],
                 precedence_keyword: str):
        """
        Args:
            keyword: Simple operation keyword (e.g., "plus")
            nested_keyword: Nested operation delimiter (e.g., "plus the result of")
            func: Binary operation function
            precedence_keyword: Comma-separated keyword (e.g., "plus ")
        """
        self.keyword = keyword
        self.nested_keyword = nested_keyword
        self.nested_keyword_with_comma = f", {nested_keyword}"
        self.func = func
        self.precedence_keyword = precedence_keyword
        self.pattern = f" {keyword} "


def _add(a: float, b: float) -> float:
    return a + b


def _subtract(a: float, b: float) -> float:
    return a - b


def _multiply(a: float, b: float) -> float:
    return a * b


def _divide(a: float, b: float) -> float:
    return a / b


# Operation registry
OPERATIONS = {
    'plus': Operation('plus', 'plus the result of', _add, 'plus '),
    'minus': Operation('minus', 'minus the result of', _subtract, 'minus '),
    'times': Operation('times', 'times the result of', _multiply, 'times '),
    'divided by': Operation('divided by', 'divided by the result of', _divide, 'divided by '),
}


def get_operation_by_nested_keyword(line: str):
    """
    Find which operation keyword exists in the line for nested expressions.

    Returns:
        (operation, delimiter, has_comma): The operation, which delimiter matched,
                                           and whether it included a comma
    """
    for op in OPERATIONS.values():
        if op.nested_keyword_with_comma in line:
            return op, op.nested_keyword_with_comma, True
        if f" {op.nested_keyword}" in line:
            return op, f" {op.nested_keyword}", False
    return None, None, None


def get_operation_by_pattern(expression: str):
    """Find which operation keyword exists in the expression."""
    for op in OPERATIONS.values():
        if op.pattern in expression:
            return op
    return None
