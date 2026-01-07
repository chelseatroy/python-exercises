"""Core calculation logic for the natural language calculator."""
from utils import get_num
from operations import OPERATIONS, get_operation_by_nested_keyword, get_operation_by_pattern


def process_line(line):
    """
    Process a single line of natural language math expression.

    Args:
        line: Natural language math expression

    Returns:
        Calculated result as int (if whole number) or float (rounded to 2 decimals)
    """
    normalized = _normalize_input(line)
    result = _evaluate_expression(normalized)
    return _format_result(result)


def _normalize_input(line):
    """Normalize input by lowercasing and removing 'the result of' prefix."""
    line = line.lower()
    if line.startswith('the result of '):
        line = line[14:]
    return line


def _is_nested_expression(line):
    """Check if expression contains nested 'result of' operations."""
    return 'the result of' in line


def _evaluate_expression(line):
    """Evaluate expression, handling both nested and simple operations."""
    if _is_nested_expression(line):
        return _evaluate_nested(line)
    else:
        return calc_simple(line)


def _evaluate_nested(line):
    """
    Evaluate nested expression like 'X minus the result of Y'.

    Uses operation registry to handle all operators uniformly.
    """
    op, delimiter, _ = get_operation_by_nested_keyword(line)
    if not op:
        raise ValueError(f"No valid operation found in nested expression: {line}")

    parts = line.split(delimiter, 1)
    left = parts[0]
    right = parts[1]

    left_val = calc_simple(left)
    right_val = calc_simple(right)

    return op.func(left_val, right_val)


def _format_result(result):
    """Format result as int if whole number, otherwise round to 2 decimals."""
    if result == int(result):
        return int(result)
    else:
        return round(result, 2)


def calc_simple(expression):
    """
    Calculate a simple expression (no nested 'result of' clauses).

    Handles both comma-separated precedence and space-separated operations.
    """
    expression = expression.strip()

    # Handle comma-separated operations (precedence)
    if ', ' in expression:
        return _evaluate_comma_precedence(expression)

    # Handle space-separated simple operations
    return _evaluate_simple_operation(expression)


def _evaluate_comma_precedence(expression):
    """
    Evaluate comma-separated expression like 'four plus one, minus five'.

    Comma creates precedence: left side evaluated first.
    """
    parts = expression.split(', ', 1)
    left = parts[0]
    rest = parts[1]

    left_val = calc_simple(left)

    # Find which operation is in the rest
    for op in OPERATIONS.values():
        if rest.startswith(op.precedence_keyword):
            right_val = get_num(rest[len(op.precedence_keyword):])
            return op.func(left_val, right_val)

    raise ValueError(f"No valid operation found in precedence expression: {rest}")


def _evaluate_simple_operation(expression):
    """Evaluate simple operation like 'one plus three'."""
    op = get_operation_by_pattern(expression)

    if not op:
        # Might be just a number
        return get_num(expression)

    parts = expression.split(op.pattern, 1)
    num1 = get_num(parts[0].strip())
    num2 = get_num(parts[1].strip())

    return op.func(num1, num2)
