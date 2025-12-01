from utils import *

def normalize_line(line):
    line = line.lower()
    if line.startswith('the result of '):
        line = line[14:]
    return line

def process_line(line):
    line = normalize_line(line)

    # Check if it has nested result
    if ', minus the result of ' in line or ', plus the result of ' in line or ', times the result of ' in line or ', divided by the result of ' in line or ' minus the result of ' in line or ' plus the result of ' in line or ' times the result of ' in line or ' divided by the result of ' in line:
        # Handle nested operations
        if ', minus the result of ' in line:
            parts = line.split(', minus the result of ')
            left = parts[0]
            right = parts[1]
            left_val = calc_simple(left)
            right_val = calc_simple(right)
            result = left_val - right_val
        elif ' minus the result of ' in line:
            parts = line.split(' minus the result of ')
            left = parts[0]
            right = parts[1]
            left_val = calc_simple(left)
            right_val = calc_simple(right)
            result = left_val - right_val
        elif ', plus the result of ' in line:
            parts = line.split(', plus the result of ')
            left = parts[0]
            right = parts[1]
            left_val = calc_simple(left)
            right_val = calc_simple(right)
            result = left_val + right_val
        elif ' plus the result of ' in line:
            parts = line.split(' plus the result of ')
            left = parts[0]
            right = parts[1]
            left_val = calc_simple(left)
            right_val = calc_simple(right)
            result = left_val + right_val
        elif ', times the result of ' in line:
            parts = line.split(', times the result of ')
            left = parts[0]
            right = parts[1]
            left_val = calc_simple(left)
            right_val = calc_simple(right)
            result = left_val * right_val
        elif ' times the result of ' in line:
            parts = line.split(' times the result of ')
            left = parts[0]
            right = parts[1]
            left_val = calc_simple(left)
            right_val = calc_simple(right)
            result = left_val * right_val
        elif ', divided by the result of ' in line:
            parts = line.split(', divided by the result of ')
            left = parts[0]
            right = parts[1]
            left_val = calc_simple(left)
            right_val = calc_simple(right)
            result = left_val / right_val
        elif ' divided by the result of ' in line:
            parts = line.split(' divided by the result of ')
            left = parts[0]
            right = parts[1]
            left_val = calc_simple(left)
            right_val = calc_simple(right)
            result = left_val / right_val
    else:
        # Simple operation
        result = calc_simple(line)

    # Format the result
    if result == int(result):
        return int(result)
    else:
        return round(result, 2)

def calc_simple(expression):
    expression = expression.strip()

    # Handle comma-separated operations (precedence)
    if ', ' in expression:
        parts = expression.split(', ')
        left = parts[0]
        rest = parts[1]
        left_val = calc_simple(left)
        # Now parse the rest to see what operation
        if rest.startswith('minus '):
            right_val = get_num(rest[6:])
            return left_val - right_val
        elif rest.startswith('plus '):
            right_val = get_num(rest[5:])
            return left_val + right_val
        elif rest.startswith('times '):
            right_val = get_num(rest[6:])
            return left_val * right_val
        elif rest.startswith('divided by '):
            right_val = get_num(rest[11:])
            return left_val / right_val

    # Try to parse the expression
    if ' plus ' in expression:
        parts = expression.split(' plus ')
        num1 = get_num(parts[0].strip())
        num2 = get_num(parts[1].strip())
        return num1 + num2
    elif ' minus ' in expression:
        parts = expression.split(' minus ')
        num1 = get_num(parts[0].strip())
        num2 = get_num(parts[1].strip())
        return num1 - num2
    elif ' times ' in expression:
        parts = expression.split(' times ')
        num1 = get_num(parts[0].strip())
        num2 = get_num(parts[1].strip())
        return num1 * num2
    elif ' divided by ' in expression:
        parts = expression.split(' divided by ')
        num1 = get_num(parts[0].strip())
        num2 = get_num(parts[1].strip())
        return num1 / num2
    else:
        # Might be just a number
        return get_num(expression)
