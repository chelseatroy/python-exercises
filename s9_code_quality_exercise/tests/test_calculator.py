import unittest
import sys
import os

# Add src directory to path so we can import from it
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from processor import process_line, calc_simple
from utils import get_num


class TestGetNum(unittest.TestCase):
    """Test the word-to-number conversion."""

    def test_word_numbers(self):
        """Test that word numbers convert correctly."""
        self.assertEqual(get_num('zero'), 0)
        self.assertEqual(get_num('one'), 1)
        self.assertEqual(get_num('five'), 5)
        self.assertEqual(get_num('ten'), 10)

    def test_word_numbers_case_insensitive(self):
        """Test that word numbers work regardless of case."""
        self.assertEqual(get_num('One'), 1)
        self.assertEqual(get_num('FIVE'), 5)
        self.assertEqual(get_num('TEN'), 10)

    def test_digit_strings(self):
        """Test that digit strings convert correctly."""
        self.assertEqual(get_num('0'), 0)
        self.assertEqual(get_num('5'), 5)
        self.assertEqual(get_num('10'), 10)

    def test_whitespace_handling(self):
        """Test that leading/trailing whitespace is handled."""
        self.assertEqual(get_num('  five  '), 5)
        self.assertEqual(get_num('\ttwo\n'), 2)

    def test_invalid_word_raises_error(self):
        """Test that invalid words raise an error."""
        with self.assertRaises(ValueError):
            get_num('eleven')
        with self.assertRaises(ValueError):
            get_num('abc')


class TestCalcSimple(unittest.TestCase):
    """Test simple calculation operations."""

    def test_addition(self):
        """Test simple addition."""
        self.assertEqual(calc_simple('one plus three'), 4)
        self.assertEqual(calc_simple('five plus five'), 10)

    def test_subtraction(self):
        """Test simple subtraction."""
        self.assertEqual(calc_simple('eight minus two'), 6)
        self.assertEqual(calc_simple('ten minus one'), 9)

    def test_multiplication(self):
        """Test simple multiplication."""
        self.assertEqual(calc_simple('three times three'), 9)
        self.assertEqual(calc_simple('six times five'), 30)

    def test_division(self):
        """Test simple division."""
        self.assertEqual(calc_simple('six divided by two'), 3)
        self.assertAlmostEqual(calc_simple('three divided by nine'), 0.333, places=2)

    def test_just_a_number(self):
        """Test that a single number is returned as-is."""
        self.assertEqual(calc_simple('five'), 5)
        self.assertEqual(calc_simple('10'), 10)

    def test_precedence_with_comma(self):
        """Test that comma creates precedence (calculates left side first)."""
        # "four plus one, minus five" means "(4 + 1) - 5" = 0
        self.assertEqual(calc_simple('four plus one, minus five'), 0)
        # "three times three, divided by two" means "(3 * 3) / 2" = 4.5
        self.assertEqual(calc_simple('three times three, divided by two'), 4.5)
        # "two times three, plus one" means "(2 * 3) + 1" = 7
        self.assertEqual(calc_simple('two times three, plus one'), 7)
        # "five plus two, times three" means "(5 + 2) * 3" = 21
        self.assertEqual(calc_simple('five plus two, times three'), 21)


class TestProcessLine(unittest.TestCase):
    """Test the full line processing including result formatting."""

    def test_simple_addition(self):
        """Test simple addition returns integer when appropriate."""
        result = process_line('one plus three')
        self.assertEqual(result, 4)
        self.assertIsInstance(result, int)

    def test_simple_subtraction(self):
        """Test simple subtraction."""
        result = process_line('eight minus two')
        self.assertEqual(result, 6)
        self.assertIsInstance(result, int)

    def test_division_with_decimal(self):
        """Test that division returns rounded decimal when needed."""
        result = process_line('three divided by nine')
        self.assertEqual(result, 0.33)
        self.assertIsInstance(result, float)

    def test_division_returning_integer(self):
        """Test that division returns integer when result is whole number."""
        result = process_line('six divided by two')
        self.assertEqual(result, 3)
        self.assertIsInstance(result, int)

    def test_nested_with_comma_minus(self):
        """Test nested operation: 'result of X, minus five'."""
        result = process_line('the result of four plus one, minus five')
        self.assertEqual(result, 0)
        self.assertIsInstance(result, int)

    def test_nested_with_comma_divided(self):
        """Test nested operation with division."""
        result = process_line('The result of three times three, divided by two')
        self.assertEqual(result, 4.5)
        self.assertIsInstance(result, float)

    def test_nested_with_result_of_minus(self):
        """Test nested: 'X minus the result of Y'."""
        # "six times five minus the result of nine times two"
        # = 30 - 18 = 12
        result = process_line('six times five minus the result of nine times two')
        self.assertEqual(result, 12)
        self.assertIsInstance(result, int)

    def test_nested_both_sides(self):
        """Test nested on both sides: 'result of X, minus result of Y'."""
        # "result of ten plus three, minus result of one plus seven"
        # = 13 - 8 = 5
        result = process_line('the result of ten plus three, minus the result of one plus seven')
        self.assertEqual(result, 5)
        self.assertIsInstance(result, int)

    def test_nested_both_sides_plus(self):
        """Test nested on both sides: 'result of X, plus result of Y'."""
        result = process_line('the result of two plus three, plus the result of one plus four')
        self.assertEqual(result, 10)
        self.assertIsInstance(result, int)

    def test_nested_with_result_of_plus(self):
        """Test nested: 'X plus the result of Y'."""
        result = process_line('six plus the result of two times three')
        self.assertEqual(result, 12)
        self.assertIsInstance(result, int)

    def test_nested_both_sides_times(self):
        """Test nested on both sides: 'result of X, times result of Y'."""
        result = process_line('the result of two plus three, times the result of one plus one')
        self.assertEqual(result, 10)
        self.assertIsInstance(result, int)

    def test_nested_with_result_of_times(self):
        """Test nested: 'X times the result of Y'."""
        result = process_line('three times the result of two plus two')
        self.assertEqual(result, 12)
        self.assertIsInstance(result, int)

    def test_case_insensitive(self):
        """Test that capitalization doesn't matter."""
        result = process_line('ONE PLUS THREE')
        self.assertEqual(result, 4)

        result = process_line('The Result Of Three Times Three, Divided By Two')
        self.assertEqual(result, 4.5)

    def test_strips_result_of_prefix(self):
        """Test that 'the result of' prefix is properly stripped."""
        # With and without prefix should give same result
        result1 = process_line('the result of one plus three')
        result2 = process_line('one plus three')
        self.assertEqual(result1, result2)


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and potential error conditions."""

    def test_zero_operations(self):
        """Test operations involving zero."""
        self.assertEqual(process_line('zero plus five'), 5)
        self.assertEqual(process_line('ten minus zero'), 10)
        self.assertEqual(process_line('zero times five'), 0)

    def test_division_by_non_zero(self):
        """Test various division scenarios."""
        # Note: We're not testing division by zero because the current
        # implementation doesn't handle it - students might add this!
        self.assertEqual(process_line('ten divided by two'), 5)
        self.assertEqual(process_line('ten divided by three'), 3.33)

    def test_same_number_operations(self):
        """Test operations with the same number."""
        self.assertEqual(process_line('five plus five'), 10)
        self.assertEqual(process_line('five minus five'), 0)
        self.assertEqual(process_line('five times five'), 25)
        self.assertEqual(process_line('five divided by five'), 1)


class TestCompleteExamples(unittest.TestCase):
    """Test the complete examples from sample_expressions.txt."""

    def test_all_sample_expressions(self):
        """Test all expressions from the sample file."""
        test_cases = [
            ('one plus three', 4),
            ('eight minus two', 6),
            ('the result of four plus one, minus five', 0),
            ('six times five minus the result of nine times two', 12),
            ('the result of ten plus three, minus the result of one plus seven', 5),
            ('The result of three times three, divided by two', 4.5),
            ('Three divided by nine', 0.33),
        ]

        for expression, expected in test_cases:
            with self.subTest(expression=expression):
                result = process_line(expression)
                self.assertEqual(result, expected,
                    f"Expression '{expression}' should equal {expected}, got {result}")


if __name__ == '__main__':
    unittest.main()
