import unittest
import os
import tempfile
import shutil
from pathlib import Path
import subprocess
import sys


class TestFileOperations(unittest.TestCase):
    """Test the file I/O operations of the main calculator script."""

    def setUp(self):
        """Create a temporary directory for test files."""
        self.test_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.test_dir)
        # Get path to the script in ../src/
        self.script_path = os.path.join(os.path.dirname(__file__), '..', 'src', 'natural_language_calculator.py')

    def _create_test_file(self, filename, content):
        """Helper to create a test input file."""
        filepath = os.path.join(self.test_dir, filename)
        with open(filepath, 'w') as f:
            f.write(content)
        return filepath

    def _run_calculator(self, input_file):
        """Helper to run the calculator script and capture output."""
        result = subprocess.run(
            [sys.executable, self.script_path, input_file],
            capture_output=True,
            text=True
        )
        return result

    def test_basic_file_processing(self):
        """Test that the script reads input and writes output correctly."""
        # Create input file
        input_content = "one plus three\neight minus two\n"
        input_file = self._create_test_file('test_input.txt', input_content)

        # Run the calculator
        result = self._run_calculator(input_file)

        # Check it succeeded
        self.assertEqual(result.returncode, 0,
            f"Script failed with error: {result.stderr}")

        # Check output file was created
        output_file = input_file.replace('.txt', '_results.txt')
        self.assertTrue(os.path.exists(output_file),
            "Output file was not created")

        # Check output file contents
        with open(output_file, 'r') as f:
            lines = f.readlines()

        self.assertEqual(len(lines), 2)
        self.assertEqual(lines[0].strip(), '4')
        self.assertEqual(lines[1].strip(), '6')

    def test_all_sample_expressions(self):
        """Test processing the complete sample file."""
        input_content = """one plus three
eight minus two
the result of four plus one, minus five
six times five minus the result of nine times two
the result of ten plus three, minus the result of one plus seven
The result of three times three, divided by two
Three divided by nine
"""
        input_file = self._create_test_file('expressions.txt', input_content)

        # Run the calculator
        result = self._run_calculator(input_file)

        # Check it succeeded
        self.assertEqual(result.returncode, 0,
            f"Script failed with error: {result.stderr}")

        # Check output
        output_file = input_file.replace('.txt', '_results.txt')
        with open(output_file, 'r') as f:
            results = [line.strip() for line in f.readlines()]

        expected = ['4', '6', '0', '12', '5', '4.5', '0.33']
        self.assertEqual(results, expected)

    def test_empty_lines_ignored(self):
        """Test that empty lines in input are skipped."""
        input_content = "one plus three\n\neight minus two\n\n"
        input_file = self._create_test_file('with_blanks.txt', input_content)

        result = self._run_calculator(input_file)

        self.assertEqual(result.returncode, 0)

        output_file = input_file.replace('.txt', '_results.txt')
        with open(output_file, 'r') as f:
            lines = f.readlines()

        # Should only have 2 results, not 4
        self.assertEqual(len(lines), 2)
        self.assertEqual(lines[0].strip(), '4')
        self.assertEqual(lines[1].strip(), '6')

    def test_output_filename_convention(self):
        """Test that output filename follows the _results.txt convention."""
        input_file = self._create_test_file('my_math.txt', 'one plus one\n')

        result = self._run_calculator(input_file)
        self.assertEqual(result.returncode, 0)

        # Output should be my_math_results.txt
        expected_output = os.path.join(self.test_dir, 'my_math_results.txt')
        self.assertTrue(os.path.exists(expected_output))

    def test_missing_input_file(self):
        """Test error handling when input file doesn't exist."""
        nonexistent_file = os.path.join(self.test_dir, 'does_not_exist.txt')

        result = self._run_calculator(nonexistent_file)

        # Should exit with error
        self.assertNotEqual(result.returncode, 0,
            "Script should fail when input file doesn't exist")

        # Should print error message (even if it's just "Error")
        self.assertTrue(len(result.stdout) > 0 or len(result.stderr) > 0,
            "Script should print error message")

    def test_no_arguments(self):
        """Test error handling when no input file is provided."""
        result = subprocess.run(
            [sys.executable, self.script_path],
            capture_output=True,
            text=True
        )

        # Should exit with error
        self.assertNotEqual(result.returncode, 0,
            "Script should fail when no arguments provided")

        # Should print error message
        self.assertTrue(len(result.stdout) > 0 or len(result.stderr) > 0,
            "Script should print error message")

    def test_invalid_expression_in_file(self):
        """Test error handling when file contains invalid expression."""
        input_content = "one plus three\ninvalid expression here\n"
        input_file = self._create_test_file('invalid.txt', input_content)

        result = self._run_calculator(input_file)

        # Should exit with error (current implementation does this)
        self.assertNotEqual(result.returncode, 0,
            "Script should fail on invalid expression")

    def test_single_expression(self):
        """Test processing a file with just one expression."""
        input_file = self._create_test_file('single.txt', 'five times five\n')

        result = self._run_calculator(input_file)

        self.assertEqual(result.returncode, 0)

        output_file = input_file.replace('.txt', '_results.txt')
        with open(output_file, 'r') as f:
            content = f.read().strip()

        self.assertEqual(content, '25')

    def test_preserves_integer_vs_float_format(self):
        """Test that output correctly formats integers vs floats."""
        input_content = "six divided by two\nthree divided by nine\n"
        input_file = self._create_test_file('formatting.txt', input_content)

        result = self._run_calculator(input_file)
        self.assertEqual(result.returncode, 0)

        output_file = input_file.replace('.txt', '_results.txt')
        with open(output_file, 'r') as f:
            lines = [line.strip() for line in f.readlines()]

        # 6/2 = 3 (integer)
        self.assertEqual(lines[0], '3')
        self.assertNotIn('.', lines[0], "Integer result should not have decimal point")

        # 3/9 = 0.33 (float)
        self.assertEqual(lines[1], '0.33')
        self.assertIn('.', lines[1], "Float result should have decimal point")

    def test_success_message_printed(self):
        """Test that script prints success message with output filename."""
        input_file = self._create_test_file('test.txt', 'one plus one\n')

        result = self._run_calculator(input_file)

        self.assertEqual(result.returncode, 0)

        # Should print something about results being written
        output_filename = input_file.replace('.txt', '_results.txt')
        self.assertIn('results', result.stdout.lower(),
            "Should print message about results")
        self.assertIn(os.path.basename(output_filename), result.stdout,
            "Should mention output filename")

    def test_multiple_runs_overwrite_output(self):
        """Test that running script twice overwrites previous output."""
        input_file = self._create_test_file('test.txt', 'one plus one\n')
        output_file = input_file.replace('.txt', '_results.txt')

        # First run
        result1 = self._run_calculator(input_file)
        self.assertEqual(result1.returncode, 0)

        with open(output_file, 'r') as f:
            first_result = f.read()

        # Modify input file
        with open(input_file, 'w') as f:
            f.write('two plus two\n')

        # Second run
        result2 = self._run_calculator(input_file)
        self.assertEqual(result2.returncode, 0)

        with open(output_file, 'r') as f:
            second_result = f.read()

        # Results should be different
        self.assertNotEqual(first_result, second_result)
        self.assertEqual(second_result.strip(), '4')


class TestEdgeCasesFileIO(unittest.TestCase):
    """Test edge cases in file I/O."""

    def setUp(self):
        """Create a temporary directory for test files."""
        self.test_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.test_dir)
        self.script_path = os.path.join(os.path.dirname(__file__), '..', 'src', 'natural_language_calculator.py')

    def _create_test_file(self, filename, content):
        """Helper to create a test input file."""
        filepath = os.path.join(self.test_dir, filename)
        with open(filepath, 'w') as f:
            f.write(content)
        return filepath

    def _run_calculator(self, input_file):
        """Helper to run the calculator script."""
        result = subprocess.run(
            [sys.executable, self.script_path, input_file],
            capture_output=True,
            text=True
        )
        return result

    def test_empty_input_file(self):
        """Test processing an empty file."""
        input_file = self._create_test_file('empty.txt', '')

        result = self._run_calculator(input_file)

        # Should succeed (no expressions to process)
        self.assertEqual(result.returncode, 0)

        # Output file should exist but be empty
        output_file = input_file.replace('.txt', '_results.txt')
        self.assertTrue(os.path.exists(output_file))

        with open(output_file, 'r') as f:
            content = f.read()

        self.assertEqual(content, '', "Empty input should produce empty output")

    def test_only_blank_lines(self):
        """Test file with only whitespace/blank lines."""
        input_file = self._create_test_file('blanks.txt', '\n\n  \n\t\n')

        result = self._run_calculator(input_file)

        self.assertEqual(result.returncode, 0)

        output_file = input_file.replace('.txt', '_results.txt')
        with open(output_file, 'r') as f:
            content = f.read()

        self.assertEqual(content, '', "Blank lines should be ignored")

    def test_trailing_whitespace_in_expressions(self):
        """Test that trailing whitespace doesn't break parsing."""
        input_file = self._create_test_file('whitespace.txt',
            'one plus three   \n  eight minus two\t\n')

        result = self._run_calculator(input_file)

        self.assertEqual(result.returncode, 0)

        output_file = input_file.replace('.txt', '_results.txt')
        with open(output_file, 'r') as f:
            lines = [line.strip() for line in f.readlines()]

        self.assertEqual(lines, ['4', '6'])


if __name__ == '__main__':
    unittest.main()
