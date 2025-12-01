import pytest
import os
import tempfile
import shutil
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from natural_language_calculator import main


@pytest.fixture
def test_dir():
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


def create_test_file(test_dir, filename, content):
    filepath = os.path.join(test_dir, filename)
    with open(filepath, 'w') as f:
        f.write(content)
    return filepath


def test_basic_file_processing(test_dir, capsys):
    input_content = "one plus three\neight minus two\n"
    input_file = create_test_file(test_dir, 'test_input.txt', input_content)

    exit_code = main(input_file)

    assert exit_code == 0

    output_file = input_file.replace('.txt', '_results.txt')
    assert os.path.exists(output_file)

    with open(output_file, 'r') as f:
        lines = f.readlines()

    assert len(lines) == 2
    assert lines[0].strip() == '4'
    assert lines[1].strip() == '6'


def test_all_sample_expressions(test_dir):
    input_content = """one plus three
eight minus two
the result of four plus one, minus five
six times five minus the result of nine times two
the result of ten plus three, minus the result of one plus seven
The result of three times three, divided by two
Three divided by nine
"""
    input_file = create_test_file(test_dir, 'expressions.txt', input_content)

    exit_code = main(input_file)

    assert exit_code == 0

    output_file = input_file.replace('.txt', '_results.txt')
    with open(output_file, 'r') as f:
        results = [line.strip() for line in f.readlines()]

    expected = ['4', '6', '0', '12', '5', '4.5', '0.33']
    assert results == expected


def test_empty_lines_ignored(test_dir):
    input_content = "one plus three\n\neight minus two\n\n"
    input_file = create_test_file(test_dir, 'with_blanks.txt', input_content)

    exit_code = main(input_file)

    assert exit_code == 0

    output_file = input_file.replace('.txt', '_results.txt')
    with open(output_file, 'r') as f:
        lines = f.readlines()

    assert len(lines) == 2
    assert lines[0].strip() == '4'
    assert lines[1].strip() == '6'


def test_output_filename_convention(test_dir):
    input_file = create_test_file(test_dir, 'my_math.txt', 'one plus one\n')

    exit_code = main(input_file)
    assert exit_code == 0

    expected_output = os.path.join(test_dir, 'my_math_results.txt')
    assert os.path.exists(expected_output)


def test_missing_input_file(test_dir, capsys):
    nonexistent_file = os.path.join(test_dir, 'does_not_exist.txt')

    exit_code = main(nonexistent_file)

    assert exit_code != 0

    captured = capsys.readouterr()
    assert len(captured.out) > 0 or len(captured.err) > 0


def test_no_arguments(capsys):
    exit_code = main()

    assert exit_code != 0

    captured = capsys.readouterr()
    assert len(captured.out) > 0


def test_invalid_expression_in_file(test_dir):
    input_content = "one plus three\ninvalid expression here\n"
    input_file = create_test_file(test_dir, 'invalid.txt', input_content)

    exit_code = main(input_file)

    assert exit_code != 0


def test_single_expression(test_dir):
    input_file = create_test_file(test_dir, 'single.txt', 'five times five\n')

    exit_code = main(input_file)

    assert exit_code == 0

    output_file = input_file.replace('.txt', '_results.txt')
    with open(output_file, 'r') as f:
        content = f.read().strip()

    assert content == '25'


def test_preserves_integer_vs_float_format(test_dir):
    input_content = "six divided by two\nthree divided by nine\n"
    input_file = create_test_file(test_dir, 'formatting.txt', input_content)

    exit_code = main(input_file)
    assert exit_code == 0

    output_file = input_file.replace('.txt', '_results.txt')
    with open(output_file, 'r') as f:
        lines = [line.strip() for line in f.readlines()]

    assert lines[0] == '3'
    assert '.' not in lines[0]

    assert lines[1] == '0.33'
    assert '.' in lines[1]


def test_success_message_printed(test_dir, capsys):
    input_file = create_test_file(test_dir, 'test.txt', 'one plus one\n')

    exit_code = main(input_file)

    assert exit_code == 0

    captured = capsys.readouterr()
    output_filename = input_file.replace('.txt', '_results.txt')
    assert 'results' in captured.out.lower()
    assert os.path.basename(output_filename) in captured.out


def test_multiple_runs_overwrite_output(test_dir):
    input_file = create_test_file(test_dir, 'test.txt', 'one plus one\n')
    output_file = input_file.replace('.txt', '_results.txt')

    exit_code = main(input_file)
    assert exit_code == 0

    with open(output_file, 'r') as f:
        first_result = f.read()

    with open(input_file, 'w') as f:
        f.write('two plus two\n')

    exit_code = main(input_file)
    assert exit_code == 0

    with open(output_file, 'r') as f:
        second_result = f.read()

    assert first_result != second_result
    assert second_result.strip() == '4'


def test_empty_input_file(test_dir):
    input_file = create_test_file(test_dir, 'empty.txt', '')

    exit_code = main(input_file)

    assert exit_code == 0

    output_file = input_file.replace('.txt', '_results.txt')
    assert os.path.exists(output_file)

    with open(output_file, 'r') as f:
        content = f.read()

    assert content == ''


def test_only_blank_lines(test_dir):
    input_file = create_test_file(test_dir, 'blanks.txt', '\n\n  \n\t\n')

    exit_code = main(input_file)

    assert exit_code == 0

    output_file = input_file.replace('.txt', '_results.txt')
    with open(output_file, 'r') as f:
        content = f.read()

    assert content == ''


def test_trailing_whitespace_in_expressions(test_dir):
    input_file = create_test_file(test_dir, 'whitespace.txt',
        'one plus three   \n  eight minus two\t\n')

    exit_code = main(input_file)

    assert exit_code == 0

    output_file = input_file.replace('.txt', '_results.txt')
    with open(output_file, 'r') as f:
        lines = [line.strip() for line in f.readlines()]

    assert lines == ['4', '6']


def test_subprocess_integration():
    script_path = os.path.join(os.path.dirname(__file__), '..', 'src', 'natural_language_calculator.py')
    test_dir = tempfile.mkdtemp()

    try:
        input_file = create_test_file(test_dir, 'subprocess_test.txt', 'one plus three\n')

        result = subprocess.run(
            [sys.executable, script_path, input_file],
            capture_output=True,
            text=True
        )

        assert result.returncode == 0

        output_file = input_file.replace('.txt', '_results.txt')
        assert os.path.exists(output_file)

        with open(output_file, 'r') as f:
            content = f.read().strip()

        assert content == '4'
    finally:
        shutil.rmtree(test_dir)
