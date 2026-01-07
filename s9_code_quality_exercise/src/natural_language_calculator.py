"""Natural language calculator main entry point."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from processor import process_line


def main():
    """Main entry point for the calculator."""
    if len(sys.argv) < 2:
        print("Error: No input file specified. Usage: python natural_language_calculator.py <input_file>")
        sys.exit(1)

    input_file = sys.argv[1]

    # Read input file
    try:
        with open(input_file, 'r') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"Error: Input file not found: {input_file}")
        sys.exit(1)
    except PermissionError:
        print(f"Error: Permission denied reading file: {input_file}")
        sys.exit(1)
    except IOError as e:
        print(f"Error: Failed to read input file: {e}")
        sys.exit(1)

    # Process each line
    results = []
    for line_num, line in enumerate(lines, start=1):
        line = line.strip()
        if line:
            try:
                result = process_line(line)
                results.append(result)
            except ValueError as e:
                print(f"Error: Invalid expression on line {line_num}: {line}")
                print(f"  Details: {e}")
                sys.exit(1)
            except ZeroDivisionError:
                print(f"Error: Division by zero on line {line_num}: {line}")
                sys.exit(1)
            except Exception as e:
                print(f"Error: Unexpected error processing line {line_num}: {line}")
                print(f"  Details: {e}")
                sys.exit(1)

    # Write output file
    output_file = input_file.replace('.txt', '_results.txt')
    try:
        with open(output_file, 'w') as f:
            for r in results:
                f.write(str(r) + '\n')
    except PermissionError:
        print(f"Error: Permission denied writing to file: {output_file}")
        sys.exit(1)
    except IOError as e:
        print(f"Error: Failed to write output file: {e}")
        sys.exit(1)

    print(f"Results written to {output_file}")


if __name__ == "__main__":
    main()
