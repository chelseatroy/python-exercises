import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from processor import process_line

def main(input_file=None):
    if input_file is None:
        if len(sys.argv) < 2:
            print("Error")
            return 1
        input_file = sys.argv[1]

    try:
        with open(input_file, 'r') as f:
            lines = f.readlines()
    except:
        print("Error")
        return 1

    results = []
    for line in lines:
        line = line.strip()
        if line:
            try:
                result = process_line(line)
                results.append(result)
            except:
                print("Error")
                return 1

    output_file = input_file.replace('.txt', '_results.txt')
    try:
        with open(output_file, 'w') as f:
            for r in results:
                f.write(str(r) + '\n')
    except:
        print("Error")
        return 1

    print(f"Results written to {output_file}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
