# Natural Language Calculator - Refactoring Exercise

## Running the Program
```bash
cd src
python natural_language_calculator.py sample_expressions.txt
```

This will create a file called `sample_expressions_results.txt` with the calculated results.

## Your Task
Refactor this code to improve its quality. As you work, consider the concepts we've discussed:

### 1. Coupling and Cohesion
- **Coupling**: How dependent are the modules on each other?
- **Cohesion**: Does each module have a single, clear purpose?

Look for:
- Are the files in `src/` organized logically?
- Does each function have a single responsibility?
- Are there dependencies that shouldn't exist?

### 2. Code Heuristics (Remember: They're Guidelines, Not Rules!)

Look for:
- Repetitive code patterns (especially in `src/processor.py`)
- Long functions that could be broken up
- Opportunities for meaningful abstractions

### 3. Error Messages and Context Preservation
Error messages are documentation for future developers!

Look for:
- Vague error messages in `src/natural_language_calculator.py`
- Silent failures
- Places where debugging would be difficult

## Questions to Guide Your Refactoring

1. **If this code breaks in production, how would you debug it?**
2. **If you need to add a new number word (e.g., "eleven"), how many places do you need to change?**
3. **If you need to add a new operation (e.g., "modulo"), how hard would it be?**
4. **Can you understand what each function does without reading the implementation?**

## Using the Test Suite

The `tests/` directory contains tests. 

```bash
# Run all tests
python -m unittest discover -v

# Run just unit tests
python -m unittest tests.test_calculator -v
```

## Challenge:
Refactor the code and be prepared to explain:
- What specific problems you identified
- Why they were problems
- How your refactoring improves the code
- Any tradeoffs you made
