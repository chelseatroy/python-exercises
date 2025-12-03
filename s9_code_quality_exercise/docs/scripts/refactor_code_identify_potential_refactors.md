# Identify Potential Refactorings

The purpose of this action is to read the code and identify potential refactorings as well as whether or not the code is good enough to STOP refactoring. 

## Output

The output will be either to:

IF the code is good enough, THEN remove the `__potential_refactorings.md` file
Otherwise, create a new `__potential_refactorings.md` file with a few of the top places to refactor. 

Suggestions should include where, what, and why. Be brief. No examples. 