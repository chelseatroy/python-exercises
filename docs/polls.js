// The polls shown on the page, in the order they appear in s1_Intro_Examples.ipynb.
//
// Each poll's `id` is the key its votes are stored under, so changing an id
// orphans that poll's existing votes. Reordering or editing `options` is fine
// between class sessions, but not while people are voting: votes are stored as
// the index of the chosen option.

export const POLLS = [
  {
    id: "eight-equals",
    scenario: "Scenario 1",
    code: "eight = 8\neight_again = 8",
    question: "Does eight == eight_again?",
    options: ["True", "False", "It raises an error"],
  },
  {
    id: "eight-is",
    scenario: "Scenario 1",
    code: "eight = 8\neight_again = 8",
    question: "Do you think eight is eight_again?",
    options: ["True", "False", "It raises an error"],
  },
  {
    id: "257-equals",
    scenario: "Scenario 1",
    code: "two_hundred_fifty_seven = 257\ntwo_hundred_fifty_seven_again = 257",
    question: "Does two_hundred_fifty_seven == two_hundred_fifty_seven_again?",
    options: ["True", "False", "It raises an error"],
  },
  {
    id: "257-is",
    scenario: "Scenario 1",
    code: "two_hundred_fifty_seven = 257\ntwo_hundred_fifty_seven_again = 257",
    question: "Do you think two_hundred_fifty_seven is two_hundred_fifty_seven_again?",
    options: ["True", "False", "It raises an error"],
  },
  {
    id: "pop-alias",
    scenario: "Scenario 2",
    code: "a = [0,1,2]\nb = a\n\na.pop()\nb",
    question: "What does this code show for b?",
    options: ["[0, 1, 2]", "[0, 1]", "2", "It raises an error"],
  },
  {
    id: "chained-assignment",
    scenario: "Scenario 3",
    code: "x = y = [2]\nx = [3]\n\ny",
    question: "What will y be here?",
    options: ["[2]", "[3]", "[2, 3]", "It raises an error"],
  },
  {
    id: "matrix-before",
    scenario: "Scenario 4",
    code: "row = [0, 0]\nmatrix = [row, row]\nmatrix",
    question: "What do you think matrix looks like right now?",
    options: ["[[0, 0], [0, 0]]", "[0, 0, 0, 0]", "[[0, 0]]", "It raises an error"],
  },
  {
    id: "matrix-after",
    scenario: "Scenario 4",
    code: "row = [0, 0]\nmatrix = [row, row]\n\nmatrix[0][0] = 1\nmatrix",
    question: "How about NOW? What does matrix look like NOW?",
    options: ["[[1, 0], [0, 0]]", "[[1, 0], [1, 0]]", "[[1, 1], [0, 0]]", "It raises an error"],
  },
  {
    id: "mutable-default",
    scenario: "Scenario 5",
    code:
      'def append(item, to=[]):\n    to.append(item)\n    print(f"List: {to}, length: {len(to)}")\n\n' +
      "append(5, to=[1, 2, 3, 4])  # List: [1, 2, 3, 4, 5], length: 5\n" +
      "append(3)                   # List: [3], length: 1\n\n" +
      "append(4)\nappend(5)",
    question: "What does the append function do if you do not pass in a list to append to?",
    options: [
      "Starts a fresh empty list every call: [4], then [5]",
      "Keeps adding to the same list: [3, 4], then [3, 4, 5]",
      "It raises an error",
    ],
  },
  {
    id: "tuple-type",
    scenario: "Scenario 6",
    code: "t = (4, 6, [8,9])",
    question: "What data structure is this creating? In other words, what type is t?",
    options: ["tuple", "list", "set", "dict"],
  },
  {
    id: "tuple-after-error",
    scenario: "Scenario 6",
    code:
      "t = (4, 6, [8,9])\n\nt[2] += [10]\n" +
      "# TypeError: 'tuple' object does not support item assignment\n\nt",
    question: "So what is t right now?",
    options: ["(4, 6, [8, 9])", "(4, 6, [8, 9, 10])", "(4, 6, [8, 9], [10])", "t no longer exists"],
  },
  {
    id: "loop-target",
    scenario: "Scenario 7",
    code: "x = {'a': 1, 'b': 2}\ny = {}\n\nfor k, y[k] in x.items():\n    pass\n\ny",
    question: "What do you think y is right now?",
    options: ["{}", "{'a': 1, 'b': 2}", "{'b': 2}", "It raises an error"],
  },
];
