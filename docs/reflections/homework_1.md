---
title: Homework 1 Reflections
layout: syllabus
permalink: /reflections/homework-1/
---

<link rel="stylesheet" href="{{ '/reflections/reflections.css' | relative_url }}">
<script src="{{ '/reflections/reflections.js' | relative_url }}" defer></script>

Ten of you shared reflections on building the natural-language calculator. Each one is below, exactly as written, with names and links removed so you can read them all anonymously. First, here are some patterns that showed up when reading them all together.

## What we noticed across all the reflections

### At a glance

Each numbered column is one reflection, and each row is something reflections described doing. Click a technique to see who mentioned it, or click a number to see everything one reflection mentioned. A blank means the reflection didn't *mention* it, not that its author didn't do it.

<div id="approach-matrix" class="refl-widget"></div>

### Everyone started with the same toolkit, then hit the same wall

Most reflections begin the same way: a dictionary that turns number words into numbers, often a second one for operators, and later another dictionary for identifiers. That part went smoothly for nearly everyone. The trouble started with **"the result of"** and **commas**. One student put it most directly: "Whenever I thought I had something that worked, it would break on any instance of 'the result of.'"

<details class="refl-example" markdown="1">
<summary>Show a minimal example: the dictionaries</summary>

```python
NUMBERS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
           "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10}
OPERATORS = {"plus": "+", "minus": "-", "times": "*", "divided": "/"}

def tokenize(line):
    tokens = []
    for word in line.split():
        if word in NUMBERS:
            tokens.append(NUMBERS[word])
        elif word in OPERATORS:
            tokens.append(OPERATORS[word])
    return tokens

tokenize("six times four plus two")   # [6, '*', 4, '+', 2]

# "by" isn't in either dictionary, so it's skipped:
tokenize("ten divided by five")       # [10, '/', 5]
```

</details>

People handled it in two main ways:

- **Turn it into parentheses.** Several of you rewrote "the result of …" as an opening parenthesis during parsing. Then evaluating the line meant evaluating the innermost group first, which is a problem you already know how to solve.
- **Recurse.** Others had their evaluator call itself on the sub-expression. In a few cases the recursion also tracked a position in the line, so the outer call knew where to resume.

<details class="refl-example" markdown="1">
<summary>Show a minimal example: “the result of” as parentheses</summary>

Turn the phrase into `(` while tokenizing, and let a comma (or the end of the line) close it:

```python
def tokenize(line):
    words = line.replace(",", " , ").split()
    tokens, depth, i = [], 0, 0
    while i < len(words):
        if words[i:i + 3] == ["the", "result", "of"]:
            tokens.append("(")
            depth += 1
            i += 3
            continue
        word = words[i]
        if word == "," and depth > 0:
            # A comma closes the innermost "the result of".
            tokens.append(")")
            depth -= 1
        elif word in NUMBERS:
            tokens.append(NUMBERS[word])
        elif word in OPERATORS:
            tokens.append(OPERATORS[word])
        i += 1
    # Anything still open closes at the end of the line.
    return tokens + [")"] * depth

tokenize("the result of five times four, plus six")
# ['(', 5, '*', 4, ')', '+', 6]
```

Then evaluate the innermost group, replace it with its value, and repeat. (`calculate()` is the two-pass function under *Order of operations* below.)

```python
def evaluate(tokens):
    while "(" in tokens:
        # The first ")", and the "(" that matches it:
        close = tokens.index(")")
        open_ = max(i for i in range(close) if tokens[i] == "(")
        inner = calculate(tokens[open_ + 1:close])
        # Swap the group for its value. The list gets shorter here.
        tokens = tokens[:open_] + [inner] + tokens[close + 1:]
    return calculate(tokens)

evaluate(["(", 5, "*", 4, ")", "+", 6])   # 26
```

</details>

<details class="refl-example" markdown="1">
<summary>Show a minimal example: “the result of” by recursion</summary>

When the evaluator meets “the result of”, it calls itself. The call returns its value *and* the position where it stopped, so the outer call knows where to pick up. Commas are kept as markers: inside a sub-expression, a comma ends it.

```python
def evaluate(tokens, i=0, nested=False):
    """Evaluate from position i. Return (value, where to resume)."""
    parts = []
    while i < len(tokens):
        token = tokens[i]
        if token == "result_of":
            # The call returns its value and where it stopped.
            value, i = evaluate(tokens, i + 1, nested=True)
            parts.append(value)
        elif token == ",":
            if nested:
                # Inside "the result of", a comma ends it.
                return calculate(parts), i + 1
            i += 1    # at the top level, a comma is just a pause
        else:
            parts.append(token)
            i += 1
    return calculate(parts), i

# "six times the result of five plus one, plus two"
evaluate([6, "*", "result_of", 5, "+", 1, ",", "+", 2])
# (38, 9), meaning 6 * (5 + 1) + 2

# The same line with the comma thrown away:
evaluate([6, "*", "result_of", 5, "+", 1, "+", 2])
# (48, 8), meaning 6 * (5 + 1 + 2)
```

</details>

Commas turned out to matter more than they first appeared to. Several of you planned to ignore or delete them, and then found they mark where a sub-expression ends. One reflection describes this bug in detail. Once the commas were removed, "six times the result of five plus one plus two" and "six times the result of five plus one, plus two" gave the same answer. The fix was to keep the commas as markers. Other students used commas to break each line into chunks and carried a running result from one chunk to the next.

Try it: the same calculator, with the commas kept and with them thrown away.

<div id="comma-demo" class="refl-demo"></div>

<details class="refl-example" markdown="1">
<summary>Show a minimal example: splitting the line into chunks</summary>

Split on commas, solve the first chunk, then apply each later chunk to the running result. (`tokenize()` is the dictionary example above.)

```python
def evaluate_line(line):
    chunks = line.split(",")
    result = calculate(tokenize(chunks[0]))
    for chunk in chunks[1:]:
        op, *rest = tokenize(chunk)   # "plus two" -> "+", [2]
        result = calculate([result, op] + rest)
    return result

evaluate_line("six times four, plus two, divided by two")
# ((6 * 4) + 2) / 2 = 13.0
```

</details>

### Parsing decides how hard everything else is

The strongest reflections said this in different ways: **the shape you parse a line into determines how hard evaluation will be.** One student wrote that "the real difficulty was deciding what information preprocessing should keep and what it can safely remove." Another worked out the target token format by hand on paper before writing `parse()`, and was most proud of that function. A third was proud of moving from "an enormous match/case block with all possible words" to a cleaner way of parsing. If you got evaluation working by switching to a different intermediate representation, that switch was the real design decision.

### The case-sensitivity tradeoff

Several of you first tried, or planned, calling `.lower()` on the whole line. It makes "Let" and "let" (or "Five" and "five") match. Then the `f` vs. `F` example broke it, because identifiers are case-sensitive. This is a good small example of a tradeoff where you can't have both options for free. The reflections describe two ways around it:

1. List both capitalized and lowercase forms of number words and keywords explicitly. This duplicates entries but is easy to understand.
2. Keep two versions of each word, the original and a lowercased copy, and use whichever one the situation needs.

<details class="refl-example" markdown="1">
<summary>Show a minimal example: both ways around it</summary>

Listing both spellings:

```python
# ...and so on: every word, twice.
NUMBERS = {"one": 1, "One": 1, "two": 2, "Two": 2}
KEYWORDS = {"let", "Let", "calculate", "Calculate"}
```

Keeping two versions of each word, next to what lowercasing the whole line does:

```python
line = "Calculate F plus one"

# Lowercasing the whole line: "Calculate" matches now,
# but F has turned into f.
line.lower().split()   # ['calculate', 'f', 'plus', 'one']

# Keeping two versions of each word:
KEYWORDS = {"let", "equal", "calculate"}
for word in line.split():
    lower = word.lower()
    if lower in NUMBERS or lower in OPERATORS or lower in KEYWORDS:
        # Match keywords and numbers in any case.
        print("keyword or number:", lower)
    else:
        # Keep the original, so f and F stay different names.
        print("identifier:", word)
```

</details>

Try all of them on a short program. The third line should be an error, because nothing named `F` exists yet.

<div id="case-demo" class="refl-demo"></div>

### Order of operations: two passes

Students who handled operator precedence mostly used the same approach: one pass through the token list for multiplication and division, then a second for addition and subtraction. A few mentioned the cost (an extra loop) and the benefit (it's easier to follow). One reflection says honestly that it doesn't handle precedence at all.

<details class="refl-example" markdown="1">
<summary>Show a minimal example: two passes</summary>

```python
def calculate(tokens):
    # Pass 1: fold each * and / into the number just before it.
    first = [tokens[0]]
    for op, number in zip(tokens[1::2], tokens[2::2]):
        if op == "*":
            first[-1] *= number
        elif op == "/":
            first[-1] /= number
        else:
            first += [op, number]
    # Pass 2: what's left is only + and -, so go left to right.
    result = first[0]
    for op, number in zip(first[1::2], first[2::2]):
        result = result + number if op == "+" else result - number
    return result

calculate([2, "+", 3, "*", 4])   # 14, not 20
```

</details>

### Indexes are where the bugs were

Several of the reported struggles come down to **index management**. Lists got shorter when a parenthesized group was replaced with its result. Operators like "divided by" and "the result of" span several words. Positions had to be tracked across recursive calls. One student explained why they switched from a `for` loop to a `while` loop with a manual index: a `for` loop is more readable, but it couldn't consume multi-word operators. Another worked out the identifier indexes on scratch paper. If your bugs were off-by-one errors, you weren't the only one.

<details class="refl-example" markdown="1">
<summary>Show a minimal example: why a while loop</summary>

`for word in words` hands you one word at a time, so it can't take “divided” and “by” together. A `while` loop with your own index can:

```python
words = "ten divided by two".split()

tokens, i = [], 0
while i < len(words):
    if words[i:i + 2] == ["divided", "by"]:
        tokens.append("/")
        i += 2              # use up both words at once
    else:
        tokens.append(NUMBERS.get(words[i], words[i]))
        i += 1

tokens   # [10, '/', 2]
```

The parentheses example above has the other classic index bug built in: replacing a group with its value makes the list shorter, so every position after it moves.

</details>

### Plan first, or iterate?

This was the most common theme in the "what did you struggle with" answers, and you didn't all agree.

- Several of you wished you had **planned more before coding**: "it definitely would have helped to stop coding for a minute and write/draw out what I was trying to achieve." Others tried to handle every case at once and ended up rewriting, or built on a base that couldn't support the later features.
- One student said they don't know **how much planning is enough** and asked for a framework. That's an excellent question, and we'll come back to it in class.
- In contrast, one student **deliberately wrote a minimal parser first**, expecting to change it once evaluation showed what it needed. They changed it several times, and that went fine because they had planned for it.

Both groups are describing the same thing: you can't know the right representation until you've tried to use it. What differs is whether you expected the rework or were surprised by it.

### Specific vs. general

Several of you made a deliberate choice to support **the phrasings in the examples** rather than every possible English sentence. For example, "let c equal three" works, but "let c be equal to three" doesn't. The reflections that named this as a tradeoff and gave a reason ("which felt reasonable for hw1") show exactly the kind of thinking this course is about. A limitation you know about and chose is very different from one you didn't notice.

### Being honest about your own code

A few reflections openly criticize the author's own design. One student realized afterward that recursion wasn't needed, because their evaluator already found the innermost parentheses with a loop: "I fell a little short of an ideal solution by getting too attached to implementing recursion." Another described their code as "very, very messy" because they started late and added features to a weak foundation. Seeing these things clearly is the hard part, and writing them down is how you get better at it.

### The empty-file decision: a real split

<div id="empty-file-chart" class="refl-widget" markdown="1">

| Choice | How many | Reasoning given |
|---|---|---|
| Write an empty solution file | 3 | Consistency: every input produces an output with one line per problem, and zero problems means zero lines. Works well when processing many files in a row. Needs no special-case code. |
| Print a message | 3 | Silent success is confusing. Users should learn they gave a bad or empty input instead of finding a blank output. |
| Both: print a message *and* write an empty file | 1 | Get consistent output and also tell the user. |
| Raise / report an error | 1 | In real software, you'd want to know about suspicious input instead of letting blank files pile up. |
| Unclear / not done | 2 | |

</div>

There is no single right answer here. What matters is that the most convincing answers name **who** they're designing for: a script processing a batch of files, or a person typing a filename by hand. One student also treated a file that contains only blank lines as empty, which is an edge case worth considering.

<details class="refl-example" markdown="1">
<summary>Show a minimal example: the three choices</summary>

```python
# A file of only blank lines counts as empty, too.
problems = [line for line in open(path) if line.strip()]

if not problems:
    # Choice 1: write an empty solution file.
    # Zero problems, zero answers.
    open(solution_path, "w").close()

    # Choice 2: say so, so an empty result doesn't look like a crash.
    print(f"{path} has no problems in it, so there's nothing to solve.")

    # Choice 3: treat it as a mistake in the input.
    raise ValueError(f"{path} is empty")
```

</details>

### Other things worth mentioning

- **Learning Python itself** was the main struggle for some of you, including people coming from Rust and Go or returning to Python after a break. Spending more effort on syntax than on the math is normal this early.
- Several of you were proud of **breaking the program into helper functions** instead of writing one huge function, and of reusing one evaluation path for both plain expressions and identifier assignments.
- People who **wrote their own extra test inputs** (operator precedence, empty files, long chains of "the result of") found bugs that the provided examples didn't catch.

---

## The reflections

### Reflection 1

**Describe your overall process and steps for approaching this week's programming problem.**

I broke the problem into reading the file, solving each line, and writing the answers. I used a dictionary to match number words with numbers, then put the numbers and operations into a list. The program solves grouped parts first, then multiplication and division, and finally addition and subtraction. Variable names and their values go in another dictionary. I checked the results against both example files and used extra examples for empty files and operation order.

**Did you face any tradeoffs while developing your solution? If so, what were they, and how did you select your course of action?**

Solving everything from left to right would make the code shorter, but it would give the wrong answer when multiplication comes after addition. I used two passes through the list so multiplication and division happen first. This takes another loop, but the order is easier to follow. For the output, I followed the examples: whole numbers have no decimal places, and other answers have two decimal places.

**What about your solution to the programming problem did you struggle with the most?**

The hardest part was getting used to Python. I am not very familiar with it yet, so understanding the syntax and following what the lists and dictionaries were doing took more effort than the math itself. The lecture examples help me connect the syntax to what is happening in the program.

**What about your solution to the programming problem are you most proud of?**

I am happy that the same program handles both example files, including the variables. It can keep a value for later lines, treat f and F as different names, and clear the saved values when a new section starts. Those small details make a difference in whether the answers come out correctly.

**Don't forget to tell me why you picked the "empty file" solution you picked.**

I chose to create an empty output file. If there are no problems in the input, there are no answers to write. This keeps the behavior consistent with a normal input file and avoids treating an empty file as an error.

---

### Reflection 2

**Describe your overall process and steps for approaching this week’s programming problem.**

I started by reading the input file and splitting it into the individual math problems. I had to brainstorm a bit about how to convert the numbers into ints, because for digits 1-9 I could just do int(), but for digits >10 those would be split into two strings. I decided to create a dictionary that mapped number words to their numerical values, which would be used to kind of "create" the double digit numbers used in the problems. I aso created a dictionary for the four operations and mapped it to the corresponding operators. I worked on getting the basic arithmetic problems (math_problems.txt) working first using just these two dictionaries.

I then wrote a function to calculate expressions following order of operations. I basically had it calculate multiplication and division first, then moved on to addition and subtraction. I realized that this code got a bit repetitive after a while, which is why I created two separate helper functions for better readability. Once I moved on to the harder set of problems, I added support for phrases containing "the result of" by having the calculator recursively calculate the expression inside the phrase (this part was a bit tricky).

Once I got the first set of problems working, I started focusing on identifiers. I immediately knew I wanted to create a dictionary to store the identifier mappings. I then added a lot of if checks for "let", "calculate" and "new section," which cleared the identifiers dictionary. This was also where I added the .lower() because I knew let vs Let meant phrases/identifiers needed to be case-sensitive. Finally, I added the last bit of code that created the new solution file based on the name of the input file. I also came up with a variety of random test cases just to make sure my code worked correctly.

**Did you face any tradeoffs while developing your solution? If so, what were they, and how did you select your course of action?**

One tradeoff I faced was deciding the natural language parsing style and getting the "Let m equal six, calculate the result of ten plus three, minus m" line to output the correct answer. I could have tried to account for many different ways of phrasing, like "let m be the value of ...", but I made the assumption that I would not have to account for that and based it off the types of problem that were given ("let b equal ..."). I focused more on supporting the patterns shown in the assignment rather than creating something general, which meant my solution was simpler in the sense that I hard coded the parsing for identifiers.

**What about your solution to the programming problem did you struggle with the most?**

The part I struggled most with was definitely the sentence structure parsing step. The initial basic math problems were okay, I just had to experiment with .pop() and .insert() for a bit. However, when I got to the "the result of" and identifiers part, I realized I needed to start incorporating indexes. It took a good amount of testing to make sure the program was able to distinguish expressions assigned to identifiers and expressions that just needed to be calculated later. I thought I would have to worry more about operator precedence but that part was simpler than I expected.

**What about your solution to the programming problem are you most proud of?**

I think overall I'm most proud of actually finishing this assignment. I haven't coded in Python in a while so it took some time remembering a lot of the syntax (my most recent languages were Rust and Go which are decently different from Python) and functions. I did have to do a lot of googling at the beginning of this to remember basic things like how to read a file but it was a semi smooth process after that.

For something specific about my solution, I'd say that I'm impressed that I figured out the indexes for the identifiers part. I had to work it out on scratch paper, a little bit frustrating at first, but once I recognized the pattern and where to hardcode the indexes, I got everything working. I'm also proud that my initial basic arithmetic implementation worked consistently for every single test cast because that meant I didn't have to go back and fix something I wrote at the beginning, I could just keep implementing my calculator step by step and not worry about the foundation.

**Don’t forget to tell me why you picked the “empty file” solution you picked.**

I chose to print a message stating that the file is empty, and also provide an empty solution file. This way, the user will get notified that their file is empty, while also printing an expected output: an empty output file for an empty input file with no problems. If I had just outputted the empty file, it could look like the program failed, but with the printed message, the user will know that the file was just empty. I assumed that we would not have to print something if a file does not exist, but the terminal prints a FileNotFoundError already.

---

### Reflection 3

**Overall Process and Steps**

First, I wrote the overall framework for the task, which includes reading the lines from the file, parsing each line into a form that is easier to handle, and a main function that performs the calculation. At that stage I only wrote a basic version of the parsing function, because I expected to run into new challenges as I developed the main function, and I could then adjust the parsing to make those cases easier to handle.

Next, I worked on handling the actual commands, which fall into two categories. The first is assigning a value to a variable, which I implemented with a dictionary. The second is performing a calculation. I use the keyword "let" to distinguish between the two cases.

While implementing the calculation, I noticed that some expressions are recursive. For example, in "ten plus two, minus the result of ...", I need to compute the inner result first and then apply the minus operation, so I designed the calculation using recursive calls. To make the calculation logic more reusable, I needed the input commands to have a consistent format. So in addition to splitting each line by commas, I also split it by operators. For example, "six times five minus two" and "six times five, minus two" are both converted to "six | times five | minus two". This was the first version of my parsing: split by commas first, then by operators. With this version, my program passed all of the test examples.

However, this version had a bug: "six times the result of five plus one plus two" and "six times the result of five plus one, plus two" produced the same result. This happened because I discarded the commas, so the recursive call for "the result of ..." kept evaluating all the way to the end of the line. To fix this, I kept the commas in the command list as markers and added a parameter called `nested`, which indicates whether the current call is evaluating a "the result of ..." sub-expression that follows an operator (e.g., "minus the result of two plus one"). If `nested` is True, a comma ends the sub-expression: the function returns immediately and counts the comma as handled, so the caller skips over it. If `nested` is False, the comma is simply skipped and evaluation continues.

**Tradeoff: Handling Upper and Lower Case**

At first, I wanted to convert the whole input to lowercase, because that would make it much simpler to recognize number words and keywords such as "let", "the", and "calculate". However, this breaks the second test file, which requires variable names to be case-sensitive: after defining `f`, the program should report an error for `F`. If everything is lowercased, `F` becomes `f`, and the program would print the value of `f` instead of the error. So I kept the original case of the input and handled both capitalized and lowercase forms of keywords and number words explicitly. The cost is some duplication, such as listing every number word twice in my dictionary.

**Biggest Struggle: Preprocessing**

The hardest part was preprocessing. The calculation logic depends heavily on how each line is parsed: a consistent structure lets one function handle every case, while an inconsistent one creates many special cases. I adjusted my parsing several times as I developed the main function. For example, I split each line by both commas and operators so that similar sentences produce the same structure, but discarding the commas lost the boundaries of "the result of ..." sub-expressions and caused a bug, so I had to add them back as markers. The real difficulty was deciding what information preprocessing should keep and what it can safely remove.

**Most Proud Of: The Recursive Design**

I am most proud of how my program handles "the result of ..." with recursion. Because the parsed commands share a consistent structure, the same function can evaluate both a whole line and any nested sub-expression, so each operator only needs one piece of logic regardless of how deeply expressions are nested. When I found the comma bug, this design also made the fix small: I only needed to add the comma markers and a `nested` parameter, without rewriting the calculation logic.

**Empty File Handling**

For an empty input file, my program outputs an empty solution file. I chose this because it keeps the behavior consistent: every input file gets a solution file with one line per problem, so zero problems simply means zero lines. It also works better when running many input files in a row, since every input always has a matching output file, and it needs no special-case code.

---

### Reflection 4

**Describe your overall process and steps for approaching this week’s programming problem.**

*(No answer given.)*

**Did you face any tradeoffs while developing your solution? If so, what were they, and how did you select your course of action?**

I face a tradeoff when dealing with lowercase and uppercase letters in the input. Sometimes when a new line starts,
the first letter is ann uppercase letter. To deal with it, I turned the whole line into lowercase before processing it.
But then I found that the identifiers in the beginning of a line can also be turned into lowercase, which cause bugs.
To make sure all identifers' names are correct, I delete the line.lower() method. I hardcode uppercase numbers like "One"
and "Five" into the NUMBERS dictionary. I also hardcode the uppercase keywords in condition checking,
like: if words[i] == "let" or words[i] == "Let".

Although this method takes more time to put every uppercase word condition into checking process, I don't need to worry
about lowercase or uppercase matter not only in the beginning of the line, but also anywehre else in the line.

**What about your solution to the programming problem did you struggle with the most?**

I struggled most with coming up with a correct frame for processing input. At first, I thought that I can just simply
check keywords like numbers, operators, "the result of", split the operation into 3 parts: num1 + operator + num2, and
do the operation to get the result. But I found later that there're many nested expressions containing "the result of",
which can be infinitely nested. That's the time when I realize that I need to do recursions while also keeping track of
the current index in my line. So I wrote 2 functions, one is called "calculate_normal" for getting the result from simple
calculation like (num1 + operator + num2), and another function called "calculate_result" for doing recursions for calculating
the nested "the result of". My function "calculate_result" do recursion first and get the result for num1 and num2, and
then put them into function "calculate_normal".

**What about your solution to the programming problem are you most proud of?**

I think I'm skilled at using the appropriate data structures like dictionary and sets to handle data like numbers, operators,
and identifiers.

And also, I try to write helper functions rather than just put everything inside a huge function(which I used to do).
This makes debugging and reading my code easier. Although my read input function still looks huge, I think I'm making
improvements.

**Don’t forget to tell me why you picked the “empty file” solution you picked.**

I choose to print a message because although an empty file input is not an error, I don't want the program to finish without
any visible feedback. This may confuse the user wondering whether the program actually runs. So a clear message just make
everything easier and visible.

---

### Reflection 5

**Describe your overall process and steps for approaching this week’s programming problem.**

A: I break this question into small parts and started by thinking how I translate written texts into math equations and tried to find out some rules.
Since the number given in the input file(math_problems.txt) were restrained to be integers, and were all less or equal to ten, I found out that
the pattern for the written math problems should generally be number + operator + number + operator + number +...+ operator + number, and that comma was skipped when seen. Also, when the phrase "the result of ..." was observed in the text, parentheses should be used to prioritize a part of calculation.
With that being found, I started by coding for the mandatory part then I moved on to solve the bonus question. For the mandatory part, my overal idea
is that I want to translate written words into a list of tokens based on a given line in the input file. The list of tokens contain numbers, parentheses, and operators wrapped in single quotes. Then, I processed the list of tokens by calculate whatever was inside the parentheses, replace the parentheses with the calculated result, and calculate whatever was outside the parentheses with multiplication/division before addition/subtraction. At this point, I returned the result.
Now I moved on to the bonus question, I used a dictionary to store identifiers and their calculated values. When the program sees the word "Let", it calculates the expression after “equal” and saves the result under the identifier’s name. When that identifier appears later, the program retrieves its value from the dictionary. Identifiers are case-sensitive, and “new section” clears all previously stored identifiers. If an identifier does not exist, the program would output an error message.

**Did you face any tradeoffs while developing your solution? If so, what were they, and how did you select your course of action?**

A: At first, I was using a for loop to loop through each word in the sentence, as I thought it is more intuitive and more aligned with what human process language. Moreover, a for loop won't require too much effort on my end to manage index movement. However, later on I discovered that some of the operators were expressed in more than one word, such as "divided by" and "the result of", so that the for loop, though more manageble, was not the most flexible tool to process words. I then choose to use while loop with a int variable that indicated the position to manually control what word to process. Although it would require more careful index management, it gave me more control and flexibility over what word to process next.

**What about your solution to the programming problem did you struggle with the most?**

A: I struggled most with managing indexes while processing and modifying token lists. When I replaced a parenthesized expression with its result, the list became shorter, so I had to be careful when updating the position to avoid skipping tokens or accessing an index outside the list.

**What about your solution to the programming problem are you most proud of?**

A: I am most proud that my program can reuse the same calculation logic for both regular expressions and variables, and I intentionally planned it. To process calculation with variables, variables were first replaced with their stored values, so that they could be processed like regular arithmetic expressions.

**Don’t forget to tell me why you picked the “empty file” solution you picked.**

A: My solution would output an empty file when seeing one. I picked this solution because I want to keep the output of solution consistent: it would always produce a solution file whenever given an input file. (Plus it is easier to code if I output an empty file ;)

---

### Reflection 6

**Describe your overall process and steps for approaching this week’s programming problem.**

Step 1: I started by writing out one of the expressions and then working out
some simplification steps. This part took some time, especially since I
changed my mind a couple times about the ideal format to convert the
original text lines into. For example:

    the result of five times four, plus six divided by two
    [result, five, times, four, plus, six, divided, two]
    [(, 5, times, 4, ), plus, six, divided, two]

Step 2: Then I worked on a parse() function to achieve the results in the
last line above. I went back and forth between steps 1 and 2 a few times
while testing out different inputs to make sure that my parser could handle
any of the possible inputs. There were cases I had to consider or
reconsider, such as handling uppercase words and maintaining order of
operations. I added parentheses into the lines to help with the latter.

Here is an example input and output from my finalized parse() function:

    "Let f equal the result of four plus one, calculate six times one" ->
    ['let', 'f', '(', 4, 'plus', 1, ')', 'calculate', 6, 'times', 1]

Step 3: Once I was happy with parsing, I decided I wanted to use recursion
to deal with any nested parenthesis. This eventually turned into a process()
function for handling variable assignment and calling evaluate() to deal
with the arithmetic. This step took foreverrr, though it definitely would
have helped to stop coding for a minute and write/draw out what I was
trying to achieve. Despite having the parentheses, I had to make sure that
multiplication and division were done before addition and subtraction. I
also made the assumption that 'calculate' always comes after a variable
assignment based on the examples provided.

Step 4: Finally, I wrote main() and a number formatting helper to deal with
the input anf output files. This step was pretty straightforward since all
the processing logic was complete. I ran a lot of individual tests after
writing each major function to ensure they worked as intended.

**Did you face any tradeoffs while developing your solution? If so, what were they, and how did you select your course of action?**

I wanted to avoid iterating through the lines an excessive number of times,
but I did eventually have to settle for making multiple loops to keep the
logic correct, such as looping through an expression once for multiplication
and division, and then again for addition and subtraction. Iterative code
is usually easier to read, and in retrospect, I actually did not need to use
recursion at all because I already iteratively search for the innermost
parentheses at the start of evaluate(). So there was a simplicity vs
efficiency tradeoff, and I fell a little short of an ideal solution by
getting too attached to implementing recursion. I don't think it was needed
or useful in my code.

One other tradeoff is that the code is designed for a lot of specific cases
rather than being as general as possible. This allowed me to pass all the
tests that I anticipate, but I imagine there could be some test cases that
may not fall into the specific cases in my many if statements. So for the
tradeoff of passing tests vs generalizability, I did lean towards passing
tests, which felt reasonable for hw1.

**What about your solution to the programming problem did you struggle with the most?**

Writing evaluate() and process(). I had the idea in mind as mentioned above
to use recursion, but I had a hard time implementing it. There were also a
lot of different cases to consider and indexing that had to be done
carefully. The whole thing took me a lot longer than anticipated. My main
takeaway from that is that I have to spend a little more time planning
before I start coding.

**What about your solution to the programming problem are you most proud of?**

I am most proud of my parse() function because I think it did simplify the
evaluation logic, even though I did not end up making the best use of this
in the end. I took the time to work out how I wanted to parse each line in
plaintext, and that made it quick and easy to write the function.

**Don’t forget to tell me why you picked the “empty file” solution you picked.**

I think an error message is best for unexpected inputs. If this were for an
actual program, I would want to be notified that something may be wrong with
the input rather than allowing blank output files to stack up.

---

### Reflection 7

**Describe your overall process and steps for approaching this week’s programming problem.**

I had written somewhat of a parser in a previous Programming Languages class I've taken
where we looked into the formal terminolofy/process behind everything.

So going into this homework assignment, I knew that you had referred to this homework in
our Zoom last week as being somewhat of a parser. I was coming from that frame of reference,
but ultimately I do think it was quite a bit different from what I had done in my PL classes.
There was not really any required structure for me to follow that I already knew going into it,
so that meant that I was really thinking through all of the steps for decoding
and evaluating each of the lines of text.

I started off by reading through all of the requirements listed in the documentation.
I paid specific attention to all of the words in each line of the input file because I knew that
I would essentially have to use those as conditions for my control flow and my code .

**Did you face any tradeoffs while developing your solution? If so, what were they, and how did you select your course of action?**

There were definitely several trade-offs that I faced throughout this coding assignment.
Largely, all of these tradeoffs represent to me one of the ultimate coding tradeoffs: creating added modularized functionality which
ADDS COMPLEXITY, vs keeping things simpler but NOT getting as specific of functionality as I would have liked. Specifically:

On a broad level, the biggest tradeoffs were probably deciding exactly what to turn into a helper function versus just having in the main body of the code was a tradeoff.
Ultimately, I chose to build helper functions with functionality that needed to be used multiple times and, more importantly, in different
ways with a certain level of flexibility.

One trade-off was how I actually evaluated precedence with PEMDAS for multiplication, division, addition, and subtraction.
Obviously, this changes the control flow from left-to-right evaluation to following PEMDAS, for how numbers need to be
evaluated against each other. Ultimately, I decided on a two pass approach so that multiplication and division happened
before addition and subtraction. This made the evaluate function slightly more complex, but at the same time, I do think
this was a fairly simple solution.

Finally, another tradeoff was deciding when to make text lowercase versus keep it as is. This was something that I had
to rework whenever I really started building the variable handler, because variable names in different cases (like the example
with the lowercase f and uppercase F) will evaluate differently. Ultimately, I decided to handle this trade-off by creating
a version of the word as it originally came in, and another version in the lowercase so that I could use both when I needed to,
which gave me some added flexibility.

**What about your solution to the programming problem did you struggle with the most?**

The thing about my solution I struggled with the most was a lack of planning on my part for implementation,
thus creating a lot of re-work throughout the programming process.  Looking back, I wish that I had sketched out a rough
idea of what I was going to do on paper rather than just immediately diving into the code. I definitely had
to rework a lot of code I had already written, and this is honestly something that I personally have done a lot in programming
that I need to work on improving. It is hard for me to differentiate/recognize how much planning is necessary BEFORE writing any code,
vs. how much would be overkill. For instance, I don't necessarily think that writing all of this out in pseudocode would be the best
use of my time. And I also am very aware that sometimes there are complexity details that you can't really anticipate or account for
until you actually have a good portion of the code written. So if you have any advice for this or could help me think of a framework
that works and is commonplace in the professional environment, I would absolutely love that and that would help me a lot!!!

**What about your solution to the programming problem are you most proud of?**

The part of my solution that I'm most proud of is maybe not the most complex part:  how specifically to handle commas. I recognized that the
commas meant 1, the line contains separate but related pieces of logic which each need different handling, AND 2, the result of one piece may need to carry over into the next.
What I ended up doing was splitting the line into chunks seperated by comma, keeping track of the current result across those chunks, and then deciding what to do based on what the next chunk meant.
Figuring out the exact logic for handling this took me a while, but it was well worth it in the end and it was rewarding when I saw it all come together!

**Don’t forget to tell me why you picked the “empty file” solution you picked.**

I chose to print a helpful message because as a user of software who is very prone to sloppily giving
incorrect inputs, i know it is always helpful for me to be told directly when I’m doing this rather
than just seeing a blank page show up that doesn’t really hint at any user-error on my part.

---

### Reflection 8

**Describe your overall process and steps for approaching this week’s programming problem.**

First, I worked on reading the input file from sys.argv and writing a _solutions.txt file named after it.
Then, I worked on converting number words to numbers with a dictionary. I did the same for the operators
with special case for divided by. After, I handled "the result of" treating it like parentheses for the
following expression. Finally, I added code to work with all the other expressions appearing in the second file including the error handling.

**Did you face any tradeoffs while developing your solution? If so, what were they, and how did you select your course of action?**

Yes, I struggled with managing the expressions with multiple words (the result of and divided by).
It was easier when I did it just for plus, minus and times, but then considering the “divided by”
made me change a little bit things to delete the “by” and treat it as just one word. Also, I had
a hard time thinking about how to manage the comma, which I thought I could just ignore and delete
from each line at first but that could eventually cause a wrong result for certain order of operators,
so I used it as a way to separate statements. Again, I thought similar about “the result of” but then
I treated it as a parenthesis.

**What about your solution to the programming problem did you struggle with the most?**

I think it was trying to create a code that covered all the conditions at once instead of going step
by step from the easier parsing of numbers and plus/minus operators and then dealing with the more complex statements.

**What about your solution to the programming problem are you most proud of?**

I would say how to parse the words because I initially had a terrible idea of having an enormous
match/case block with all possible words including versions of each word with a comma after and
creating specific approaches for each one of them. Also, the fact that I had to use the error messages to obtain the indicated output.

**Don’t forget to tell me why you picked the “empty file” solution you picked.**

For this, I created cleanedlines variable that only stores nonempty lines. In this way, if after going
through all the lines of the file there is no content, the cleanedlines variable is going to be empty.
Then I use the condition “if not cleanedlines:” to easily determine if the file is empty or not. If it
is indeed empty, a “file empty” message is prompted. In this way even if the file had multiple “\n”
this would still be considered a blank file and I choose to print the message because it better
0communicates the user about what happened.

---

### Reflection 9

**Describe your overall process and steps for approaching this week’s programming problem.**

I think how late I was on this assignment played a big part in how I wrote it.
There may have been a better framework that handled things like variables and separate clauses from the start, leading to a more streamlined program. However, with the mindset of “it’s better to get some points than no points,” I thought it more apt to get something that sometimes works working, rather than being partway done with a more sophisticated implementation.
Everything completely falls apart if the wording is slightly different from the original. “let c equal three” is fine, but “let c be equal to three” will produce garbage. I’m also aware of the lack of order of operations.
As a result of just trying to add functionality on top of an existing flawed base, the code is very, very messy. There are a lot of redundant sections and processes that could be a lot simpler. I maybe could’ve cleaned it up a bit if I had more time (or was more responsible with my time.)

**Did you face any tradeoffs while developing your solution? If so, what were they, and how did you select your course of action?**

On a personal level, the main trade-off was time versus quality. The main cause of this was starting quite late. This perceived lack of time also led to the calculator being slow, long, and complicated.

**What about your solution to the programming problem did you struggle with the most?**

“The result of.” Whenever I thought I had something that worked, it would break on any instance of “the result of.” The simplest solution at the time was just checking if the words were there, and again, specifically worded the way they were in the example. It ended up needing a lot of if statements, and taking up a lot of space.

**What about your solution to the programming problem are you most proud of**

If I had to pick something, I’d say the recursion with subclauses. I was putting off dealing with commas, and instead just focused on the first clause. It didn’t end up being as much of an issue as I thought it would. I liked that it worked with long chains, more than was shown in the example problems.
In my mind, the commas kind of represent parentheses in a non-natural language equation, along with the phrase “the result of.”
I had hoped to include the latter in splitting up the expression into sections, but by the time I had started in earnest on the commas, I had already baked in searching for the specific phrase in several places, and I thought it would be more efficient to leave it as it was.

**Don’t forget to tell me why you picked the “empty file” solution you picked.**

I looked it up, and saw some fine folks on stackoverflow discussing the topic. Afterwards, I read again that I wasn’t able to import things, and importing os was needed for the method I found.
After re-reading the documentation though, I realized I could just read it. I think not making an exception would still have it return an empty file, but to be frank, I’m not sure what “hacking up a stack trace” entailed.
Also, I misunderstood the “running files in succession” thing. I thought it was about running the same file, so I had something that incremented the file names, which needed pathlib imported. At least, I think I misunderstood it.

---

### Reflection 10

**Overall process and steps for approaching this week’s programming problem.**

My thought first is to design the basic stuff. Beginning with writing a way to read the input file correctly. I read what the input file gonna look like and found there is going to be arguments in the command so I just use sys.argv to get the input.

Then I decided to work on reading the input. Since all inputs are in text like one/two etc, My program needs to convert these text into numbers. I initial thought is to use if/else but this is not effcient as there are a lot of numbers. So I use a dictionary to convert text into numbers.

Now I decided to find a way to access the input line by line, as the example is giving formulas line by line. So I use a for loop to read input. Then to convert the text in line into numbers/operators, I set up some dictionaries for the conversion. Also the syllabus has not specify what is the range of the numbers will be input though the example is just using 1-20. So I set up just a dictionary for 1-20. I also set up a dictionary for the operators.

I first tried to let my code deal with the easy input, tha means input with only numbers and operators.

During the process of writing the line processing function, I found using the dictionary to convert operators is not working well as it would make the line convertion more complicated as the type of the operators would be str while numbers are int. So I decided to use if/else to convert operators.

To deal with the easy input I deisgn my code to be 5 parts. Main for I/O, then it call a function A to process each line. Function A call the function B to convert each word in the line into numbers/operators. Then B call the function C to convert the numbers into a single value. Then A calculate the result of the line. Then A call the function D to convert the result into text. Finally A return the result to main and main print it out.

Now I have a solution for the easier input. I then needed to deal with the harder ones, which is the input with "the result of" in it.

When I tried to work on the "the result of" inputs, I found that the code has too many variables, so I change the format so that each case can be in a function.

**Did you face any tradeoffs while developing your solution? If so, what were they, and how did you select your course of action?**

One tradeoff I faced is to use dictionary or if/else to convert the operators. I first used dictionary but then I found it is not working well as it would make the line convertion more complicated as the type of the operators would be str while numbers are int. So I decided to use if/else to convert operators.

**What about your solution to the programming problem did you struggle with the most?**

I would say the most challenging part is to design the overall structure of the program. I had to think about how to read the input, how to convert the text into numbers/operators, and how to process each line. Also I had to design functions for each part of the program and make them work with each other. It took me a while to figure out the best way to organize my code and distrubute the tasks to functions.

Another thing that is quite complex is to deal with the harder inputs. I initially used a way to processed the lines and it works well with the eas inputs, but when I tried to process the harder inputs, I found that my code was not able to handle them correctly. So I had to come up with a new way to process the lines and make sure it works for all inputs. The re-designing of this part challenged me as I had to re-think about the logic of my code.

**What about your solution to the programming problem are you most proud of?**

I am most proud of the way I designed the overall structure of my program. I think it is well-organized and easy to understand. Each function has a clear purpose and they work together to achieve the final result. I think my code is efficient and can handle a variety of inputs. Also I think the part where I deal with the "the resulf of" inputs are also quite good as the solution I developed is able to handle them correctyly no matter where they are in the input, and can convert them into the easy case and let the easy case functions to process them.

**Why I picked the “empty file” solution I picked?**

I haven't worked on this part.
