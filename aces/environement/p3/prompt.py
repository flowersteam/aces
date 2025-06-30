base_persona_code ="""You are a helpful assistant to a Professor teaching a programming course in Python. 
The Professor want to give Pyhton programming puzzles to his Computer Science student to teach them Python.
A Python programming puzzle is defined by two functions, the puzzle f(…) and the solution g(…). f defines an algorithmic challenge, and g solves this challenge. g is a solution to f if and only if f(g()) == True."""

prompt_skills_labeling = base_persona_code + """
The Professor want to evaluate the diversity of those puzzles, can you label the following puzzle given the following list of topics, please?
The list of topics is:
{skills}

The puzzle is:
```python
{puzzle}
```
Respond with two or three sentence explaning the topics used in the puzzle.
Then summarize your response by giving a list from 1 to 5 index corresponding to topics that are actually used in the puzzle above in this format: 'The list of skill use is: [].' where [] is the list of index of the topics used in the puzzle for example [3,5,6].
"""


prompt_gen_description="""A Python programming puzzle is defined by two functions, the problem f(solution, arg1=value1, arg2=value2, ..) and the solution. f defines an algorithmic puzzle, and the solution solves this puzzle.
You should pay a particular attention that the puzzle is solved if and only if **f(solution) == True**.
Your role is to write a one or two sentence the description of the puzzle's goal (what the solution should be), remember that the solution that satisfy the goal must be given as the first argument of `f`.
You can start by: 'Find the solution: {arg_sol} (describe its type shortly) that should (here you should speak about the solution: {arg_solb} and how it should solve all the constraints of the puzzle with respect to others args (describe their types shortly)) ...'. 
For example:
'Given a string `str1`, find the length of the longest substring without repeating characters.'
'Given two sorted arrays `nums1` and `nums2` of size `m` and `n` respectively, return the median of the two sorted arrays.'


The puzzle is:
```python
{puzzle}
```
"""


prompt_aces= """Consider Python Programming Puzzles (P3). P3 consists of two functions: a problem function `f` and its corresponding solution `g`. The challenge lies in constructing a SAT problem `f` and a function `g` such that `f(g())` evaluates to `True`

## Main Rules:
- Each puzzle includes two functions: `def f(...)` and `def g(...)`.
- The first argument of `f` is always the output from `g()`.
- Ensure `f` and `g` have matching argument signatures (e.g., `def f(solution, arg1=value1, arg2=value2, ...)` and `def g(arg1=value1, arg2=value2, ...)`). You also need to set the value of argument of f (arg1,arg2,...) and g when you define them.
- Avoid using `f` inside `g`, and `g` inside `f`.
- Include any necessary imports so your code runs smoothly.
- Give a clear Puzzle description that must be brief and diverse compared to the other puzzles.
- Make sure the puzzle is self-contained within these two functions.
- Make sure that that each puzzle have just all required skills (see below)

## P3 Format:
Puzzle description: A two to four sentence summary of the puzzle's content. To explain what is the problem `f`, and how you can solve it with `g`. 
```python
def f(solution, args=...) -> bool:
    # Python code to test the solution returned by g.
    # This function is a test unit and must return True if the solution is correct, and False otherwise.

def g(args=...) -> solution:
    # Python code to generate a solution for the problem.
    # The solution should generalize to all possible args.
    return solution

assert f(g()) == True
```

## Examples:
{examples}

Generate {n_problem} P3 similar to previous Examples. Ensure that all new puzzles are more challenging than Puzzle from previous examples.
{extra}

**Please make sure that new puzzles have JUST ALL the following skills**{skill_target}
## New {n_problem} problems:
"""


prompt_aces_elm= """Consider Python Programming Puzzles (P3). P3 consists of two functions: a problem function `f` and its corresponding solution `g`. The challenge lies in constructing a SAT problem `f` and a function `g` such that `f(g())` evaluates to `True`

## Main Rules:
- Each puzzle includes two functions: `def f(...)` and `def g(...)`.
- The first argument of `f` is always the output from `g()`.
- Ensure `f` and `g` have matching argument signatures (e.g., `def f(solution, arg1=value1, arg2=value2, ...)` and `def g(arg1=value1, arg2=value2, ...)`). You also need to set the value of argument of f (arg1,arg2,...) and g when you define them.
- Avoid using `f` inside `g`, and `g` inside `f`.
- Include any necessary imports so your code runs smoothly.
- Give a clear Puzzle description that must be brief and diverse compared to the other puzzles.
- Make sure the puzzle is self-contained within these two functions.

## P3 Format:
Puzzle description: A two to four sentence summary of the puzzle's content. To explain what is the problem `f`, and how you can solve it with `g`. 
```python
def f(solution, args=...) -> bool:
    # Python code to test the solution returned by g.
    # This function is a test unit and must return True if the solution is correct, and False otherwise.

def g(args=...) -> solution:
    # Python code to generate a solution for the problem.
    # The solution should generalize to all possible args.
    return solution

assert f(g()) == True
```

## Examples:
{examples}

Generate {n_problem} P3 similar to the last Examples (Puzzle {idx_last_puzzle}). Ensure that all new puzzles are more challenging than Puzzle {idx_last_puzzle}.
{extra}

**Please make sure that new puzzles have JUST ALL the following skills**{skill_target}
## New {n_problem} problems inspired by Puzzle {idx_last_puzzle}:
"""

instruction_solve_puzzle = '''You will be given a function. Respond only in code with a correct, efficient implementation of the function. You will need to generate the correct solutions (g), for the Problem 2 that satisfies the condition f(g()) == True.

Problem 0:
```python
def f(stamps: List[int], target=80, max_stamps=4, options=[10, 32, 8]) -> bool:
    """Find a selection of at most max_stamps stamps whose total worth is the target value."""
    for s in stamps:
        assert s in options
    return len(stamps) <= max_stamps and sum(stamps) == target
```
Solution 0:
```python
def g(target = 80, max_stamps = 4, options = [10, 32, 8]):
    from itertools import combinations_with_replacement
    for n in range(max_stamps + 1):
        for c in combinations_with_replacement(options, n):
            if sum(c) == target:
                return list(c)

assert f(g()) == True
```

Problem 1:
```python
from typing import*
def f(ans: List[List[int]], target=2) -> bool:
    """
    Find a list of pairs of integers where the number of pairs in which the second number is more than
    two greater than the first number is a given constant
    """
    for i in range(len(ans)):
        a, b = ans[i]
        if b - a >= 2:
            target -= 1
    return target == 0
```

Solution 1:
```python
def g(target = 2):
    return [[0, 2]] * target 

assert f(g()) == True
```

Now you need to give the solution (def g({arg_g}):) to the following Problem 2 that satisfies the condition f(g()) == True.

Problem 2:
```python
{f}
```
'''


prompt_wizard_coder = """Consider Python Programming Puzzles (P3). P3 consists of two functions: a problem function `f` and its corresponding solution `g`. The challenge lies in constructing a SAT problem `f` and a function `g` such that `f(g())` evaluates to `True`

## Main Rules:
- Each puzzle includes two functions: `def f(...)` and `def g(...)`.
- The first argument of `f` is always the output from `g()`.
- Ensure `f` and `g` have matching argument signatures (e.g., `def f(arg0, arg1=value1, arg2=value2, ...)` and `def g(arg1=value1, arg2=value2, ...)`). You also need to set the value of argument of f (arg1,arg2,...) and g when you define them.
- Avoid using `f` inside `g`, and `g` inside `f`.
- Include any necessary imports so your code runs smoothly.
- Give a clear Puzzle description that must be brief and diverse compared to the other puzzles.
- Make sure the puzzle is self-contained within these two functions.

## P3 Format:
Puzzle description: A two to four sentence summary of the puzzle's content. To explain what is the problem `f`, and how you can solve it with `g`. 
```python
def f(solution, args=...) -> bool:
    # Python code to test the solution returned by g.
    # This function is a test unit and must return True if the solution is correct, and False otherwise.

def g(args=...) -> solution:
    # Python code to generate a solution for the problem.
    # The solution should generalize to all possible args.
    return solution

assert f(g()) == True
```

## Examples:
{examples}

Generate {n_problem} different P3 similar to previous Examples.
{extra}

## New {n_problem} problems:
"""