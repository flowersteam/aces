import ast
import json

def find_first_argument_of_first_function(code) -> str:
    parsed_code=ast.parse(code)
    for node in ast.walk(parsed_code):
        if isinstance(node, ast.FunctionDef) and node.name == 'f':
            first_arg = node.args.args[0].arg  # Get the first argument
            # print(f"The first argument of the function '{node.name}' is: {first_arg}")
            return first_arg
        
def extract_function_name(code) -> str:
    parsed_code = ast.parse(code)

    # Extract function names using list comprehension
    function_names = [node.name for node in parsed_code.body if isinstance(node, ast.FunctionDef)]
    return function_names

def rm_given_function(code, list_function_name):
        # Parse the solution code to an AST
    solution_ast = ast.parse(code)

    # Filter out functions from the code that are in the list_function_name
    filtered_body = [
        node for node in solution_ast.body
        if not isinstance(node, ast.FunctionDef) or node.name not in list_function_name
    ]
    # Create a new module with the filtered body
    new_solution_ast = ast.Module(body=filtered_body, type_ignores=[])

    # Convert the AST back to source code
    new_solution_code = ast.unparse(new_solution_ast)
    return new_solution_code

def extract_f(code):
    return code.split("def g")[0].strip()

def extract_arguments_except_first_specific(func_code, function_name='f') -> str:
    # Parse the source code into an AST
    tree = ast.parse(func_code)
    
    # Initialize the result string
    result = []
    
    # Visit each node in the AST
    for node in ast.walk(tree):
        # Check if the node is a function definition and matches the specified function name
        if isinstance(node, ast.FunctionDef) and node.name == function_name:
            # Get the arguments from the function definition
            args = node.args
            
            # Exclude the first positional argument
            pos_args = args.args[1:]  # Skip the first argument
            
            # Handle positional arguments with defaults
            defaults = args.defaults
            num_defaults = len(defaults)
            num_pos_args = len(pos_args)
            default_start_index = num_pos_args - num_defaults

            # Handle non-default arguments
            for i, arg in enumerate(pos_args):
                if i >= default_start_index:
                    # If the argument has a default value, include it
                    default_value = defaults[i - default_start_index]
                    result.append(f"{ast.unparse(arg)}={ast.unparse(default_value)}")
                else:
                    # If no default, just add the argument
                    result.append(ast.unparse(arg))
            
            # Include *args and **kwargs
            if args.vararg:
                result.append(ast.unparse(args.vararg))
            if args.kwarg:
                result.append(ast.unparse(args.kwarg))

            # Handle keyword-only arguments with defaults
            for kw, kw_default in zip(args.kwonlyargs, args.kw_defaults):
                if kw_default is None:
                    result.append(ast.unparse(kw))
                else:
                    result.append(f"{ast.unparse(kw)}={ast.unparse(kw_default)}")
            break  # Stop if the target function is found
    
    return ', '.join(result)

def extract_solution(out) -> str:
    """exctract solution"""
    try:
        solution = out.replace("```python","```").replace("```\n","```")
        if "```" in solution:
            solution = solution.split("```")[1]

        else:
            solution = solution
    except:
        solution = ""
    return solution

def extract_skill(out,n_skills) -> list[list[int],str]:
    """exctract list of sem"""
    split_sentence="The list of skill use is:".lower()
    explanation_skill=out
    if split_sentence in out.lower():
        try:
            out=out.split("use is:")[1]
        except:
            pass
    try:
        out = out.split("[")[1].split("]")[0]
        out = "["+out+"]"
        skill = json.loads(out)
    except:
        skill=[]
        pass
    isallint = all(isinstance(i, int) for i in skill)
    if not isallint:
        skill = []

    skill =[1 if i in skill else 0 for i in range(n_skills)]

    return skill, explanation_skill
