import ast
import json

def find_first_argument_of_first_function(code):
    parsed_code=ast.parse(code)
    for node in ast.walk(parsed_code):
        if isinstance(node, ast.FunctionDef) and node.name == 'f':
            first_arg = node.args.args[0].arg  # Get the first argument
            # print(f"The first argument of the function '{node.name}' is: {first_arg}")
            return first_arg

def extract_arguments_except_first_specific(func_code, function_name='f'):
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


def extract_skill(out):
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
    return skill, explanation_skill
