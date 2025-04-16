def print_trainable_parameters(model):
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_params}")
    print(f"All parameters: {all_params}")
    print(f"Percentage of trainable parameters: {100 * trainable_params / all_params:.2f}%")


###### QA Format Utils ######

import re

def format_reward_func_qa(completions, **kwargs):
    pattern = r"\n#### The final answer is \d+"    
    completion_contents = [completion for completion in completions]    
    matches = [re.search(pattern, content) for content in completion_contents]
    return [0.5 if match else 0.0 for match in matches]

def correctness_reward_func_qa(completions, final_answer, **kwargs):
    rewards = []
    
    for completion, ground_truth in zip(completions, final_answer) :
        try:
            match = re.search(r'####.*?([\d,]+(?:\.\d+)?)', completion)
            if match:
                answer = match.group(1)
                
                for remove_char in [',', '$', '%', 'g']:
                    answer = answer.replace(remove_char, '')
                    
                if abs(float(answer)-float(ground_truth)) < 1e-3:
                    rewards.append(1.0)
                else:
                    rewards.append(0.0)
                
            else:
                rewards.append(0.0)
        except ValueError:
            rewards.append(0.0)
            
    return rewards

###### CODE Format Utils ######

import math
import datetime
import sys
from io import StringIO
import signal
import logging
import ast
from contextlib import contextmanager

class TimeoutException(Exception):
    pass

@contextmanager
def timeout(seconds):
    def signal_handler(signum, frame):
        raise TimeoutException("Code execution timed out")
    
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)

    try:
        yield
    finally:
        signal.alarm(0)

class LastAssignmentFinder(ast.NodeVisitor):
    def __init__(self):
        self.last_assigned_var = None
        self.has_return = False
    
    def visit_Assign(self, node):
        # Track only simple variable assignments
        if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            self.last_assigned_var = node.targets[0].id
        self.generic_visit(node)
    
    def visit_Return(self, node):
        self.has_return = True
        self.generic_visit(node)

def filter_code_to_function_only(code, func_name="solution"):
    """
    Filters the code to include only the function body by using indentation levels.
    """
    lines = code.splitlines()
    function_found = True
    filtered_lines = []
    function_indent = None
    function_indent = len(lines[0]) - len(lines[0].lstrip())
    
    for line in lines:
        stripped_line = line.strip()

        if function_found:
            current_indent = len(line) - len(line.lstrip())
            if function_indent is not None and current_indent < function_indent and stripped_line:
                # Stop processing lines when encountering less-indented code after function starts
                break
            filtered_lines.append(line)
    
    return "\n".join(filtered_lines)

def analyze_solution_function(code):
    try:
        # Parse the code into an AST
        tree = ast.parse(code)
        
        # Find the solution function definition
        solution_func = None
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == 'solution':
                solution_func = node
                break
        
        if solution_func is None:
            return None, False
        
        # Analyze the function body
        finder = LastAssignmentFinder()
        finder.visit(solution_func)
        
        return finder.last_assigned_var, finder.has_return
    except:
        return None, False

def extract_final_answer_from_code(code):
    """
    Executes the generated code and captures either the return value or the last assigned variable.
    """
    try:
        old_stdout = sys.stdout
        
        # Filter the code to keep only the solution function
        code = filter_code_to_function_only(code)
        
        # Remove all print statements from the code
        code_lines = code.splitlines()
        code_lines_no_print = [line for line in code_lines if 'print(' not in line and 'input(' not in line]
        code = '\n'.join(code_lines_no_print)
        
        # Prepare the code for execution
        import_statements = "import math\nimport datetime\n"
        def_function = "def solution():\n"
        # Indent the user code to fit inside the function
        indented_code = '\n'.join(line for line in code.splitlines() if "input(" not in line)
        
        # Analyze the function
        last_var, has_return = analyze_solution_function(def_function + indented_code)
        
        if has_return:
            # Use original execution approach
            code_to_exec = import_statements + def_function + indented_code + "\nresult = solution()"
        elif last_var:
            # Modify function to return the last assigned variable
            code_to_exec = (
                import_statements + 
                def_function + 
                indented_code + 
                f"\n    return {last_var}\n" +
                "result = solution()"
            )
        else:
            # No return and no assignments found
            return None

        # Redirect stdout to capture print output
        sys.stdout = mystdout = StringIO()
        
        # Use default built-ins and include necessary modules
        exec_globals = {
            '__builtins__': __builtins__,
            'math': math,
            'datetime': datetime,
        }
        
        # Execute the code with timeout
        with timeout(1):
            exec(code_to_exec, exec_globals)
            
        # Get the result
        result = exec_globals.get('result', None)
        
        # Return the result, ignoring any printed output since we're focusing on returns/assignments
        if result is not None:
            return str(result)
        return None 
    except TimeoutException:
        logging.warning("Code execution timed out after 1 second")
        return None
    except Exception as e:
        logging.warning(f"Execution error: {e}")
        return None
    finally:
        # Restore stdout
        sys.stdout = old_stdout


def correctness_reward_func_code(completions, final_answer, **kwargs):
    rewards = []
    
    for completion, ground_truth in zip(completions, final_answer) :
        try:
            answer = extract_final_answer_from_code(code=completion)
            
            if answer is None:
                rewards.append(0.0)
            elif abs(float(answer)-float(ground_truth)) < 1e-3:
                rewards.append(1.0)
            else:
                rewards.append(0.0)
        except ValueError:
            rewards.append(0.0)
            
    return rewards


def format_reward_func_code(completions, **kwargs):
    pattern = r"\n    return\s+(.+)\n"
    completion_contents = [completion for completion in completions]    
    matches = [re.search(pattern, content) for content in completion_contents]
    return [0.5 if match else 0.0 for match in matches]
