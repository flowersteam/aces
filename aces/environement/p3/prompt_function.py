from typing import Optional, Union, List
import json

import numpy as np
import textwrap
import copy

from pydantic import BaseModel,Field
from aces.environement.p3.utils import find_first_argument_of_first_function, extract_arguments_except_first_specific
from aces.environement.p3.skill_list import skill_list
from aces.environement.p3.prompt import (
prompt_aces, prompt_aces_elm,
prompt_gen_description, prompt_skills_labeling,
base_persona_code, instruction_solve_puzzle
)
from aces.environement.p3.p3_genotype import P3


# add Set Operations and Hashing




# class for instructor skill labelling prompt for P3
def get_class_PuzzleCheck(mode):
    match mode:
        case "description":
            class PuzzleCheck(BaseModel):
                """Puzzle description and if it should be given to the student or not."""
                puzzle_description: str = Field(description="Provide a brief, one to two sentence summary of the puzzle's content.")

        case "description+is_valid":
            class PuzzleCheck(BaseModel):
                """Puzzle description and if it should be given to the student or not."""
                puzzle_description: str = Field(description="Provide a brief, one to two sentence summary of the puzzle's content.")
                explanations: str = Field(decription="Short explanation of whether the puzzle should be given to the student or not.")
                give_puzzle_to_student: bool = Field(description="Whether the puzzle should be given to student or not based on the previous explanations")
    return PuzzleCheck

class Topics_evaluation(BaseModel):
    """List of topics that are used in the problem and solution."""
    explanations_index_topics: str = Field(decription="Short explanation of the specific topics employed in the puzzle.")
    index_topics: List[int] = Field(description="list of at most 5 index correponding to topics that are actually used in the problem `f` or the solution `g`")






# def create_prompt_label(puzzle : str, mode="give_skills"):
#     """
#     create prompt for label_puzzle goes with Topics_evaluation class with give_skills=True
#     mode = "give_skills", "is_valid", "description", "description+is_valid", "general"
#     is_valid -> filtering 
#     description use to give a description of the puzzle
#     """

#     format_skills=""
#     for idx,skill in enumerate(skill_list):
#         format_skills+=f"{idx}. {skill}\n"
#     skills = f"\n{format_skills}"
    
#     base_persona = base_persona_code#.format(level=level)
#     match mode:
#         case "is_valid": # WIP should also use a persona to label the puzzle
#             prompt=base_persona
#             prompt += "Your role is to check if the following puzzle could be used or not."
#             prompt += "\n\nThe puzzle is:\n```python\n" + puzzle + "\n```\n"
#         case "description": # WIP 
#             arg=find_first_argument_of_first_function(puzzle)
#             puzzle=puzzle.split('def g')[0].strip() + "\n\ndef g(...):\n\nassert f(g()) == True"
#             prompt=prompt_gen_description.format(arg_sol=arg,arg_solb=arg,puzzle=puzzle)
#             prompt += "\n\nThe puzzle is:\n```python\n" + puzzle + "\n```\n"
#         case "give_skills":
#             prompt = base_persona+"\n"
#             prompt+= "The Professor want to evaluate the diversity of those puzzles, can you label the following puzzle given the following list of topics, please?"
#             # prompt = "Your role is: given the following puzzle, and the list of topics, exctract the information requested."
#             prompt += "\nThe list of topics is:\n"+ skills 
#             prompt += "\n\nThe puzzle is:\n```python\n" + puzzle + "\n```\n"
#         case "give_skills_no_instructor": # WIP 
#             prompt = base_persona+"\n"
#             prompt+= "The Professor want to evaluate the diversity of those puzzles, can you label the following puzzle given the following list of topics, please?"
#             # prompt = "Your role is: given the following puzzle, and the list of topics, exctract the information requested."
#             prompt += "\nThe list of topics is:\n"+ skills 
#             prompt += "\n\nThe puzzle is:\n```python\n" + puzzle + "\n```\n"            
#             prompt += "Respond with two or three sentence explaning the topics used in the puzzle.\n"
#             prompt += "Then summarize your response by giving a list from 1 to 5 index corresponding to topics that are actually used in the puzzle above in this format: 'The list of skill use is: [].' where [] is the list of index of the topics used in the puzzle for example [3,5,6]."
#         case "general":
#             prompt= "Given the following puzzle, exctract the information requested."
#             prompt += "\n\nThe puzzle is:\n```python\n" + puzzle + "\n```\n"
#     return prompt


def get_prompt_label_p3(puzzle,skill_list=skill_list):
    format_skills=""
    for idx,skill in enumerate(skill_list):
        format_skills+=f"{idx}. {skill}\n"
    skills = f"\n{format_skills}"
    prompt_skills_labeling_formated = prompt_skills_labeling.format(skills=skills,puzzle=puzzle)
    return prompt_skills_labeling_formated

def get_prompt_description_p3(puzzle):
    arg=find_first_argument_of_first_function(puzzle)
    #just want description of the problem
    puzzle=puzzle.split('def g')[0].strip() + "\n\ndef g(...):\n\nassert f(g()) == True"
    prompt = prompt_gen_description.format(arg_sol=arg,arg_solb=arg,puzzle=puzzle)
    prompt += "\n\nThe puzzle is:\n```python\n" + puzzle + "\n```\n"
    return prompt


def get_programming_puzzles_prompt(
        list_few_shot_example : List[P3],
        skill_targeted: Optional[List[int]]=None,
        puzzle_generation_strategy = "aces",
        difficulty_range=(90,100),
        n_problem_to_gen =5

    ):
    """
    should change that to list_few_shot_example from list to Phenotype type
    skill_targeted list of binary vector [(0/1)]^n_skills indicating if the skill is targeted or not
    """
    extra_prompt=""
    prompt = prompt_aces
    if puzzle_generation_strategy == "aces":
        prompt = prompt_aces
    elif puzzle_generation_strategy == "aces_elm":
        prompt = prompt_aces_elm

    # if wizard_coder:
    #     prompt = copy.deepcopy(prompt_wizard_coder)
    #     few_shot_example_gen_puzzle="base"
    #     extra_prompt += evolve_instructions()
    if not isinstance(list_few_shot_example, list):
        list_few_shot_example = [list_few_shot_example]
    if all(isinstance(x, str) for x in list_few_shot_example):
        raise NameError("should be phenotype not str") 
    
    examples = ""
    for i, puzzle in enumerate(list_few_shot_example):
        puzzle_description = puzzle.description 
        prompt_cot_fitness = ""
        skill_puzzle_i=""
        puzzle_fitness = puzzle.fitness
        if puzzle_fitness == -np.inf:
            puzzle_fitness = 0
        prompt_cot_fitness = f"\n\n- Difficulty score: {int((puzzle_fitness+1)*100)} out of 100"

        skill_puzzle_i="\n\n- This puzzle has the following skills:"
        idx_skill_targeted = [idx for idx, val in enumerate(puzzle.emb) if val]
        for idx in idx_skill_targeted:
            skill_puzzle_i += f"\n* {skill_list[idx]}"

        examples += f"\nPuzzle {i}:\nPuzzle description: {puzzle_description}{prompt_cot_fitness}{skill_puzzle_i}\n\n```python\n{puzzle.program_str.strip()}\n```\n"    

    skill_target=":"
    idx_skill_targeted = [idx for idx, val in enumerate(skill_targeted) if val]
    for idx in idx_skill_targeted:
        skill_target += f"\n- {skill_list[idx]}"

    extra_prompt += f"You should aim to generate puzzles with a Difficulty score between {difficulty_range[0]} and {difficulty_range[1]} out of 100."
    if puzzle_generation_strategy == "aces_elm":
        prompt = prompt.format(examples=examples,skill_target=skill_target,extra=extra_prompt,n_problem=n_problem_to_gen,idx_last_puzzle=len(list_few_shot_example)-1)
    else:
        prompt = prompt.format(examples=examples,skill_target=skill_target,extra=extra_prompt,n_problem=n_problem_to_gen)

    return prompt


def evolve_instructions() -> None:
    """wizard coder instruction from https://github.com/nickrosh/evol-teacher/blob/main/generate_evol.py"""
    methods = [
    'Add new constraints and requirements to the original problem, adding approximately 10 additional words.',
    'Replace a commonly used requirement in the programming task with a less common and more specific one.',
    'If the original problem can be solved with only a few logical steps, please add more reasoning steps.',
    'Provide a piece of erroneous code as a reference to increase misdirection.',
    'Propose higher time or space complexity requirements, but please refrain from doing so frequently.'
    ]
    chosen_method = np.random.choice(methods)
    prompt_extra = f"Generate 5 Python Programming Puzzles by increasing the difficulty of the given programming puzzles a bit.\n\nYou can increase the difficulty using, but not limited to, the following methods:\n{chosen_method}"
    return prompt_extra


def prompt_solve_puzzle_given_f(problem_str: str): 
    """
    prompt to solve a puzzle (generate g) given f
    """
    try:
        arg_sol = extract_arguments_except_first_specific(problem_str)
    except:
        arg_sol= "..."
    # arg_sol= "..."#get_inputs(problem)
    f = problem_str.split("def g")[0].strip()
    full_prompt=instruction_solve_puzzle.format(f=f,arg_g=arg_sol)
    

    return full_prompt





if __name__ == "__main__":
    from aces.environement.p3.p3_genotype import P3

    # example of prompt to generate new puzzles
    p3_1 = P3(program_str="def f(a,b,c): return False", emb=[1,0,1,0,0],fitness=0.5 )
    p3_2 = P3(program_str="puzzle test2", emb=[1,0,1,0,0],fitness=0.5 )
    list_p3 = [p3_1, p3_2]
    skill_targeted=[1,0,1,0,1]
    print("prompt generate new puzzles:")
    print(get_programming_puzzles_prompt(list_p3,skill_targeted,n_fewshot_ex=2))
    print("prompt description:")
    print(get_prompt_description_p3(p3_1))