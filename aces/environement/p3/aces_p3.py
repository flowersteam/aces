import random
from typing import List, Dict
from aces.llm_client import LLMClient
from dataclasses import dataclass, field
import json
from aces.environement.p3.p3_genotype import P3
from aces.environement.p3.prompt_function import get_prompt_label_p3, get_prompt_description_p3, prompt_solve_puzzle_given_f, get_programming_puzzles_prompt
from aces.environement.p3.skill_list import skill_list
from aces.environement.p3.utils import extract_skill, extract_solution, extract_f, extract_function_name, rm_given_function
from aces.code_sandbox import evaluate, pass_at_k
from aces.aces import ACES_base
import numpy as np
import os
from itertools import combinations
from scipy.spatial.distance import cdist
import pickle

os.environ["TOKENIZERS_PARALLELISM"] = "true"
#TODO inherite from base ACES class with common stuff
class ACES_p3(ACES_base):
    def __init__(self, AcesArguments: dataclass, LLMArguments : dataclass, *args, **kwargs):
        super().__init__(AcesArguments=AcesArguments, LLMArguments=LLMArguments, *args, **kwargs)
    
    def init_skill_list(self):
        #TODO: automatically generate skill list given initial puzzles
        self.skill_list = skill_list

    def initialize_environment(self) -> None:
        if self.aces_args.path_checkpoint_archive == "":
            path_archive = self.aces_args.path_archive
            if self.aces_args.path_archive == "":
                # load default archive if empty
                current_dir = os.path.dirname(os.path.abspath(__file__))
                path_archive = os.path.join(current_dir, 'preprocess_p3_emb_dedup_puzzles.json')

            print("load initial archive: ", path_archive)
            if "json" in path_archive:
                with open(path_archive, 'r') as f:
                    list_codes = json.load(f)
            else:
                with open(path_archive, 'rb') as f:
                    list_codes = pickle.load(f)

            self.idx_generation = 0
            list_code_formated = []

            # generate semantic descriptor
            print("generate semantic descriptors (initial archive)")
            for p in list_codes:
                list_code_formated.append(P3(program_str = p['program_str'], idx_generation = self.idx_generation))
            list_codes = self.generate_semantic_descriptors(list_code_formated)
            
            # generate dfficulty
            ## generate multiple solutions
            print("generating multiple solutions (initial archive)")
            list_codes = self.generate_multiple_solutions(list_codes)
            ## evaluate python code
            print("evaluate solutions (initial archive)")
            list_codes = self.evaluate_python_code(list_codes)
            #check how many problem are solved
            codes_valid = sum([np.isfinite(codes.fitness) for codes in list_codes])
            print("valid codes in initial archive: ", codes_valid, " / ", len(list_codes))

            ## generate description
            print("generate description (initial archive)")
            list_codes = self.generate_description(list_codes)
            # rm_fitness_condition = True because initial puzzles should be solvable
            print("update archive with initial puzzles")
            self.update_archive(list_codes, rm_fitness_condition = True)
        else:
            print("resume experiment from the given a archive checkpoint: ", self.aces_args.path_checkpoint_archive)
            if "json" in self.aces_args.path_checkpoint_archive:
                with open(self.aces_args.path_checkpoint_archive, 'r') as f:
                    list_codes = json.load(f)
            else:
                with open(self.aces_args.path_checkpoint_archive, 'rb') as f:
                    list_codes = pickle.load(f)

            self.idx_generation = list_codes[-1].idx_generation
            self.unique_id = list_codes[-1].unique_id + 1
            self.update_archive(list_codes, update_id= False)
            
    def update_archive(self,list_p3: list[P3], rm_fitness_condition = False,update_id = True):
        """update archive with valid puzzles"""
        for p in list_p3:
            condition_add_individual = p.fitness != -np.inf
            if rm_fitness_condition:
                # remove fitness condition when initializing the archive
                condition_add_individual = True
                if p.fitness == -np.inf:
                    p.fitness = 0 #if it was unsolved give max fitness
            if condition_add_individual:
                niche_idx = tuple(p.emb)
                if update_id:
                    p.unique_id = self.unique_id
                    self.unique_id +=1
                self.archive.append(p)
                self.fitnesses.append(p.fitness)
                if not niche_idx in self.niche_to_idx_archive:
                    self.niche_to_idx_archive[niche_idx] = []
                self.niche_to_idx_archive[niche_idx].append(p.unique_id)
                
    
    def generate_multiple_solutions(self, puzzles: list[P3]) -> List[P3]:
        """Use LLM to generate multiple solutions for a list of puzzle"""
        list_prompt_sol = []
        for p in puzzles:
            list_prompt_sol.append(prompt_solve_puzzle_given_f(p.program_str))
        list_prompt_sol_chat = self.formating_chat_prompt(list_prompt_sol)
        list_solutions = self.llm.multiple_completion(list_prompt_sol_chat, n = self.aces_args.num_solutions)
        assert len(list_solutions) == len(puzzles)
        for id_puzzle in range(len(puzzles)):
            problem = puzzles[id_puzzle].program_str 
            n_solutions = [self.process_solutions(solution=sol,problem=problem) for sol in list_solutions[id_puzzle].response]
            puzzles[id_puzzle].all_solution = n_solutions
        # don't forget to verify solution with python
        return puzzles
    

    def process_solutions(self, solution: str, problem: str) -> str: 
        """Process solution and return full puzzle (f+g)"""
        just_problem = extract_f(problem) 
        solution = extract_solution(solution)
        try:
            name_functions_problem = extract_function_name(just_problem)
            solution = rm_given_function(solution, name_functions_problem)
        except:
            #solution is not ast parsable so it is incorrect
            pass
        puzzle = just_problem + "\n" + solution
        puzzle = puzzle.split("\nassert f")
        puzzle = puzzle[0] + "\nassert f(g()) == True\n"
        return puzzle
    
    def evaluate_python_code(self, puzzles: list[P3]) -> List[P3]:
        """Evaluate python code"""
        list_task_id = []
        list_task_id_unique = []
        list_codes_to_test = []
        str_to_add = (
            f"\nimport random\n"
            f"import numpy as np\n"
            f"random.seed(42)\n"
            f"np.random.seed(42)\n"
            f"def run_eval():\n"
            f"    return f(g()) == True"
        )
        for id_puz,p in enumerate(puzzles):
            list_task_id_unique.append(id_puz)
            for id_sol in range(len(p.all_solution)):
                list_task_id.append(id_puz)
                list_codes_to_test.append(p.all_solution[id_sol] + str_to_add)


        results = evaluate(list_codes_to_test, list_task_id, entry_point="run_eval")
        # dic_passk = results["pass@k"] # {task_id: pass@k} 
        raw_result = results["raw_result"] 
        for task_id in list_task_id_unique:
            all_solution = []
            all_solution_correct = []
            for id_completion in range(len(raw_result[task_id])):
                all_solution.append(raw_result[task_id][id_completion]["code"].split(str_to_add)[0])
                all_solution_correct.append(raw_result[task_id][id_completion]["correct"])
            
            puzzles[task_id].all_solution = all_solution
            puzzles[task_id].all_solution_correct = all_solution_correct

            number_solution = len(all_solution)
            c = sum(all_solution_correct)
            k=1 # estimation of pass@1
            
            if c==0:
                fitness = -np.inf
            else:
                #fitnes equal to - pass@1
                fitness = - pass_at_k(n=number_solution, c=c, k=k)
                list_correct_solution = [all_solution[i] for i in range(len(all_solution)) if all_solution_correct[i]]
                id_rd = random.randint(0,len(list_correct_solution)-1)
                puzzles[task_id].program_str = list_correct_solution[id_rd]
            puzzles[task_id].fitness = fitness

        return puzzles
    

    def generate_semantic_descriptors(self, puzzles: list[P3]) -> list[P3]:
        # Use LLM to evaluate puzzle along N programming skill dimensions
        #TODO in aces the temperature is equal to 0. (however for reasoning temperature is often set to be >0. so I should do write an option to let user choose) 
        # get prompt
        list_prompt = []
        for p in puzzles:
            list_prompt.append(get_prompt_label_p3(p.program_str, self.skill_list))
        list_prompt_chat = self.formating_chat_prompt(list_prompt)
        list_skills = self.llm.multiple_completion(list_prompt_chat,temperature = self.llm_args.temperature_labeller)
        assert len(list_skills) == len(puzzles)
        for i in range(len(puzzles)):
            skill, explanation_skill = extract_skill(list_skills[i].response[0],n_skills=len(self.skill_list))
            puzzles[i].emb = skill
            puzzles[i].explanation_emb = explanation_skill
            # puzzle[i].phenotype = skill
        return puzzles
    
    def generate_description(self, puzzles: list[P3]) -> list[P3]:
        # Use LLM to evaluate puzzle along N programming skill dimensions
        # get prompt
        list_prompt = []
        for p in puzzles:
            list_prompt.append(get_prompt_description_p3(p.program_str))
        list_description = self.llm.multiple_completion(self.formating_chat_prompt(list_prompt))
        for i in range(len(puzzles)):
            puzzles[i].description = list_description[i].response[0]
        return puzzles

    def generate_new_problems(self,list_goal_with_examples):
        list_prompt = []
        difficulty_range = (self.aces_args.difficulty_min_target,self.aces_args.difficulty_max_target)
        list_few_shot_ex_id = []
        list_goal = []
        for (list_few_shot_example_phenotypes, goal) in list_goal_with_examples:
            list_few_shot_ex_id.append([ex.unique_id for ex in list_few_shot_example_phenotypes])
            list_goal.append(goal)
            prompt = get_programming_puzzles_prompt(list_few_shot_example_phenotypes,goal,
                        puzzle_generation_strategy = self.aces_args.puzzle_generation_strategy,
                        difficulty_range = difficulty_range)
            
            list_prompt.append(prompt)

        list_prompt_chat = self.formating_chat_prompt(list_prompt)
        news_puzzles = self.llm.multiple_completion(list_prompt_chat)
        news_puzzles = [p.response[0] for p in news_puzzles]
        #TODO: exctract puzzles + ...
        list_new_p3 = []
        
        for id_puzzle,puzzle in enumerate(news_puzzles):
            split_puzzles = puzzle.replace("```python","```").replace("``` python","```").split("```")
            for idx in range(len(split_puzzles)):
                if "def f" in split_puzzles[idx] and "def g" in split_puzzles[idx]:
                    split_puzzles[idx] = split_puzzles[idx].split("\nassert f(")[0]
                    split_puzzles[idx] = split_puzzles[idx] + "\nassert f(g()) == True\n"
                    new_p3 = P3(split_puzzles[idx],target_skills=list_goal[id_puzzle],puzzles_id_fewshot=list_few_shot_ex_id[id_puzzle], idx_generation=self.idx_generation)
                    list_new_p3.append(new_p3)
        return list_new_p3

    def filter_problems(self, puzzles: list[P3]) -> List[P3]:
        """Filter problems based on if they are solved with trivial solution [], 0, None, '' """
        print("filtering problems")
        list_task_id = []
        list_task_id_unique = []
        list_codes_to_test = []
        str_to_add = (
            f"\nimport random\n"
            f"import numpy as np\n"
            f"random.seed(42)\n"
            f"np.random.seed(42)\n"
            f"def run_eval():\n"
            f"    cond = False\n"
            f"    list_test_sol = [[],None,'',[[]]]\n"
            f"    for test_sol in list_test_sol:\n"
            f"        try:\n"
            f"            cond = f(test_sol) == True\n"
            f"        except:\n"
            f"            cond = False\n"
            f"        if cond:\n"
            f"            return True\n"
            f"    return False\n"
        )
        for id_puz,p in enumerate(puzzles):
            list_task_id_unique.append(id_puz)
            # for id_sol in range(len(p.all_solution)):
            #     list_task_id.append(id_puz)
            #     list_codes_to_test.append(p.all_solution[id_sol] + str_to_add)

            list_task_id.append(id_puz)
            list_codes_to_test.append(p.program_str.split("\nassert f(")[0] + str_to_add)
        results = evaluate(list_codes_to_test, list_task_id, entry_point="run_eval",min_time_limit= 5*5, gt_time_limit_factor= 2*5)
        # dic_passk = results["pass@k"] # {task_id: pass@k} 
        raw_result = results["raw_result"] 
        list_rm_task_id = []
        for task_id in list_task_id_unique:
            all_solution = []
            all_solution_correct = []
            for id_completion in range(len(raw_result[task_id])):
                all_solution.append(raw_result[task_id][id_completion]["code"].split(str_to_add)[0])
                all_solution_correct.append(raw_result[task_id][id_completion]["correct"])
            
            # puzzles[task_id].all_solution = all_solution
            # puzzles[task_id].all_solution_correct = all_solution_correct

            number_solution = len(all_solution)
            c = sum(all_solution_correct)
            k=1 # estimation of pass@1
            
            # if c==0 keep the puzzle
            # else remove it
            if c!=0: 
                list_rm_task_id.append(task_id)
        print(f"removing {len(list_rm_task_id)} / {len(puzzles)} puzzle with trivial solution")
        if len(list_rm_task_id) > 0:
            print("removing puzzle with trivial solution:\n", puzzles[list_rm_task_id[0]].program_str)
        for id_sol_rm in list_rm_task_id[::-1]:
            # remove puzzle from the list
            del puzzles[id_sol_rm]
        return puzzles
    
if __name__ == '__main__':
    from dataclasses import dataclass, field
    from typing import Optional
    from aces.environement.p3.config_class import AcesArguments,LLMArguments


    # parser = HfArgumentParser((AcesArguments,QdArguments,LLMArguments))
    # model_args, data_args, training_args = parser.parse_args_into_dataclasses()#["--output_dir", "/home/flowers/work/hf/trained/"])
    aces_args, llm_args = AcesArguments(), LLMArguments()
    aces= ACES_p3(aces_args, llm_args)
    aces.run()