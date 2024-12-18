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
            for p in list_codes:
                list_code_formated.append(P3(program_str = p['program_str'], idx_generation = self.idx_generation))
            list_codes = self.generate_semantic_descriptors(list_code_formated)
            
            # generate dfficulty
            ## generate multiple solutions
            list_codes = self.generate_multiple_solutions(list_codes)
            ## evaluate python code
            list_codes = self.evaluate_python_code(list_codes)
            #check how many problem are solved
            codes_valid = sum([np.isfinite(codes.fitness) for codes in list_codes])
            print("valid codes in initial archive: ", codes_valid, " / ", len(list_codes))

            ## generate description
            list_codes = self.generate_description(list_codes)
            # rm_fitness_condition = True because initial puzzles should be solvable
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
            self.update_archive(list_codes)
            
    def update_archive(self,list_p3: list[P3], rm_fitness_condition = False):
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
                if self.aces_args.path_checkpoint_archive == "":
                    p.unique_id = self.unique_id
                self.archive.append(p)
                self.fitnesses.append(p.fitness)
                if not niche_idx in self.niche_to_idx_archive:
                    self.niche_to_idx_archive[niche_idx] = []
                self.niche_to_idx_archive[niche_idx].append(p.unique_id)
                self.unique_id +=1
    
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
        str_to_add=str(
                    f"\ndef run_eval():\n"
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

if __name__ == '__main__':
    from dataclasses import dataclass, field
    from typing import Optional


    @dataclass
    class AcesArguments:
        """
        Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.DataTrainingArguments
        """
        environement_name : str = field( default = "p3", metadata={"help": "environment name"})
        path_archive : str = field(
            default = "", 
            metadata={"help": "path to the archive if empty load the default archive"}
        )
        path_save: str = field( 
            default = "",
            metadata={"help": "path to save the archive"}
        )
        name_experience: str = field( 
            default = "aces_P3_expe",
            metadata={"help": "name of the experience (use for saving)"}
        )
        n_generation: int = field( 
            default = 100,
            metadata={"help": "number of generation to run"}
        )
        num_solutions: int = field(
            default = 50, metadata={"help": "number of solutions to generate to compute the difficulty score"}
        )
        batch_size: int = field( 
            default = 32, 
            metadata={"help": "number of query to send to the LLM to create new puzzles (multiple this number by 5 to get the number of generated puzzles as 5 puzzles are generated per query)"})
        n_fewshot_examples: int = field( default = 3, metadata={"help": "number of example in context" })
        max_descriptor_targeted: int = field(
            default = 5,
            metadata={"help": "number of max descriptor to target (at most `max_descriptor_targeted` semantic descriptor sample as goal)"})
        mode_sampling_goal: str = field(
            default = "uniform",
            metadata={"help": "['uniform','smart','none'], uniform sample goal uniformely, smart: sample unexplored goal close that are within 1 of distance of already explored goal in the semantic space"})
        seed: int = field(default=0)
        sampling_strategy_examples_from_niche: str = field(
            default='soft_normalised',
            metadata={"help": "sampling strategy to sample examples from a niche, choice: 'uniform','prob_best_5','soft_normalised'; need to explain difference"}
        )
        temperature_sampling_strategy_examples_from_niche: float = field(
            default= 0.2, 
            metadata={"help": "temperature softmax to sample example given their fitness given a niche"}
        )
        puzzle_generation_strategy: str = field(
        default= "aces_elm", 
        metadata={"help":"startegy to generate new puzzle, choice: ['aces','aces_elm'] todo 'wizard_coder'"})
        difficulty_min_target: int = field(default = 90, metadata={"help":"difficulty min to target /100"})
        difficulty_max_target: int = field(default = 100, metadata={"help":"difficulty min to target /100"})
        save_every_n_generations: int = field(default = 3, metadata={"help":"save archive every n generations"})
        path_checkpoint_archive: str = field(
            default="",
            metadata={"help":"if != '' resume experiment from the given a archive checkpoint "})
        

    @dataclass
    class LLMArguments:
        """
        Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.DataTrainingArguments
        """

        model_name_or_path: str = field(
            default="/home/flowers/work/hf/Qwen2.5-0.5B-Instruct",#"/home/flowers/work/hf/Qwen2.5-Coder-3B-Instruct",
            metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
        )
        online: Optional[bool] = field(
            default = False,
            metadata={
                "help": "use vllm server if True else use offline vllm"
            },
        )
        base_url: Optional[str] = field(
            default="http://localhost:8000",
            metadata={
                "help": "base url for vllm server"
            },
        )
        api_key: Optional[str] = field(
            default="",
            metadata={
                "help": "api key "
            },
        )
        gpu: Optional[int] = field(
            default = 1,
            metadata={
                "help": "number of gpus to use (vllm)"
            },
        )
        temperature: Optional[float] = field(
            default = 1.0,
            metadata={
                "help": "temperature"
            },
        )
        temperature_labeller: Optional[float] = field(
            default = 0.,
            metadata={
                "help": "temperature labeller (semantic descriptor)"
            },
        )
        min_p: Optional[float] = field(
            default = 0.05,
            metadata={
                "help": "min_p"
            },
        )
        max_tokens: Optional[int] = field(
            default = 4000,
            metadata={
                "help": "max tokens"
            },
        )
        max_model_length: Optional[int] = field(
            default = 25000,
            metadata={
                "help": "max context size"
            },
        )
        swap_space: Optional[float] = field(
            default=5,
            metadata={
                "help": "swap space (RAM memory for cache)"
            }
        )


    # parser = HfArgumentParser((AcesArguments,QdArguments,LLMArguments))
    # model_args, data_args, training_args = parser.parse_args_into_dataclasses()#["--output_dir", "/home/flowers/work/hf/trained/"])
    aces_args, llm_args = AcesArguments(), LLMArguments()
    aces= ACES_p3(aces_args, llm_args)
    aces.run()