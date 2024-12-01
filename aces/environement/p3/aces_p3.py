import random
from typing import List, Dict
from aces.llm_client import LLMClient
from dataclasses import dataclass, field
import json
from aces.environement.p3.p3_genotype import P3
from aces.environement.p3.prompt_function import get_prompt_label_p3, get_prompt_description_p3, prompt_solve_puzzle_given_f
from aces.environement.p3.skill_list import skill_list
from aces.environement.p3.utils import extract_skill, extract_solution, extract_f
from aces.code_sandbox import evaluate, pass_at_k
import numpy as np
import os
os.environ["TOKENIZERS_PARALLELISM"] = "true"
#TODO inherite from base ACES class with common stuff
class ACES_p3:
    def __init__(self, AcesArguments: dataclass, LLMArguments : dataclass):
        # initialize LLM client
        self.llm_args = LLMArguments
        self.skill_list = skill_list

        self.init_llm()
        # initialize environment
        self.aces_args = AcesArguments 
        self.initialize_environment()
        self.archive = []
        self.semantic_descriptors = []

    def init_llm(self,) -> None:
        """init LLM client"""
        print("init LLM client")
        cfg_generation ={"model": self.llm_args.model_name_or_path, "temperature": self.llm_args.temperature,  "max_tokens": self.llm_args.max_tokens}

        self.llm = LLMClient(model = self.llm_args.model_name_or_path, 
                             cfg_generation = cfg_generation,
                             base_url = self.llm_args.base_url, 
                             api_key = self.llm_args.api_key, 
                             online = self.llm_args.online, 
                             gpu = self.llm_args.gpu,
                             max_model_length = self.llm_args.max_model_length)
        print("LLM client initialized")
    
    def initialize_environment(self) -> None:
        with open(self.aces_args.path_archive, 'r') as f:
            self.archives = json.load(f)
        list_p3 = []

        # generate semantic descriptor
        for p in self.archives:
            list_p3.append(P3(program_str = p['program_str']))
        list_p3 = self.generate_semantic_descriptors(list_p3)
        
        # generate dfficulty
        ## generate multiple solutions
        list_p3 = self.generate_multiple_solutions(list_p3)
        ## evaluate python code
        list_p3 = self.evaluate_python_code(list_p3)
        ## generate description
        list_p3 = self.generate_description(list_p3)
        self.archives = list_p3

    def formating_chat_prompt(self, list_prompt_str: list[str]) -> list[list[dict]]:
        """Format list of prompt string to chat prompt"""
        list_prompt_chat=[]
        for prompt in list_prompt_str:
            # check whether I used syst prompt or not
            list_prompt_chat.append([{"role": "user", "content": prompt}])
        return list_prompt_chat
    
    def generate_multiple_solutions(self, puzzles: list[P3]) -> List[P3]:
        """Use LLM to generate multiple solutions for a list of puzzle"""
        list_prompt_sol = []
        for p in puzzles:
            list_prompt_sol.append(prompt_solve_puzzle_given_f(p.program_str))
        list_solutions = self.llm.multiple_completion(self.formating_chat_prompt(list_prompt_sol),n = self.aces_args.num_solutions)
        assert len(list_solutions) == len(puzzles)
        for id_puzzle in range(len(puzzles)):
            problem = puzzles[id_puzzle].program_str 
            n_solutions = [self.process_solutions(solution=sol,problem=problem) for sol in list_solutions[id_puzzle].response]
            puzzles[id_puzzle].all_solution = n_solutions
        # don't forget to verify solution with python
        return puzzles
    

    def process_solutions(self, solution: str, problem: str) -> str: 
        """Process solution and return full puzzle (f+g)"""
        puzzle = extract_f(problem) + "\n" + extract_solution(solution)
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
                fitness = pass_at_k(n=number_solution, c=c, k=k)
                list_correct_solution = [all_solution[i] for i in range(len(all_solution)) if all_solution_correct[i]]
                id_rd = random.randint(0,len(list_correct_solution)-1)
                puzzles[task_id].program_str = list_correct_solution[id_rd]
            puzzles[task_id].fitness = fitness

        return puzzles
    

    def generate_semantic_descriptors(self, puzzles: list[P3]) -> list[P3]:
        # Use LLM to evaluate puzzle along N programming skill dimensions
        # get prompt
        list_prompt = []
        for p in puzzles:
            list_prompt.append(get_prompt_label_p3(p.program_str, self.skill_list))
        list_prompt_chat = self.formating_chat_prompt(list_prompt)
        list_skills = self.llm.multiple_completion(list_prompt_chat)
        assert len(list_skills) == len(puzzles)
        for i in range(len(puzzles)):
            skill, explanation_skill = extract_skill(list_skills[i].response[0])
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
    
    def explore(self, num_iterations: int):
        for _ in range(num_iterations):
            # Generate novel target in semantic space
            target_descriptors = self.generate_novel_target()
            
            # Generate puzzle matching target
            candidate_puzzle = self.generate_puzzle(target_descriptors)
            
            # Verify feasibility
            if self.evaluate_feasibility(candidate_puzzle):
                actual_descriptors = self.generate_semantic_descriptors(candidate_puzzle)
                self.generated_puzzles.append({
                    'puzzle': candidate_puzzle,
                    'descriptors': actual_descriptors
                })

    
    def generate_novel_target(self) -> List[float]:
        # Generate target that maximizes diversity from existing puzzles
        #TODO: reproduce aces targeted
        if not self.generated_puzzles:
            return [random.random() for _ in range(self.num_dimensions)]
            
        # Find underexplored regions in semantic space
        existing_descriptors = [p['descriptors'] for p in self.generated_puzzles]
        target = self.find_diverse_target(existing_descriptors)
        return target
    

if __name__ == '__main__':
    from dataclasses import dataclass, field
    from typing import Optional

    @dataclass
    class AcesArguments:
        """
        Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.DataTrainingArguments
        """

        environement_name : str = field( default = "p3", metadata={"help": "environment name"})
        path_archive : str = field( default = "/home/flowers/work/aces/aces/environement/p3/preprocess_p3_emb_dedup_puzzles.json", metadata={"help": "path to the archive"})
        num_solutions: int = field( default = 2, metadata={"help": "number of solutions to generate to compute the difficulty score"})
        
    @dataclass
    class QdArguments:
        """
        Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.DataTrainingArguments
        """

        a: str = field(
            default="/home/flowers/work/hf/Qwen2.5-Coder-3B-Instruct",
            metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
        )

    @dataclass
    class LLMArguments:
        """
        Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.DataTrainingArguments
        """

        model_name_or_path: str = field(
            default="/home/flowers/work/hf/Qwen2.5-0.5B-Instruct",
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
        gpu: Optional[bool] = field(
            default = 1,
            metadata={
                "help": "number of gpus to use (vllm)"
            },
        )
        cfg_generation : Optional[bool] = field(
            default = False,
            metadata={
                "help": "use cfg generation"
            },
        ),
        temperature: Optional[float] = field(
            default = 1.0,
            metadata={
                "help": "temperature"
            },
        )
        max_tokens: Optional[int] = field(
            default = 4000,
            metadata={
                "help": "max tokens"
            },
        )
        max_model_length: Optional[int] = field(
            default = 20000,
            metadata={
                "help": "max context size"
            },
        )

    # parser = HfArgumentParser((AcesArguments,QdArguments,LLMArguments))
    # model_args, data_args, training_args = parser.parse_args_into_dataclasses()#["--output_dir", "/home/flowers/work/hf/trained/"])
    aces_args, qd_args, llm_args = AcesArguments(), QdArguments(), LLMArguments()
    aces= ACES_p3(aces_args, llm_args)