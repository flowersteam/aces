import random
from typing import List, Dict
from aces.llm_client import LLMClient
from dataclasses import dataclass, field
import json
from aces.environement.p3.p3_genotype import P3
from aces.environement.p3.prompt import get_prompt_label_p3, get_prompt_description_p3
from aces.environement.p3.skill_list import skill_list
from aces.environement.p3.utils import extract_skill
#TODO inherite from base ACES class with common stuff
class ACES_p3:
    def __init__(self, AcesArguments: dataclass, qd_args: dataclass, LLMArguments : dataclass):
        # initialize LLM client
        self.llm_args = LLMArguments
        self.init_llm()
        # initialize environment
        self.aces_args = AcesArguments 
        self.initialize_environment()
        self.archive = []
        self.semantic_descriptors = []
        self.skill_list = skill_list

    def init_llm(self,) -> None:
        """init LLM client"""
        print("init LLM client")
        cfg_generation ={"model": self.llm_args.model_name_or_path, "temperature": self.llm_args.temperature,  "max_tokens": self.llm_args.max_tokens}

        self.llm = LLMClient(model = self.llm_args.model_name_or_path, 
                             cfg_generation = cfg_generation,
                             base_url = self.llm_args.base_url, 
                             api_key = self.llm_args.api_key, 
                             online = self.llm_args.online, gpu = self.llm_args.gpu)
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
        

                
    def generate_semantic_descriptors(self, puzzles: list[P3]) -> list[P3]:
        # Use LLM to evaluate puzzle along N programming skill dimensions
        # get prompt
        list_prompt=[]
        for p in puzzles:
            list_prompt.append(get_prompt_label_p3(p,self.skill_list))
        list_skills = self.llm.multiple_completion(list_prompt)
        for i in range(len(puzzles)):
            skill, explanation_skill = extract_skill(list_skills[i])
            puzzles[i].emb = skill
            puzzles[i].explanation_emb = explanation_skill
            # puzzle[i].phenotype = skill
        return puzzles
    

    
    def generate_puzzle(self, target_descriptors: List[float]) -> str:
        # Use LLM to generate a puzzle matching target skill descriptors
        puzzle = self.llm.generate_puzzle(target_descriptors)
        return puzzle
        
    def evaluate_feasibility(self, puzzle: str) -> bool:
        # Use LLM to check if puzzle is solvable
        return self.llm.check_solvability(puzzle)
    
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
        if not self.generated_puzzles:
            return [random.random() for _ in range(self.num_dimensions)]
            
        # Find underexplored regions in semantic space
        existing_descriptors = [p['descriptors'] for p in self.generated_puzzles]
        target = self.find_diverse_target(existing_descriptors)
        return target