import random
from typing import List, Dict
from aces.llm_client import LLMClient
from dataclasses import dataclass, field
import json
class ACES:
    def __init__(self, AcesArguments: dataclass, qd_args: dataclass, LLMArguments : dataclass):
        # initialize LLM client
        self.llm_args = LLMArguments
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
                             online = self.llm_args.online, gpu = self.llm_args.gpu)
        print("LLM client initialized")
    
    def initialize_environment(self) -> None:
        with open(self.aces_args.path_archive, 'r') as f:
            self.archives = json.load(f)
        #TODO: label archive



    def generate_semantic_descriptors(self, puzzle: str) -> List[float]:
        # Use LLM to evaluate puzzle along 10 programming skill dimensions
        descriptors = self.llm.evaluate_skills(puzzle)
        return descriptors
    
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