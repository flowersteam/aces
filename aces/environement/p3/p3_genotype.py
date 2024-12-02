from typing import List, Optional
class P3:
    def __init__(self, program_str: str, emb: list= None,
                  idx_generation: int=-1,target_skills=None,fitness: int =None, quality: int =None,
                  description:str=" description of the puzzle",  is_valid:bool=None, puzzle_history: list = [],puzzles_id_fewshot:list[str]=[],
                  is_valid_explanation:str=None,result_obj: Optional[dict]={}, explanation_emb=None,
                  all_solution:List[str]=None, all_solution_correct:List[bool]=None,unique_id:str=None,  **kwargs) -> None:
        """
        Genotype for a programming puzzle problem+solution pair.
        Args:
            program_str: the code for the pair.
            result_obj: the result of the solution.
            config: environment config
            idx_generation: -1 -> from intialisation, ...
            puzzle_history: few shot example to generate this puzzle
            all_solution: if multiple were generated
            all_solution_correct: list of bool to check which one was correct
        """
        self.fitness=fitness
        self.program_str = program_str
        self.result_obj = result_obj
        self.emb = emb
        self.explanation_emb = explanation_emb
        self.idx_generation = idx_generation
        self.target_skills = target_skills
        self.puzzle_history = puzzle_history
        self.puzzles_id_fewshot = puzzles_id_fewshot
        self.quality = quality
        self.description=description
        self.is_valid = is_valid
        self.is_valid_explanation = is_valid_explanation
        self.all_solution = all_solution
        self.all_solution_correct = all_solution_correct
        self.id = unique_id
        self.phenotype = emb   

    def __str__(self) -> str:
        return self.program_str
    
