from typing import List
from aces.llm_client import LLMClient
from dataclasses import dataclass
import json
from aces.genotype import Genotype
import numpy as np
import os
from itertools import combinations
from scipy.spatial.distance import cdist
import pickle
from tqdm import trange


class ACES_base:
    def __init__(self, AcesArguments: dataclass, LLMArguments : dataclass):
        # initialize LLM client
        self.llm_args = LLMArguments
        self.llm_args.seed = AcesArguments.seed
        self.unique_id = 0
        self.idx_generation = 0
        self.archive = []
        self.fitnesses = []
        self.niche_to_idx_archive = {}
        self.semantic_descriptors = []
        self.init_skill_list()
        self.init_llm()
        # initialize environment
        self.aces_args = AcesArguments 
        self.initialize_environment()
        self.rng = np.random.default_rng(self.aces_args.seed)


    def niches_filled(self):
        """Get the number of niches that have been explored in the map."""
        return len(self.niche_to_idx_archive.keys())

    def max_fitness(self):
        """Get the maximum fitness value in the map."""
        return (np.array(self.fitnesses)[np.isfinite(self.fitnesses)]).max()

    def mean_fitness(self):
        """Get the mean fitness value in the map."""
        return (np.array(self.fitnesses)[np.isfinite(self.fitnesses)]).mean()

    def min_fitness(self):
        """Get the minimum fitness value in the map."""
        return (np.array(self.fitnesses)[np.isfinite(self.fitnesses)]).min()
    
    def init_skill_list(self):
        """initialize skill list"""
        raise NotImplementedError
    
    def init_llm(self,) -> None:
        """init LLM client"""
        print("init LLM client")
        cfg_generation = {"model": self.llm_args.model_name_or_path, "temperature": self.llm_args.temperature}
        if self.llm_args.max_tokens!= -1:
            cfg_generation["max_tokens"] = self.llm_args.max_tokens
        if self.llm_args.min_p!=0:
            if "extra_body" not in cfg_generation:
                cfg_generation["extra_body"] = {}

            cfg_generation["extra_body"]["min_p"] = self.llm_args.min_p
            cfg_generation["min_p"] = self.llm_args.min_p
        if self.llm_args.top_k!=-1:
            if "extra_body" not in cfg_generation:
                cfg_generation["extra_body"] = {}
            cfg_generation["extra_body"]["top_k"] = self.llm_args.top_k
        if self.llm_args.top_p!=1:
            cfg_generation["top_p"] = self.llm_args.top_p
        self.llm = LLMClient(model = self.llm_args.model_name_or_path, 
                             cfg_generation = cfg_generation,
                             base_url = self.llm_args.base_url, 
                             api_key = self.llm_args.api_key, 
                             online = self.llm_args.online, 
                             gpu = self.llm_args.gpu,
                             max_model_length = self.llm_args.max_model_length,
                             azure = self.llm_args.azure,
                             local_server = self.llm_args.local_server,
                             seed = self.llm_args.seed,
                             fp8 = self.llm_args.fp8,
                             gpu_memory = self.llm_args.gpu_memory,
                             sglang=self.llm_args.sglang,
                             log_level= self.llm_args.log_level,
                             enable_thinking = self.llm_args.enable_thinking,
                             ep_moe=self.llm_args.ep_moe,
                             kwargs_engine=self.llm_args.kwargs_engine
                            )
        print("LLM client initialized")


    def exctract_reasoning_response(self, response: str, think_stop_tag: str= "</think>") -> str:
        """Extract reasoning from the response"""
        reasoning = None
        sol = response
        if think_stop_tag in response:
            reasoning = response.split(think_stop_tag)[0].strip() + think_stop_tag
            sol = response.split(think_stop_tag)[1].strip()
        return reasoning, sol
    

    def initialize_environment(self) -> None:
        if self.aces_args.path_checkpoint_archive == "":
            print("load initial archive: ", self.aces_args.path_archive)
            if "json" in self.aces_args.path_archive:
                with open(self.aces_args.path_archive, 'r') as f:
                    list_codes = json.load(f)
            else:
                with open(self.aces_args.path_archive, 'rb') as f:
                    list_codes = pickle.load(f)

            self.idx_generation = 0
            list_code_formated = []

            # generate semantic descriptor
            print("generate semantic descriptors (initial archive)")
            for p in list_codes:
                list_code_formated.append(Genotype(program_str = p['program_str'], idx_generation = self.idx_generation))
            list_codes = self.generate_semantic_descriptors(list_code_formated)
            
            # generate dfficulty
            ## generate multiple solutions
            print("generating multiple solutions (initial archive)")
            list_codes = self.generate_multiple_solutions(list_codes)
            ## evaluate python code
            print("evaluate solutions (initial archive)")
            list_codes = self.evaluate_python_code(list_codes)
            ## generate description
            print("generate description (initial archive)")
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
    
    def terminate(self):
        """Terminate the environment and the LLM client"""
        self.llm.terminate()

    def update_archive(self, inital_codes: list[Genotype], rm_fitness_condition = False):
        """update archive with valid puzzles"""
        for code in inital_codes:
            condition_add_individual = code.fitness != -np.inf
            if rm_fitness_condition:
                # remove fitness condition when initializing the archive
                condition_add_individual = True
                if code.fitness == -np.inf:
                    code.fitness = 0 #if it was unsolved give max fitness
            if condition_add_individual:
                niche_idx = tuple(code.emb)
                if self.aces_args.path_checkpoint_archive == "":
                    code.unique_id = self.unique_id
                self.archive.append(code)
                self.fitnesses.append(code.fitness)
                if not niche_idx in self.niche_to_idx_archive:
                    self.niche_to_idx_archive[niche_idx] = []
                self.niche_to_idx_archive[niche_idx].append(code.unique_id)
                self.unique_id +=1


    def formating_chat_prompt(self, list_prompt_str: list[str]) -> list[list[dict]]:
        """Format list of prompt string to chat prompt"""
        list_prompt_chat=[]
        for prompt in list_prompt_str:
            # check whether I used syst prompt or not
            list_prompt_chat.append([{"role": "user", "content": prompt}])
        return list_prompt_chat
    
    def generate_multiple_solutions(self, codes: list[Genotype]) -> List[Genotype]:
        """Use LLM to generate multiple solutions for a list of puzzle"""
        raise NotImplementedError
    
    def evaluate_python_code(self, codes: list[Genotype]) -> List[Genotype]:
        """Evaluate python code"""
        raise NotImplementedError

    def generate_semantic_descriptors(self, puzzles: list[Genotype]) -> list[Genotype]:
        """Use LLM to evaluate codes along N programming skill dimensions"""
        raise NotImplementedError
    
    def generate_description(self, codes: list[Genotype]) -> list[Genotype]:
        """generate description (optional)"""
        return codes

    def generate_new_problems(self,list_goal_with_examples):
        """Generate new puzzles"""
        raise NotImplementedError
    
    def filter_problems(self, list_codes: list[Genotype]) -> list[Genotype]:
        """Filter problems"""
        return list_codes
    
    def sample_goals(self,):
        """
        Sample goals in the semantic space (combination of skills)
        out: list[goal] with goal: list[0/1] and size(goal) = len(self.skill_list)
        """
        n_goals_to_sample = self.aces_args.batch_size
        n_skills = len(self.skill_list)
        list_skill_targeted = []
        skills = list(range(1, n_skills+1))
        # Generate all combinations of up to 5 skills
        skill_combinations = set()
        for r in range(1, self.aces_args.max_descriptor_targeted+1):  # From 1 skill to max_descriptor_targeted skills
            skill_combinations.update(combinations(skills, r))
        skill_combinations = list(skill_combinations)
        match self.aces_args.mode_sampling_goal:
            case 'uniform':
                # sample goal uniformely
                list_idx = self.rng.choice(len(skill_combinations),size=n_goals_to_sample,replace=True)
                for idx in list_idx:
                    out = skill_combinations[idx]
                    skill_targeted = [1 if i in out else 0 for i in range(n_skills)]
                    list_skill_targeted.append(skill_targeted)
            case 'smart':
                # sample unexplored goal that are within 1 of distance of already explored goal in the semantic space
                # TODO: verify smart is working
                all_emb = list(self.niche_to_idx_archive.keys())
                all_emb = np.array([list(i) for i in all_emb]) # list of all explored niches
                
                skill_combinations_bin = [[1 if i in vec else 0 for i in range(n_skills)] for vec in skill_combinations] #list of all possible niches 
                
                #compute distance between all possible niche and all explored niches
                out=cdist(skill_combinations_bin, all_emb, metric='cityblock') 
                density=(out==1).sum(axis=1) # find every niches within a distance of 1
                density=density*(out.min(axis=1)!=0) # remove already explored niches (sampling weight = 0)
                norm= np.sum(density)
                if norm == 0.:
                    norm=1
                density_norm=density/norm

                list_idx_niches_sampled=np.random.choice(len(skill_combinations_bin),p=density_norm,size=n_goals_to_sample)
                for idx_niches_sampled in list_idx_niches_sampled:
                    binary_vectors_sampled=skill_combinations_bin[idx_niches_sampled]
                    target_skill=list(binary_vectors_sampled)
                    target_skill = [int(element) for element in target_skill]
                    list_skill_targeted.append(target_skill)
                return list_skill_targeted
            case 'none':
                list_skill_targeted = []
        return list_skill_targeted

    def sample_goal_with_examples(self):
        """sample goal and examples in context
        out: list[(list[Genotype],list[goal]) 
        with goal: list[0/1] and size(goal) = len(self.skill_list)
        list[Genotype] example to use in context, they are selected among the closest niches,
        and they each example sample from a different niche 
        """
        list_goal_with_examples = []
        list_goal = self.sample_goals()
        for goal in list_goal:
            list_archive_index = []
            
            all_emb = list(self.niche_to_idx_archive.keys())
            all_emb = np.array([list(i) for i in all_emb])

            list_coord_niches_sampled = []
            
            # compute distance between all cells explored and the target cell
            dists = cdist([goal], all_emb)[0]

            # shuffle indices to have true uniform sampling of closest niches
            # (otherwise, if two niches are at the same distance, the first one will be always sampled)
            shuffled_indices = np.arange(len(dists))
            np.random.shuffle(shuffled_indices)
            nearest_niches = shuffled_indices[np.argsort(dists[shuffled_indices])]
            
            for idx in nearest_niches:
                niche_idx = list(self.niche_to_idx_archive.keys())[idx]
                if not(niche_idx in list_coord_niches_sampled):
                    list_coord_niches_sampled.append(niche_idx)
                    archive_indexs = self.sample_examples_from_niche(niche_idx)
                    list_archive_index.append(archive_indexs)
                if len(list_archive_index)>=self.aces_args.n_fewshot_examples:
                    break
            list_few_shot_example_phenotypes = [self.archive[idx] for idx in list_archive_index]
            list_goal_with_examples.append((list_few_shot_example_phenotypes, goal))
        return list_goal_with_examples

    def sample_examples_from_niche(self,niche_idx) -> int:
        """Sample one example from a niche"""

        size_niche = len(self.niche_to_idx_archive[niche_idx])
        if size_niche == 0:
            raise ValueError('Empty niche')
        if size_niche == 1:
            # archive_index = self.rng.choice(self.niche_to_idx_archive[niche_idx])
            archive_index = int(self.niche_to_idx_archive[niche_idx][0])
            return archive_index
        match self.aces_args.sampling_strategy_examples_from_niche:
            case 'uniform':
                # sample a random niche
                
                archive_index = self.rng.choice(self.niche_to_idx_archive[niche_idx]) # sample a random individual
            case 'prob_best_5':
                # sample a code among the 5 most fit individuals with probability proportional to their fitness
                # self.nonzero[niche_idx]
                # sort_keys = sorted(lisself.nonzero.keys())
                fitness_range = [self.min_fitness(), self.max_fitness()]  # can these be -inf/+inf?
                # sort indices by fitness
                fit_idx = [(idx, self.fitnesses[idx]) for idx in self.niche_to_idx_archive[niche_idx]]
                print(f'fitnesses {[f for _, f in fit_idx]}')
                print(f'fitness range {fitness_range}')
                fit_idx = sorted(fit_idx, key=lambda x: x[1])[::-1][:5]  # 5 most fit
                if fitness_range[1] - fitness_range[0] == 0:
                    L = 1.
                else:
                    L = fitness_range[1] - fitness_range[0]
                normalized_fitnesses = [(f - fitness_range[0]) / L for _, f in fit_idx]
                normalized_fitnesses = np.array(normalized_fitnesses)
                if normalized_fitnesses.sum() == 0:  # all the individuals have the lowest possible fitness
                    normalized_fitnesses = np.ones_like(normalized_fitnesses) / len(normalized_fitnesses)
                else:
                    normalized_fitnesses = normalized_fitnesses / normalized_fitnesses.sum()
                print(f'probabilities {normalized_fitnesses}')
                archive_index = self.rng.choice([idx for idx, f, in fit_idx], p=normalized_fitnesses)
                
            case 'soft_normalised':
                # sample a code among the individuals with probability proportional to their fitness (softmax)
                puzz_idx = [idx for idx in self.niche_to_idx_archive[niche_idx]]
                qualities = np.array([self.fitnesses[idx] for idx in self.niche_to_idx_archive[niche_idx]])
                min_quality = qualities.min()
                max_quality = qualities.max()
                if abs(max_quality-min_quality) < 1e-6:
                    probabilities = np.ones(len(qualities)) / len(qualities)
                else:
                    normalized_qualities = (qualities - min_quality) / (max_quality - min_quality)
                    # Softmax calculation
                    temperature = self.aces_args.temperature_sampling_strategy_examples_from_niche
                    scaled_logits = normalized_qualities / temperature
                    # Subtract the max for numerical stability
                    exp_logits = np.exp(scaled_logits - np.max(scaled_logits))
                    probabilities = exp_logits / np.sum(exp_logits)
                try:
                    archive_index = self.rng.choice(puzz_idx, p=probabilities)
                except:
                    print("proba",probabilities)
                    print("quality",qualities)
                    raise ValueError('Error in softmax sampling')
            case _:
                raise NotImplementedError(f'Unrecognized sampling strategy "{self.aces_args.sampling_strategy_examples_from_niche}"')
        archive_index = int(archive_index)
        return archive_index

    def save_archive(self):
        base_path = self.aces_args.path_save + self.aces_args.name_experience +"_" + str(self.aces_args.seed) + "/"
        if not os.path.exists(base_path):
            os.makedirs(base_path, exist_ok=True)

        with open(base_path + f"generation_{self.idx_generation}.pkl", 'wb') as f:
            pickle.dump(self.archive, f)
        
    def rm_incorrect_puzzles(self, list_p3: list[Genotype]) -> list[Genotype]:
        """Remove incorrect puzzles"""
        return [p for p in list_p3 if p.fitness != -np.inf]
    
    def run(self):
        try:

            for id_iterations in trange(self.idx_generation,self.aces_args.n_generation):
                self.idx_generation += 1
                # Generate novel targets in semantic space
                # with some few shot examples that are close in the semantic space 
                list_goal_with_examples = self.sample_goal_with_examples()
                print("generating new goal")
                list_codes = self.generate_new_problems(list_goal_with_examples)
                # filter_problems
                list_codes = self.filter_problems(list_codes)
                print(f"generation {self.idx_generation}:\n- {len(list_codes)} goal generated")
                if len(list_codes) == 0:
                    print("no puzzle generated")
                    continue

                # generate dfficulty
                ## generate multiple solutions
                print("generating multiple solutions")
                list_codes = self.generate_multiple_solutions(list_codes)
                ## evaluate python code
                print("evaluate solutions")
                list_codes = self.evaluate_python_code(list_codes)

                list_codes = self.rm_incorrect_puzzles(list_codes)
                print(f"- {len(list_codes)} puzzles are correct")
                if len(list_codes) == 0:
                    print("no correct puzzle generated")
                    continue
                # generate semantic descriptor
                list_codes = self.generate_semantic_descriptors(list_codes)
                # generate description
                list_codes = self.generate_description(list_codes)
                self.update_archive(list_codes)
                if (id_iterations) % self.aces_args.save_every_n_generations == 0:
                    print("saving archive")
                    self.save_archive()
        except Exception as e:
            print(f"Error during run: {e}")
        self.terminate()