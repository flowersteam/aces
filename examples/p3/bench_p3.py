from aces.environement.p3.aces_p3 import ACES_p3
from dataclasses import dataclass, field
from typing import Optional
from transformers import HfArgumentParser
from aces.environement.p3.config_class import AcesArguments,LLMArguments
import os
import pickle
from tqdm import tqdm
parser = HfArgumentParser((AcesArguments,LLMArguments))
aces_args, llm_args = parser.parse_args_into_dataclasses()
# aces_args, llm_args = AcesArguments(), LLMArguments()
print("args:")
print(aces_args)
print(llm_args)
aces= ACES_p3(aces_args, llm_args)


reasoning = aces.llm.enable_thinking
model_name = aces.llm.model_path.split("/")[-1].replace(".","_")

if reasoning:
    model_name += "_reasoning"
path_load = aces_args.path_checkpoint_archive
path_save = path_load.replace(".pkl", f"_{model_name}.pkl")
if os.path.exists(path_save):
    print(f"Loading archive from {path_save}")
    with open(path_save, 'rb') as f:
        archive = pickle.load(f)
else:
    print(f"Loading archive from {path_load}")
    with open(path_load, 'rb') as f:
        archive = pickle.load(f)
        
    # initialize bench_results for each puzzle
    for puzzle in archive:
        if not hasattr(puzzle, 'bench_results'):
            puzzle.bench_results = {}
        if model_name not in puzzle.bench_results:
            puzzle.bench_results[model_name] = {}

    # set those key to None: all_solution all_solution_reasoning fitness all_solution_correct 
    for puzzle in archive:
        puzzle["all_solution"] = None
        puzzle["all_solution_reasoning"] = None
        puzzle["fitness"] = None
        puzzle["all_solution_correct"] = None


# gather puzzles that still need evaluation
archive_to_solve = [
    p for p in archive
    if not p.bench_results[model_name].get('all_solution')
]
print(f"{len(archive_to_solve)} puzzles to solve with model {model_name}")

# set mini-batch size
batch_size = 200

# evaluation loop in mini-batches
# split the puzzles into mini‐batches up front
archive_to_solve = [p for p in archive if not p.bench_results[model_name].get('all_solution')]
batches = [
    archive_to_solve[i : i + batch_size]
    for i in range(0, len(archive_to_solve), batch_size)
]

for batch_idx, batch in enumerate(tqdm(batches, desc="Batches", start=1), start=1):
    print(f"Batch {batch_idx}/{len(batches)}: Generating multiple solutions for {len(batch)} puzzles …")
    list_codes = aces.generate_multiple_solutions(batch)
    print("Evaluating solutions …")
    evaluated = aces.evaluate_python_code(list_codes)

    # update each puzzle's bench_results entry
    for result in evaluated:
        puzzle = result.puzzle
        entry = puzzle.bench_results[model_name]
        entry["all_solution"] = result.solutions
        entry["all_solution_reasoning"] = getattr(result, "reasonings", [])
        entry["fitness"] = result.fitness
        entry["all_solution_correct"] = result.correct

    # report remaining puzzles
    remaining = len(
        [p for p in archive if not p.bench_results[model_name].get("all_solution")]
    )
    print(f"{remaining} puzzles remaining")

    # save intermediate results after each mini‐batch
    with open(path_save, "wb") as f:
        pickle.dump(archive, f)
    print(f"Saved intermediate archive to {path_save} after batch {batch_idx}")
