from aces.environement.p3.aces_p3 import ACES_p3
from dataclasses import dataclass, field
from typing import Optional
from transformers import HfArgumentParser
from aces.environement.p3.config_class import AcesArguments,LLMArguments
import pickle
# @dataclass
# class Label_args:
#     """
#     Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.DataTrainingArguments
#     """
#     path_save_relabel_archive: str = field(
#         default="/home/flowers/work/hf/Qwen2.5-0.5B-Instruct",#"/home/flowers/work/hf/Qwen2.5-Coder-3B-Instruct",
#         metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
#     )





# parser = HfArgumentParser((AcesArguments,LLMArguments,Label_args))
parser = HfArgumentParser((AcesArguments,LLMArguments))

# aces_args, llm_args, label_args = parser.parse_args_into_dataclasses()
aces_args, llm_args = parser.parse_args_into_dataclasses()

# aces_args, llm_args = AcesArguments(), LLMArguments()
print("args:")
print(aces_args)
print(llm_args)
# print(label_args)
from aces.environement.p3.prompt_function import get_prompt_label_p3
from aces.environement.p3.utils import extract_skill


aces = ACES_p3(aces_args, llm_args)

# check if path_save_relabel_archive exists if yes load it in aces.archive
import os
path_save = aces_args.path_checkpoint_archive
if os.path.exists(path_save):
    print(f"Loading archive from {path_save}")
    with open(path_save, 'rb') as f:
        aces.archive = pickle.load(f)

# label puzzles that are not labeled yet check archive[i].emb_bis
# puzzles = select puzzles that are not labeled yet from aces.archive
puzzles = []
puzzles_id = []
for i, puzzle in enumerate(aces.archive):
    if puzzle.emb_bis is None:
        puzzles.append(puzzle)
        puzzles_id.append(i)

puzzles_label = aces.generate_semantic_descriptors(puzzles)

# put back puzzles in list aces.archive
for i, puzzle in enumerate(puzzles):
    aces.archive[puzzles_id[i]].emb_bis = puzzles_label[i]

# save archive
with open(path_save, 'wb') as f:
    pickle.dump(aces.archive, f)

print(f"Saved archive with {len(aces.archive)} puzzles to {path_save}")