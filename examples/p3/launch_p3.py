from aces.environement.p3.aces_p3 import ACES_p3
from dataclasses import dataclass, field
from typing import Optional
from transformers import HfArgumentParser

    

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
    azure: Optional[bool] = field(
        default=False,
        metadata={
            "help": "use azure if True"
        },
    )
    openai_api: Optional[bool] = field(
        default=False,
        metadata={
            "help": "use openai_api if True"
        },
    )

parser = HfArgumentParser((AcesArguments,LLMArguments))
aces_args, llm_args = parser.parse_args_into_dataclasses()
# aces_args, llm_args = AcesArguments(), LLMArguments()
print("args:")
print(aces_args)
print(llm_args)
aces= ACES_p3(aces_args, llm_args)
aces.run()