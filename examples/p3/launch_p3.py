from aces.environement.p3.aces_p3 import ACES_p3
from dataclasses import dataclass, field
from typing import Optional
from transformers import HfArgumentParser
from aces.environement.p3.config_class import AcesArguments,LLMArguments

parser = HfArgumentParser((AcesArguments,LLMArguments))
aces_args, llm_args = parser.parse_args_into_dataclasses()
# aces_args, llm_args = AcesArguments(), LLMArguments()
print("args:")
print(aces_args)
print(llm_args)
aces= ACES_p3(aces_args, llm_args)
aces.run()