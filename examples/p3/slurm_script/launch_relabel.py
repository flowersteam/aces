import os
import subprocess
import argparse
parser = argparse.ArgumentParser()  
parser.add_argument("--path_archive", type=str, default="/lustre/fswork/projects/rech/imi/uqv82bm/aces/aces/environement/p3/preprocess_p3_emb_dedup_puzzles.json")
parser.add_argument("--path_save", type=str, default="/lustre/fswork/projects/rech/imi/uqv82bm/aces/save_data/")
parser.add_argument("--name_experience", type=str, default="test")
parser.add_argument("--n_generation", type=int, default=100)
parser.add_argument("--num_solutions", type=int, default=50)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--model_name_or_path", type=str, default="model_sweep")
parser.add_argument("--path_checkpoint_archive", type=str, default="")
parser.add_argument("--gpu", type=int, default=1, help="Number of GPUs to use per nodes(default: 1)")
parser.add_argument("--gpu_memory", type=float, default=0.9, help="GPU memory usage percentage (default: 0.9)")
parser.add_argument("--only_print", action=argparse.BooleanOptionalAction, default=False)
parser.add_argument("--long", action=argparse.BooleanOptionalAction, default=False)
parser.add_argument("--local_server", action=argparse.BooleanOptionalAction, default=True, help="Use local server for LLM")
parser.add_argument("--sglang", action=argparse.BooleanOptionalAction, default=True, help="use sglang")
parser.add_argument("--save_every_n_generations", type=int, default=5)
parser.add_argument("--n_problem_to_gen", type=int, default=5, help ="Number of problem to generate for each prompt (batch_size * n_problem_to_gen = number of generated puzzles)")
parser.add_argument("--batch_size", type=int, default=32, help="Number of query to send to the LLM to create new puzzles (multiple this number by n_problem_to_gen to get the number of generated puzzles as n_problem_to_gen puzzles are generated per query)")
parser.add_argument("--dev", action=argparse.BooleanOptionalAction, help="Development mode")
parser.add_argument("--log_level", type=str, default="info", help="Log level for sglang/vllm server")

# sampling parameters
parser.add_argument("--temperature_labeller", type=float, default=0.0, help="Temperature for sampling labeller")
parser.add_argument("--temperature", type=float, default=1.0, help="Temperature for sampling")
parser.add_argument("--min_p", type=float, default=0.0, help="Min_p for sampling")
parser.add_argument("--top_p", type=float, default=1.0, help="Top-p sampling parameter")
parser.add_argument("--top_k", type=int, default=-1, help="Top-k sampling parameter, -1 for no top-k")
parser.add_argument("--fp8", action=argparse.BooleanOptionalAction, help="fp8")
parser.add_argument("--enable_thinking", action=argparse.BooleanOptionalAction, help="enable thinking mode for qwen3 models")
parser.add_argument("--max_model_length", type=int, default=-1, help="Max model length (context size) for the LLM, default is 8192")
parser.add_argument("--max_tokens", type=int, default=-1, help="Max generated tokens for the LLM, default is 8192")
parser.add_argument("--ep_moe", action=argparse.BooleanOptionalAction, help="Enable EP for MoE models")
parser.add_argument("--env_name", type=str, default="aces_sglang49p5", help="Environment name for conda activation, default is 'aces_sglang49p5' for sglang or 'aces' for non-sglang")
# nodes
parser.add_argument("--nodes", type=int, default=1, help="Number of nodes to use for the job")
args = parser.parse_args()

if args.long:
    qos = "#SBATCH --qos=qos_gpu_h100-t4"
    h = 99
elif args.dev:
    qos = "#SBATCH --qos=qos_gpu_h100-dev"
    h = 2
else:
    qos= ""
    h=20
script_template="""#!/bin/bash
#SBATCH --account=imi@h100
#SBATCH -C h100
#SBATCH --job-name={job_name}
#SBATCH --nodes={nodes}
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:{gpu}
#SBATCH --cpus-per-task={cpu}
{qos}
#SBATCH --hint=nomultithread
#SBATCH --time={h}:00:00
#SBATCH --output=./out/{job_name}-%A.out
#SBATCH --error=./out/{job_name}-%A.out
# set -x
export TORCH_CUDA_ARCH_LIST="9.0"
export TMPDIR=$JOBSCRATCH
module purge
module load arch/h100
module load python/3.11.5
ulimit -c 0
limit coredumpsize 0
export CORE_PATTERN=/dev/null


{module_load}
conda activate {env_name}
cd /lustre/fswork/projects/rech/imi/uqv82bm/aces/examples/p3/

{export_stuff}
python label_p3.py --path_archive {path_archive} --path_save {path_save} --name_experience {name_experience} --n_generation {n_generation} --num_solutions {num_solutions} --seed {seed} --model_name_or_path {model_name_or_path} {extra}
"""
# export CUDA_VISIBLE_DEVICES={gpu}
# export WORLD_SIZE=1
module_load = ""
if args.sglang:
    module_load = "module load cuda/12.8.0"
export_stuff=""
if args.log_level:
    export_stuff += f"export VLLM_LOGGING_LEVEL=ERROR"
cpu=min(24*args.gpu,96)
# for id_part in [1, 2, 3]:
base_path_model="/lustre/fsn1/projects/rech/imi/uqv82bm/hf/"

if args.model_name_or_path =="model_sweep":
    list_model = [base_path_model+"mistral-large-instruct-2407-awq", base_path_model + "Qwen2.5-Coder-32B-Instruct"]
else:
    list_model = [args.model_name_or_path]
for model in list_model:
    extra = ""
    if args.path_checkpoint_archive!="":
        extra += f' --path_checkpoint_archive {args.path_checkpoint_archive}'
    if args.gpu:
        extra += f" --gpu {args.gpu*args.nodes}"
    extra += f" --gpu_memory {args.gpu_memory}"
    if args.local_server:
        extra += " --local_server"
    if args.sglang:
        extra += " --sglang"
    extra += f" --temperature {args.temperature}"
    extra += f" --min_p {args.min_p}"
    extra += f" --top_p {args.top_p}"
    extra += f" --top_k {args.top_k}"
    extra += f" --log_level {args.log_level}"
    extra += f" --save_every_n_generations {args.save_every_n_generations}"    
    extra += f" --temperature_labeller {args.temperature_labeller}"
    if args.n_problem_to_gen:
        extra += f" --n_problem_to_gen {args.n_problem_to_gen}"
    if args.batch_size:
        extra += f" --batch_size {args.batch_size}"
    if args.enable_thinking:
        extra += " --enable_thinking"
    if args.max_model_length!= -1:
        extra += f" --max_model_length {args.max_model_length}"
    if args.max_tokens!= -1:
        extra += f" --max_tokens {args.max_tokens}"

    if args.fp8:
        extra += " --fp8"
    else:
        extra += " --no-fp8"
    if args.ep_moe:
        extra += " --ep_moe"
    model_id = model
    if "/" in model_id:
        model_id = model_id.split("/")[-1]
    job_name = f"ACES_P3_model-{model_id}"+"_"+args.name_experience + "_nsolution-"+str(args.num_solutions)

    slurmfile_path = f'run_{job_name}.slurm'
    env_name = args.env_name #"aces_sglang49p5" if args.sglang else "aces"
    name_experience= model_id+"_"+args.name_experience +"_nsolution-"+str(args.num_solutions)+ "_seed_"+str(args.seed)
    script = script_template.format(qos=qos,h=h,gpu=args.gpu,nodes=args.nodes,cpu=cpu,path_archive=args.path_archive, path_save=args.path_save, name_experience=name_experience, n_generation=args.n_generation, num_solutions=args.num_solutions, seed=args.seed, model_name_or_path=model, extra=extra, job_name=job_name,env_name=env_name,export_stuff=export_stuff,module_load=module_load)
    if args.only_print:
        print(script)
        exit()
    with open(slurmfile_path, 'w') as f:
        f.write(script)
    subprocess.call(f'sbatch {slurmfile_path}', shell=True)
    # can you rm slurm/run_{job_name}.slurm
    os.remove(slurmfile_path)
    
        
        
