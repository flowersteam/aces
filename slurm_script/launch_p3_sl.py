import os
import subprocess
import argparse
parser = argparse.ArgumentParser()  
parser.add_argument("--path_archive", type=str, default="/lustre/fswork/projects/rech/imi/uqv82bm/aces/aces/environement/p3/preprocess_p3_emb_dedup_puzzles.json")
parser.add_argument("--path_save", type=str, default="/lustre/fswork/projects/rech/imi/uqv82bm/aces/save_data/")
parser.add_argument("--name_experience", type=str, default="test")
parser.add_argument("--n_generation", type=int, default=200)
parser.add_argument("--num_solutions", type=int, default=100)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--model_name_or_path", type=str, default="model_sweep")
parser.add_argument("--path_checkpoint_archive", type=str, default="")
parser.add_argument("--gpu", type=int, default=2)
parser.add_argument("--only_print", action=argparse.BooleanOptionalAction, default=False)


args = parser.parse_args()



script_template="""#!/bin/bash
#SBATCH --account=imi@h100
#SBATCH -C h100
#SBATCH --job-name={job_name}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=48

#SBATCH --hint=nomultithread
#SBATCH --time=20:00:00
#SBATCH --output=./out/{job_name}-%A.out
#SBATCH --error=./out/{job_name}-%A.out
# set -x

export TMPDIR=$JOBSCRATCH
module purge
module load arch/h100
module load python/3.11.5
ulimit -c 0
limit coredumpsize 0
export CORE_PATTERN=/dev/null


conda deactivate
conda activate vllm64
cd /lustre/fswork/projects/rech/imi/uqv82bm/aces/

python launch_p3.py --path_archive {path_archive} --path_save {path_save} --name_experience {name_experience} --n_generation {n_generation} --num_solutions {num_solutions} --seed {seed} --model_name_or_path {model_name_or_path} {extra}
"""



# for id_part in [1, 2, 3]:
base_path_model="/lustre/fsn1/projects/rech/imi/uqv82bm/hf/"

if args.model_name_or_path =="model_sweep":
    list_model = [base_path_model+"mistral-large-instruct-2407-awq", base_path_model + "Qwen2.5-Coder-32B-Instruct"]
else:
    list_model = [args.model_name_or_path]
for model in ["mistral-large-instruct-2407-awq","Qwen2.5-Coder-32B-Instruct"]:
    extra = ""
    if args.path_checkpoint_archive!="":
        extra += f' --path_checkpoint_archive {args.path_checkpoint_archive}'
    if args.gpu:
        extra += f" --gpu {args.gpu}"
    
    job_name = f"ACES_P3_model-{args.model_name_or_path}"

    slurmfile_path = f'run_{job_name}.slurm'
    script = script_template.format(path_archive=args.path_archive, path_save=args.path_save, name_experience=args.name_experience, n_generation=args.n_generation, num_solutions=args.num_solutions, seed=args.seed, model_name_or_path=model, extra=extra, job_name=job_name)
    if args.only_print:
        print(script)
        exit()
    with open(slurmfile_path, 'w') as f:
        f.write(script)
    subprocess.call(f'sbatch {slurmfile_path}', shell=True)
    # can you rm slurm/run_{job_name}.slurm
    os.remove(slurmfile_path)
    
        
        
