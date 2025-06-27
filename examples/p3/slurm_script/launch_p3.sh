python launch_p3_sl.py \
--path_archive "/lustre/fswork/projects/rech/imi/uqv82bm/aces/aces/environement/p3/preprocess_p3_emb_dedup_puzzles.json" \
--path_save /lustre/fswork/projects/rech/imi/uqv82bm/aces/save_data/ \
--name_experience "p3_aces_" \
--n_generation 500 \
--num_solutions 50 \
--seed 0 \
--model_name_or_path "model_sweep" \
--gpu 4 \
--swap_space 40 \
--long \
--save_every_n_generations 5


python launch_p3_sl.py \
--path_archive "/lustre/fswork/projects/rech/imi/uqv82bm/aces/aces/environement/p3/preprocess_p3_emb_dedup_puzzles.json" \
--path_save /lustre/fswork/projects/rech/imi/uqv82bm/aces/save_data/ \
--name_experience "p3_aces_100_" \
--n_generation 500 \
--num_solutions 100 \
--seed 0 \
--model_name_or_path "model_sweep" \
--gpu 4 \
--swap_space 40 \
--long \
--save_every_n_generations 4

/lustre/fsn1/projects/rech/imi/uqv82bm/hf/Qwen2.5-Coder-14B-Instruct

python launch_p3_sl.py \
--path_archive "/lustre/fswork/projects/rech/imi/uqv82bm/aces/aces/environement/p3/preprocess_p3_emb_dedup_puzzles.json" \
--path_save /lustre/fswork/projects/rech/imi/uqv82bm/aces/save_data/ \
--name_experience "qwen_c_25_7b" \
--n_generation 50 \
--num_solutions 50 \
--seed 0 \
--model_name_or_path "/lustre/fsn1/projects/rech/imi/uqv82bm/hf/Qwen2.5-Coder-7B-Instruct" \
--swap_space 20 \
--gpu 1 \
--save_every_n_generations 5 \
--dev --only_print

# test

python /home/flowers/work/aces/examples/p3/slurm_script/launch_p3_sl.py \
--path_archive "/home/flowers/work/aces/aces/environement/p3/preprocess_p3_emb_dedup_puzzles.json" \
--path_save "/home/flowers/work/aces/save_data/" \
--name_experience "qwen_c_25_3b" \
--n_generation 50 \
--num_solutions 50 \
--seed 0 \
--model_name_or_path "/home/flowers/work/hf/Qwen2.5-Coder-3B-Instruct" \
--gpu 1 \
--save_every_n_generations 5 \
--dev --only_print

# sglang
python launch_p3.py --path_archive /home/flowers/work/aces/aces/environement/p3/preprocess_p3_emb_dedup_puzzles.json --path_save /home/flowers/work/aces/save_data/ --name_experience Qwen3-4B_nsolution-10_seed_0 --n_generation 1 --num_solutions 1 --seed 0 --model_name_or_path /home/flowers/work/hf/Qwen3-4B  --gpu 1 --gpu_memory 0.9 --local_server --save_every_n_generations 1 --batch_size 4 --sglang --log_level error --temperature 0.7 --top_p 0.8 --top_k 20

# vllm
python launch_p3.py --path_archive /home/flowers/work/aces/aces/environement/p3/preprocess_p3_emb_dedup_puzzles.json --path_save /home/flowers/work/aces/save_data/ --name_experience Qwen3-4B_nsolution-10_seed_0 --n_generation 1 --num_solutions 1 --seed 0 --model_name_or_path /home/flowers/work/hf/Qwen3-4B  --gpu 1 --gpu_memory 0.9 --local_server --save_every_n_generations 1 --batch_size 4 --log_level error --temperature 0.7 --top_p 0.8 --top_k 20