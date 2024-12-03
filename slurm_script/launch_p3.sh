python launch_p3_sl.py \
--path_archive "/lustre/fswork/projects/rech/imi/uqv82bm/aces/aces/environement/p3/preprocess_p3_emb_dedup_puzzles.json" \
--path_save /lustre/fswork/projects/rech/imi/uqv82bm/aces/save_data/ \
--name_experience "p3_emb_dedup_puzzles" \
--n_generation 500 \
--num_solutions 100 \
--seed 0 \
--model_name_or_path "model_sweep" \
--gpu 2 \
--swap_space 35 \
--long \
--save_every_n_generations 5 



python launch_p3_sl.py \
--path_archive "/home/flowers/work/aces/aces/environement/p3/preprocess_p3_emb_3_puzzles.json" \
--path_save /home/flowers/work/aces/save_data/ \
--name_experience "p3_emb_dedup_puzzles" \
--n_generation 500 \
--num_solutions 5 \
--seed 0 \
--model_name_or_path "/home/flowers/work/hf/Qwen2.5-Coder-3B-Instruct" \
--gpu 1 \
--only_print