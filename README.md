# aces

## Installation steps
1. (Recommended) Create a new conda environment.
```
conda create -n aces python=3.11 -y
conda activate aces
```
2. Install vLLM (see [vllm installation for latest infor](https://docs.vllm.ai/en/latest/getting_started/installation.html))
```
pip install vllm 
```
3. install ACES
```
git clone https://github.com/flowersteam/aces.git; cd aces; pip install -e .

```

## Run Aces
See examples scipts in or notebook to launch experiment with local or API LLMs [examples](examples/p3/).
Example for SLURM cluster at (slurm)[examples/p3/slurm_script/]



📦 ACES  
┣ 📂 [`aces`](babyai-text) -- *ACES code* 
┃ ┣ 📂 [`environement`](aces/environement) -- *file containing code specific to each environement*  
┃ ┃ ┣ 📜 [`load.py`](aces/environement/load.py)
┃ ┃ ┣ 📜 [`data.json`](aces/environement/data.json) 
┣ 📂 [`examples`](examples) -- *code for our experiments*    
┃ ┣ 📂 [`p3`](examples/agents) -- *implementation of all our agents*  
┃ ┃ ┣ 📂 [`slurm_script`](examples/agents/slurm_script)  -- *bot agent leveraging BabyAI's bot*  
┃ ┃ ┣ 📜 [`launch_p3_local.ipynb`](examples/agents/slurm_script/launch_p3.ipynb)  -- *Notebook to launch ACES P3 (local mode)* 
┃ ┃ ┣ 📜 [`launch_p3_API.ipynb`](examples/agents/slurm_script/launch_p3.ipynb)  -- *Notebook to launch ACES P3 (OpenAI API mode)* 
┃ ┃ ┣ 📜 [`launch_p3.py`](examples/agents/slurm_script/launch_p3.ipynb)  -- *Notebook to launch ACES P3*  
┣ 📂 [`save_data`](save_data) -- *Folder where data is saved* 
┃ ┣ 📜 [`launch_p3.ipynb`](save_data/check_data.ipynb)  -- *Notebook to explore results* 
