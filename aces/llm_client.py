import copy
from tenacity import retry, wait_exponential, wait_random
from concurrent.futures import ThreadPoolExecutor

import subprocess
import numpy as np
from openai import OpenAI, AzureOpenAI
import requests
from requests.exceptions import RequestException
import time
import os

class Response:
    def __init__(self, response: list, logprobs):
        self.response = response
        self.logprobs = logprobs

def launch_vllm_serv(model_path: str, gpu: int = 1, max_model_length=20000, port: int = 8000, fp8: bool = False, gpu_memory=0.9, seed: int = 0, log_level="info", add_yarn=False):
    command = f"vllm serve {model_path} --tensor-parallel-size {gpu} --max-model-len {max_model_length}  --port {port} --gpu-memory-utilization {gpu_memory} --seed {seed} --trust-remote-code --uvicorn-log-level {log_level} "
    if fp8:
        command += "--quantization fp8 "
    list_mistral = ["Mistral-Small-3.2-24B-Instruct-2506","Mistral-Large-Instruct","Codestral-22B-v0.1","Devstral-Small-2505","Magistral-Small-2506"] 
    for model_name in list_mistral:
        if model_name in model_path:
            command += "--tokenizer_mode mistral --config_format mistral --load_format mistral "

    # for qwq and qwen 3 model
    if add_yarn:
        base_model_len = 32768
        if max_model_length < base_model_len:
            pass
        elif max_model_length < 2* base_model_len:
            command += """--rope-scaling '{"rope_type":"yarn","factor":2.0,"original_max_position_embeddings":32768}' --max-model-len 65536 """
        elif max_model_length < 4* base_model_len:
            command +="""--rope-scaling '{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":32768}' --max-model-len 131072 """

    server_process = execute_shell_command(
        command
    )
    print(command)
    # stuff to add later
    # --uvicorn-log-level {debug,info,warning,error,critical,trace}
    # --reasoning-parser
    # --enable-reasoning
    return server_process

def launch_sglang_serv(model_path: str, gpu: int = 1, max_model_length=20000, port: int = 8000, fp8: bool = False, gpu_memory=0.9, seed: int = 0, log_level="info", add_yarn=False):
    command = f"python -m sglang.launch_server --model-path {model_path} --tp {gpu} --port {port} --mem-fraction-static {gpu_memory} --random-seed {seed} --host 0.0.0.0 --log-level {log_level} --trust-remote-code "
    if "fp8" in model_path:
        fp8 = False
    if fp8:
        command += "--quantization fp8 "

    # for qwq and qwen 3 model
    if add_yarn:
        base_model_len = 32768
        if max_model_length < base_model_len:
            command += "--context-length {max_model_length} "
        elif max_model_length < 2* base_model_len:
            command += '--json-model-override-args '+ '{"rope_scaling":{"rope_type":"yarn","factor":2.0,"original_max_position_embeddings":32768}}'+' --context-length 65536 '
        elif max_model_length < 4* base_model_len:
            command += '--json-model-override-args '+'{"rope_scaling":{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":32768}}'+' --context-length 131072 '
    else:
        command += f"--context-length {max_model_length} "
            
    server_process = execute_shell_command(
        command
    )
    print(command)
    # stuff to add later
    # --uvicorn-log-level {debug,info,warning,error,critical,trace}
    # --reasoning-parser
    # --enable-reasoning
    return server_process

def launch_sglang_serv_multi_node(model_path: str, gpu: int = 1, max_model_length=20000, port: int = 8000, port_multinode:int = 5000, fp8: bool = False, gpu_memory=0.9, seed: int = 0, log_level="info", add_yarn=False, ep_moe=False):
    tp = gpu
    
    worker_num = int(os.environ.get('SLURM_NNODES', 2))
    n_nodes = worker_num
    SLURM_JOB_NODELIST = os.environ.get('SLURM_JOB_NODELIST')

    SLURM_JOB_ID = os.environ.get('SLURM_JOB_ID')
    nodes_result = subprocess.run(['scontrol', 'show', 'hostnames', SLURM_JOB_NODELIST], 
                                capture_output=True, text=True)

    # Split the output into individual node names
    nodes = [node.strip() for node in nodes_result.stdout.strip().split('\n') if node.strip()]
    head_node = nodes[0]

    # Get the IP address of the head node
    head_node_ip_result = subprocess.run(['srun', '--nodes=1', '--ntasks=1', '-w', head_node, 
                                        'hostname', '--ip-address'], 
                                        capture_output=True, text=True)
    head_node_ip = head_node_ip_result.stdout.strip()

    # Handle potential space-separated IP addresses (IPv4/IPv6) - take the first one
    # head_node_ip = head_node_ip.split()[0]

    # Set environment variable
    os.environ['SGLANG_HOST_IP'] = head_node_ip
    print(f"Head node: {head_node}, Head node IP: {head_node_ip}")

    head_env = os.environ.copy()
    head_env['OUTLINES_CACHE_DIR'] = f"/tmp/{SLURM_JOB_ID}_0"


    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} Job {SLURM_JOB_ID} started ...")
    model = model_path
    if "fp8" in model_path.lower():
        fp8 = False

    command_sglang = f"python3 -m sglang.launch_server --model-path {model} --tp {tp} --port {port} --mem-fraction-static {gpu_memory} --random-seed {seed} --host 0.0.0.0 --log-level {log_level} --trust-remote-code "

    if fp8:
        command_sglang += "--quantization fp8 "

    # for qwq and qwen 3 model
    if add_yarn:
        base_model_len = 32768
        if max_model_length < base_model_len:
            command_sglang += "--context-length {max_model_length} "
        elif max_model_length < 2* base_model_len:
            command_sglang += '--json-model-override-args '+ '{"rope_scaling":{"rope_type":"yarn","factor":2.0,"original_max_position_embeddings":32768}}'+' --context-length 65536 '
        elif max_model_length < 4* base_model_len:
            command_sglang += '--json-model-override-args '+'{"rope_scaling":{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":32768}}'+' --context-length 131072 '
    else:
        command_sglang += f"--context-length {max_model_length} "

    command_sglang += f"--dist-init-addr {head_node_ip}:{port_multinode} --nnodes {n_nodes} "
    if ep_moe:
        command_sglang += "--enable-ep-moe "
    
    head_bash_command = f"""echo "BEGIN_IP on Head Node:" && hostname -I && echo "END_IP on Head Node" && \
{command_sglang} --node-rank 0"""

    print(f"Head Node command: {head_bash_command}")
    head_process = subprocess.Popen([
        'srun', '--nodes=1', '--ntasks=1', '-w', head_node, 
        'bash', '-c', head_bash_command
    ], env=head_env)

    HEAD_PID = head_process.pid

    # --- Give Head Node Time to Initialize ---
    print("Waiting for head node to initialize...")
    time.sleep(10)  # Adjust this time if necessary

    # --- Launch Worker Nodes ---
    
    worker_processes = []

    # Loop starts from 1 because 0 is the head node
    for i in range(1, worker_num):
        node_i = nodes[i]
        print(f"STARTING WORKER {i} (Rank {i}) at {node_i}")
        
        worker_env = os.environ.copy()
        worker_env['OUTLINES_CACHE_DIR'] = f"/tmp/{SLURM_JOB_ID}_{i}"

        worker_bash_command = f"{command_sglang} --node-rank {i}"

        print(f"Worker {i} command: {worker_bash_command}")
        worker_process = subprocess.Popen([
            'srun', '--nodes=1', '--ntasks=1', '-w', node_i,
            'bash', '-c', worker_bash_command
        ], env=worker_env)
        
        worker_processes.append(worker_process)

    # Optional: Wait for all processes to complete or handle them as needed
    # For example, to wait for all processes:
    # head_process.wait()
    # for worker_process in worker_processes:
    #     worker_process.wait()

    # Or to keep the main script running while background processes execute:
    print("All processes launched. Main script continuing...")



    
    # command = f"python -m sglang.launch_server --model-path {model_path} --tp {gpu} --port {port} --mem-fraction-static {gpu_memory} --random-seed {seed} --host 0.0.0.0 --log-level {log_level} --trust-remote-code "
    
    # if fp8:
    #     command += "--quantization fp8 "

    # # for qwq and qwen 3 model
    # if add_yarn:
    #     base_model_len = 32768
    #     if max_model_length < base_model_len:
    #         command += "--context-length {max_model_length} "
    #     elif max_model_length < 2* base_model_len:
    #         command += '--json-model-override-args '+ '{"rope_scaling":{"rope_type":"yarn","factor":2.0,"original_max_position_embeddings":32768}}'+' --context-length 65536 '
    #     elif max_model_length < 4* base_model_len:
    #         command += '--json-model-override-args '+'{"rope_scaling":{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":32768}}'+' --context-length 131072 '
    # else:
    #     command += f"--context-length {max_model_length} "
            
    # server_process = execute_shell_command(
    #     command
    # )
    # print(command)
    # stuff to add later
    # --uvicorn-log-level {debug,info,warning,error,critical,trace}
    # --reasoning-parser
    # --enable-reasoning
    return head_process

def check_server_run(model_path, port, server_process,vllm=False):
    """Check if the server is running and serving the correct model.
    Needed when launching multiple inference processes on a cluster.
    """
    try:
        wait_for_server(f"http://localhost:{port}")
        time.sleep(15)
        if vllm:
            req= f"http://localhost:{port}/v1/models"
        else:
            req= f"http://localhost:{port}/get_model_info"
        response = requests.get(
            req,
            headers={"Authorization": "Bearer None"},
        )
        if vllm:
            model_id_serv = response.json()["data"][0]["id"]
        else:
            model_id_serv = response.json()["model_path"]
            good_model = response.json()["model_path"] == model_path
        print("model_id_serv", model_id_serv)
        print("model_path", model_path)
        good_model = model_id_serv == model_path
        print("good_model", good_model)
        if not good_model:
            raise Exception("wrong model")
    except:
        return False
    is_running = server_process.poll() is None
    print("is_running", is_running)
    if not is_running:
        return False
    return True

class LLMClient:
    def __init__(self, model: str, cfg_generation: dict, base_url: str ="",
                  api_key: str="None", online: bool = False, gpu=1,
                    max_model_length=20000, azure=False,
                    local_server=False, seed=0, fp8=False, gpu_memory=0.9,
                    sglang= False, log_level="info",enable_thinking=True, ep_moe = False):
        self.model_path = model
        self.cfg_generation = cfg_generation
        self.base_url = base_url
        self.api_key = api_key
        self.timeout = 60*60*2 # 2 h timeout
        self.online = online
        self.gpu = gpu
        self.max_model_length = max_model_length
        self.azure = azure
        self.local_server = local_server
        # should add vllm or sglang option
        model_lower = model.lower()

        # qwen3 for specific stuff link to qwen3 (yarn, hybrid thinking, etc.) 
        self.qwen3 = "qwen3" in model_lower 
        self.seed = seed
        self.fp8 = fp8
        if "fp8" in model_lower:
            self.fp8 = False
        self.gpu_memory = gpu_memory
        self.sglang = sglang
        self.log_level = log_level # default log level for sglang
        self.enable_thinking = enable_thinking 
        self.ep_moe = ep_moe
        if self.qwen3: 
            if "coder" in model_lower  or "instruct" in model_lower or "thinking" in model_lower:
                self.qwen3 = False

        if not enable_thinking: # as default, enable_thinking is True
            if not "extra_body" in self.cfg_generation:
                self.cfg_generation["extra_body"] = {}
            
            self.cfg_generation["extra_body"].update({"chat_template_kwargs":{"enable_thinking": self.enable_thinking}})
        self.reasoning_parser = ""
        if online:
            self.init_client()
        else:
            self.init_offline_model()

    def init_client(self):
        server_is_up = True
        if self.local_server:
            # stuff for slurm protection in case of multiple jobs
            port = np.random.randint(30000,30100)
            # TODO: vllm/ SGLang
            n_nodes = int(os.environ.get('SLURM_NNODES', 1))
            if self.sglang:
                if n_nodes > 1:
                    port_multinode = np.random.randint(5000,7000)
                    server_process = launch_sglang_serv_multi_node(model_path=self.model_path, gpu= self.gpu,
                                                    max_model_length=self.max_model_length, port= port,port_multinode= port_multinode,
                                                    fp8 = self.fp8, gpu_memory=self.gpu_memory, 
                                                    seed = self.seed, log_level = self.log_level,
                                                    add_yarn=self.qwen3 or "qwq" in self.model_path.lower(),
                                                    ep_moe=self.ep_moe)
                else:
                    server_process = launch_sglang_serv(model_path=self.model_path, gpu= self.gpu,
                                                    max_model_length=self.max_model_length, port= port,
                                                    fp8 = self.fp8, gpu_memory=self.gpu_memory, 
                                                    seed = self.seed, log_level = self.log_level,
                                                    add_yarn=self.qwen3 or "qwq" in self.model_path.lower())
                
            else:
                server_process = launch_vllm_serv(model_path=self.model_path, gpu= self.gpu,
                                                max_model_length=self.max_model_length, port= port,
                                                fp8 = self.fp8, gpu_memory=self.gpu_memory, 
                                                seed = self.seed, log_level = self.log_level,
                                                add_yarn = self.qwen3 or "qwq" in self.model_path.lower())
            print("check server run 0")
            if n_nodes > 1:
                n_tries = 4
            else:
                n_tries = 1
            for i_try in range(n_tries):
                is_running = check_server_run(self.model_path,port,server_process,vllm=not self.sglang)
                if is_running:
                    break

            

            if not is_running:
                for i_try in range(1,3):
                    port += 1
                    if self.sglang:
                        server_process = launch_sglang_serv(model_path=self.model_path, gpu= self.gpu,
                                                        max_model_length=self.max_model_length, port= port,
                                                        fp8 = self.fp8, gpu_memory=self.gpu_memory, seed = self.seed)
                    else:
                        # launch vllm server
                        server_process = launch_vllm_serv(model_path=self.model_path, gpu= self.gpu,
                                        max_model_length=self.max_model_length, port= port,
                                        fp8 = self.fp8, gpu_memory=self.gpu_memory, seed = self.seed)
                    print("check server run ", i_try)
                    is_good = check_server_run(self.model_path,port,server_process,vllm=not self.sglang)
                    if is_good:
                        break
            else:
                print(' /!\ Server is running /!\ ')
            self.base_url=f"http://localhost:{port}/v1" 
            self.api_key="None"
            is_good = check_server_run(self.model_path,port,server_process,vllm=not self.sglang)
            self.server_process = server_process
            if not is_good:
                raise Exception("wrong model")

        elif self.base_url != "" and self.local_server:
            # for self host server (e.g. vllm server), check if the server is up and running
            server_is_up = is_server_up(self.base_url)
        
        if server_is_up: 
            api_key = None
            if self.api_key != "":
                api_key = self.api_key
            if self.azure: 
                self.client = AzureOpenAI(base_url=self.base_url, api_key=api_key,timeout=self.timeout,api_version="2024-12-01-preview",)
            else:
                self.client = OpenAI(base_url=self.base_url, api_key=api_key,timeout=self.timeout)

            print("Server is up and running")
        else:
            print("Server is down or unreachable")
            raise Exception("Server is down or unreachable")


    def get_reasing_parser(self):
        """not use for now, (just split based on "</think>')"""
        model = self.model_path.lower()
        self.reasoning_parser =""
        if self.qwen3 and self.enable_thinking:
            self.reasoning_parser = "qwen3" 
        if "qwq" in model:
            self.reasoning_parser = "deepseek-r1"
        if "r1" in model:
            self.reasoning_parser = "deepseek-r1"
        if "deepcoder" in model:
            self.reasoning_parser = "deepseek-r1"


    def init_offline_model(self):
        from vllm import LLM
        # switch dtype to half if GPUs with compute capability is inferior to 8.0
        import torch
        compute_capability = torch.cuda.get_device_capability()
        major, minor = compute_capability
        # Compute capability as a single number (e.g., 7.5)
        gpu_compute_capability = major + minor / 10.0

        # Set dtype based on compute capability
        if gpu_compute_capability < 8.0:
            dtype = "float16"  # Use half-precision for older GPUs
        else:
            dtype = "auto"  # Use bfloat16 or keep default for newer GPUs
            
            
        self.llm = LLM(self.model_path, tensor_parallel_size = self.gpu, max_model_len=self.max_model_length, enable_prefix_caching=True,swap_space=self.swap_space,dtype=dtype)

    def terminate(self):
        try:
            terminate_process(self.server_process) 
        except Exception as e:
            print(f"Error terminating server process: {e}")

    def multiple_completion(self, batch_prompt,judge=False,guided_choice=["1","2"],n=1,temperature=None):
        batch_prompt = self.add_reasoning_system_prompt(batch_prompt)

        if self.online:
            # if judge:
            #     return get_multiple_completions_judge(guided_choice, self.client, batch_prompt, cfg_generation=self.cfg_generation)
            # else:
            return get_multiple_completions(self.client, batch_prompt, cfg_generation=self.cfg_generation,n=n,temperature=temperature)
        else:
            # if judge:
            #     return get_multiple_completions_judge_offline(guided_choice, self.llm, batch_prompt, cfg_generation=self.cfg_generation)
            # else:
            return get_completion_offline(self.llm, batch_prompt, cfg_generation=self.cfg_generation,n=n,temperature=temperature,qwen3=self.qwen3)
    
    def add_reasoning_system_prompt(self, batch_prompt):
        """Prepend the system prompt to each message in the batch."""
        model = self.model_path.lower()
        if "magistral" in model:
            sys_prompt = magistral_sys_prompt
        elif "llama-3_3-nemotron-super" in model:
            sys_prompt =  f"detailed thinking {self.enable_thinking}"
            if self.enable_thinking:
                sys_prompt =  f""
            else:
                sys_prompt =  f"/no_think"
        else:
            return batch_prompt
        patched_batch = []
        for message in batch_prompt:
            # Find the user message content
            patched_message = [{"role": "system", "content": sys_prompt}] + message
            patched_batch.append(patched_message)
        return patched_batch


def is_server_up(base_url):
    attempt=50
    while attempt>0:
        try:
            # Try to connect to the health endpoint or root endpoint
            if "/v1" in base_url:
                base_u=base_url.replace("/v1","")
            else:
                base_u=base_url
            response = requests.get(f"{base_u}/health", timeout=5)

            print(response.status_code)
            flag = response.status_code == 200
            if flag:
                print("="*20)
                print("serv succesfuly initializes")
                print("="*20)
                return True
            else:
                raise
        except RequestException as e:
            print(e)
            flag= False
        attempt-=1
        time.sleep(20)

def get_completion_offline(llm,batch_prompt,cfg_generation,n=1,temperature=None,qwen3=False):
    from vllm import  SamplingParams

    tokenizer = llm.get_tokenizer()
    list_tok_allowed=[]
    flag_judge=False
    if temperature is None:
        temperature = cfg_generation["temperature"]
    
    sampling_params = SamplingParams(
        n=n,
        temperature=temperature,
        max_tokens=cfg_generation["max_tokens"],
        min_p = cfg_generation["min_p"],
        )
    if "extra_body" in cfg_generation:
         if  "guided_choice" in cfg_generation["extra_body"]:
            guided_choice=cfg_generation["extra_body"]["guided_choice"]
            alowed_tokens=[]
            for tok in guided_choice:
                alowed_tokens.append(tokenizer.encode(tok,add_special_tokens=False)[-1])

                sampling_params = SamplingParams(
                    temperature=cfg_generation["temperature"],
                    max_tokens=1,
                    logprobs=5,
                    allowed_token_ids=alowed_tokens
                    )
                flag_judge=True
    if qwen3: # enable_thinking always true 
        batch_prompt_formated = tokenizer.apply_chat_template(batch_prompt,tokenize=False,add_generation_prompt=True,enable_thinking=True)
    else:
        batch_prompt_formated = tokenizer.apply_chat_template(batch_prompt,tokenize=False,add_generation_prompt=True)
    outs = llm.generate(batch_prompt_formated,sampling_params)
    list_out_process=[]
    logprobs=None
    
    for completion in outs:
        list_response = []
        for completion_out in completion.outputs:
            list_response.append(completion_out.text)
        
        if flag_judge:
            list_log=completion.outputs[0].logprobs
            # print(list_log)
            logprobs = extract_top_logprobs_offline(list_log,guided_choice=guided_choice)

        list_out_process.append(Response(list_response,logprobs))
    return list_out_process

def extract_top_logprobs_offline(list_log,guided_choice):
    dic_logprobs={}
    cumulative_sum=0
    for token_info in list_log[0].values():
        
        if token_info.decoded_token in guided_choice:
            dic_logprobs[token_info.decoded_token] = token_info.logprob
        cumulative_sum+=token_info.logprob
    return dic_logprobs 

def get_multiple_completions_judge_offline(guided_choice,llm,batch_prompt: list[list], cfg_generation: dict={}, max_workers=90, temperature=None, n=1)->list[list[str]]:
        cfg_generation = copy.deepcopy(cfg_generation)
        if not "extra_body" in cfg_generation:
                cfg_generation["extra_body"] ={}
        cfg_generation["max_tokens"]=1
        cfg_generation["extra_body"]["guided_choice"] = guided_choice
        cfg_generation["logprobs"]=5
        cfg_generation["top_logprobs"]=10
        
        return get_completion_offline(batch_prompt =batch_prompt,llm=llm, cfg_generation=cfg_generation)


# @retry(wait=wait_exponential(multiplier=1, min=30, max=600)+wait_random(min=0, max=1))
@retry(wait=wait_exponential(multiplier=1, min=10, max=600)+wait_random(min=0, max=1))
def get_completion(client, cfg_generation: dict, messages: list, temperature=None, n=1) -> list[str]:
    """Get completion(s) from OpenAI API"""
    kwargs = cfg_generation.copy()
    if temperature is not None:
        kwargs["temperature"] = temperature
    # if "min_p" in kwargs:
    #     del kwargs["min_p"]
        # closed API doesn't support min_p 
    kwargs["n"] = n

    try:
        completion = client.chat.completions.create(
            messages=messages,
            **kwargs
        )
    except Exception as e:
        print("completion problem: ", e)
        too_long = "longer than the model's context length" in e.body["message"]
        if too_long:
            return [e.body["message"]] * n
        return [None] * n

        raise e
        
    list_response = []
    #TODO: check that
    for completion_out in completion.choices:
        list_response.append(completion_out.message.content)
    # response = completion.choices[-1].message.content
    if "extra_body" in "guided_choice" in kwargs["extra_body"]:
        logprobs = extract_top_logprobs(completion.choices[-1],guided_choice=kwargs["extra_body"]["guided_choice"])
    else:
        logprobs = None
    
    return Response(list_response,logprobs)
        
def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]

def get_multiple_completions(client, batch_prompt: list[list], cfg_generation: dict={}, max_workers=90, temperature=None, n=1)->list[list[str]]:
    """Get multiple completions from OpenAI API"""
    if isinstance(batch_prompt, str):
        batch_prompt = [batch_prompt]
    
    completions = []
    count=0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for sub_batch in chunks(batch_prompt, max_workers):
            
            for message in sub_batch:
                count+=1
                kwargs = {
                    "client": client,
                    "messages": message,
                    "cfg_generation": cfg_generation,
                    "temperature": temperature,
                    "n": n
                }
                future = executor.submit(get_completion, **kwargs)
                completions.append(future)

            print(f"send {count} / {len(batch_prompt)} messages")

    # Retrieve the results from the futures
    out_n = [future.result() for future in completions]
    return out_n

# def get_multiple_completions_judge(guided_choice,*kwargs)->list[list[str]]:
#     copy.deepcopy()

def get_multiple_completions_judge(guided_choice,client,batch_prompt: list[list], cfg_generation: dict={}, max_workers=90, temperature=None, n=1)->list[list[str]]:
        cfg_generation = copy.deepcopy(cfg_generation)
        if not "extra_body" in cfg_generation:
                cfg_generation["extra_body"] ={}
        cfg_generation["max_tokens"]=1
        cfg_generation["extra_body"]["guided_choice"] = guided_choice
        cfg_generation["logprobs"]=True
        cfg_generation["top_logprobs"]=10

        return get_multiple_completions(batch_prompt =batch_prompt,client=client, cfg_generation=cfg_generation, max_workers=max_workers, temperature=temperature, n=n)

def extract_top_logprobs(response_choice,guided_choice):
    """
    Extract log probabilities from an OpenAI API response.
    
    Returns:*
        dict: with token that are in guided_choice with its logprobs
    """
    if not response_choice.logprobs:
        return None
        
    list_logprobs = []
    dic_logprobs={}
    cumulative_sum=0
    top_logprobs = response_choice.logprobs.content[0].top_logprobs # select first token generated
    for token_info in top_logprobs:
        
        if token_info.token in guided_choice:
            dic_logprobs[token_info.token] = token_info.logprob
        list_logprobs.append(token_info.logprob)
        cumulative_sum+=token_info.logprob
    return dic_logprobs


def execute_shell_command(command: str) -> subprocess.Popen:
    """
    Execute a shell command and return its process handle.
    """
    command = command.replace("\\\n", " ").replace("\\", " ")
    parts = command.split()
    return subprocess.Popen(parts, text=True, stderr=subprocess.STDOUT)

def wait_for_server(base_url: str, timeout: int = None) -> None:
    """Wait for the server to be ready by polling the /v1/models endpoint.

    Args:
        base_url: The base URL of the server
        timeout: Maximum time to wait in seconds. None means wait forever.
    """
    start_time = time.perf_counter()
    while True:
        try:
            response = requests.get(
                f"{base_url}/v1/models",
                headers={"Authorization": "Bearer None"},
            )
            if response.status_code == 200:
                time.sleep(5)

                break

            if timeout and time.perf_counter() - start_time > timeout:
                raise TimeoutError("Server did not become ready within timeout period")
        except requests.exceptions.RequestException:
            time.sleep(1)

# stuff to terùminate the process and release the port
import threading
import signal
import psutil
import os
import sys
import weakref
process_socket_map = weakref.WeakKeyDictionary()


def terminate_process(process):
    """
    Terminate the process and automatically release the reserved port.
    """

    kill_process_tree(process.pid)

    lock_socket = process_socket_map.pop(process, None)
    if lock_socket is not None:
        release_port(lock_socket)

def release_port(lock_socket):
    """
    Release the reserved port by closing the lock socket.
    """
    try:
        lock_socket.close()
    except Exception as e:
        print(f"Error closing socket: {e}")



def kill_process_tree(parent_pid, include_parent: bool = True, skip_pid: int = None):
    """Kill the process and all its child processes."""
    # Remove sigchld handler to avoid spammy logs.
    if threading.current_thread() is threading.main_thread():
        signal.signal(signal.SIGCHLD, signal.SIG_DFL)

    if parent_pid is None:
        parent_pid = os.getpid()
        include_parent = False

    try:
        itself = psutil.Process(parent_pid)
    except psutil.NoSuchProcess:
        return

    children = itself.children(recursive=True)
    for child in children:
        if child.pid == skip_pid:
            continue
        try:
            child.kill()
        except psutil.NoSuchProcess:
            pass

    if include_parent:
        try:
            if parent_pid == os.getpid():
                itself.kill()
                sys.exit(0)

            itself.kill()

            # Sometime processes cannot be killed with SIGKILL (e.g, PID=1 launched by kubernetes),
            # so we send an additional signal to kill them.
            itself.send_signal(signal.SIGQUIT)
        except psutil.NoSuchProcess:
            pass

magistral_sys_prompt = """A user will ask you to solve a task. You should first draft your thinking process (inner monologue) until you have derived the final answer. Afterwards, write a self-contained summary of your thoughts (i.e. your summary should be succinct but contain all the critical steps you needed to reach the conclusion). You should use Markdown and Latex to format your response. Write both your thoughts and summary in the same language as the task posed by the user.

Your thinking process must follow the template below:
<think>
Your thoughts or/and draft, like working through an exercise on scratch paper. Be as casual and as long as you want until you are confident to generate a correct answer.
</think>

Here, provide a concise summary that reflects your reasoning and presents a clear final answer to the user.

Problem:"""


if __name__ == "__main__":
    # test multi nodes
    model = "/lustre/fsn1/projects/rech/imi/uqv82bm/hf/GLM-4.5-Air-FP8"#Qwen2.5-14B-Instruct" #DeepSeek-R1-0528"
    cfg_generation = {"model": model, "temperature": 0.6}
    online=True
    gpu=4
    local_server = True
    fp8=False
    sglang = True
    ep_moe = True
    llm = LLMClient(model=model, cfg_generation=cfg_generation, online=online, gpu=gpu, local_server=local_server, fp8=fp8, sglang=sglang, ep_moe=ep_moe)
    test_messages = [
        {"role": "system", "content": "You are a good assistant"},
        {"role": "user", "content": "Is it chocolatine or pain au chocolat?"},
    ]
    
    out = llm.multiple_completion([test_messages], n=1)
    print(out[0].response)
    # llm_url = "http://localhost:8000/v1"
    # llm_model = "meta-llama/Llama-3.2-3B-Instruct"
    
    # client = OpenAI(base_url=llm_url, api_key="token-abc123")
    # cfg_generation ={"model": llm_model, "temperature": 1, "logprobs": True}

    # system_prompt = "You are a good barman"
    # user_prompt = "Yo get me a beer"

    # messages = [
    #      {"role": "system", "content": system_prompt}, 
    #      {"role": "user", "content": user_prompt}
    # ]

    # out = get_multiple_completions(client, [messages], cfg_generation)

    # print(out[0].response)
    # print(out[0].logprobs)

    # messages = [

    #     {
    #         "role": "user", 
    #         "content":  "You need to choose randomly between 'B' and 'A'"
    #     },
    # ]
    # guided_choice=["A","B"]

    # out = get_multiple_completions_judge(guided_choice, client, [messages], cfg_generation)
    # print(out[0].response)
    # print(out[0].logprobs)
