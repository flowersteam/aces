import copy
from tenacity import retry, wait_exponential, wait_random
from concurrent.futures import ThreadPoolExecutor
from vllm import LLM, SamplingParams

from openai import OpenAI
import requests
from requests.exceptions import RequestException
import time


class Response:
    def __init__(self, response: list, logprobs):
        self.response = response
        self.logprobs = logprobs


class LLMClient:
    def __init__(self, model: str, cfg_generation: dict, base_url: str, api_key: str, online: bool = False, gpu=1, max_model_length=20000,swap_space=5):
        self.model_path = model
        self.cfg_generation = cfg_generation
        self.base_url = base_url
        self.api_key = api_key
        self.timeout = 60*20 # 20 minutes timeout
        self.online = online
        self.gpu = gpu
        self.max_model_length = max_model_length
        self.swap_space = swap_space
        

        if online:
            self.init_client()
        else:
            self.init_offline_model()

    def init_client(self):
        server_is_up = True
        if self.base_url != "":
            # for self host server (e.g. vllm server), check if the server is up and running
            server_is_up = is_server_up(self.base_url)
        if server_is_up:
            base_url = None 
            if self.base_url != "":
                base_url = self.base_url
            api_key = None
            if self.api_key != "":
                api_key = self.api_key
            self.client = OpenAI(base_url=base_url, api_key=api_key,timeout=self.timeout)
            print("Server is up and running")
        else:
            print("Server is down or unreachable")
            raise Exception("Server is down or unreachable")

    def init_offline_model(self):
        self.llm = LLM(self.model_path, tensor_parallel_size = self.gpu, max_model_len=self.max_model_length, enable_prefix_caching=True,swap_space=self.swap_space)
         

    def multiple_completion(self, batch_prompt,judge=False,guided_choice=["1","2"],n=1,temperature=None):
        if self.online:
            # if judge:
            #     return get_multiple_completions_judge(guided_choice, self.client, batch_prompt, cfg_generation=self.cfg_generation)
            # else:
            return get_multiple_completions(self.client, batch_prompt, cfg_generation=self.cfg_generation,n=n,temperature=temperature)
        else:
            # if judge:
            #     return get_multiple_completions_judge_offline(guided_choice, self.llm, batch_prompt, cfg_generation=self.cfg_generation)
            # else:
            return get_completion_offline(self.llm, batch_prompt, cfg_generation=self.cfg_generation,n=n,temperature=temperature)
    

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

def get_completion_offline(llm,batch_prompt,cfg_generation,n=1,temperature=None):
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
    if "min_p" in kwargs:
        del kwargs["min_p"]
        # closed API doesn't support min_p 
    kwargs["n"] = n

    try:
        completion = client.chat.completions.create(
            messages=messages,
            **kwargs
        )
    except Exception as e:
        print("completion problem: ", e)
        raise e
        
    list_response = []
    #TODO: check that
    for completion_out in completion.choices:
        list_response.append(completion_out.message.content)
    # response = completion.choices[-1].message.content
    if "extra_body" in kwargs:
            if "guided_choice" in kwargs["extra_body"]:
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


if __name__ == "__main__":
    llm_url = "http://localhost:8000/v1"
    llm_model = "meta-llama/Llama-3.2-3B-Instruct"
    
    client = OpenAI(base_url=llm_url, api_key="token-abc123")
    cfg_generation ={"model": llm_model, "temperature": 1, "logprobs": True}

    system_prompt = "You are a good barman"
    user_prompt = "Yo get me a beer"

    messages = [
         {"role": "system", "content": system_prompt}, 
         {"role": "user", "content": user_prompt}
    ]

    out = get_multiple_completions(client, [messages], cfg_generation)

    print(out[0].response)
    print(out[0].logprobs)

    messages = [

        {
            "role": "user", 
            "content":  "You need to choose randomly between 'B' and 'A'"
        },
    ]
    guided_choice=["A","B"]

    out = get_multiple_completions_judge(guided_choice, client, [messages], cfg_generation)
    print(out[0].response)
    print(out[0].logprobs)
