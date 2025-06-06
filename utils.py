import re
import pickle
import os
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM,GenerationConfig
from loguru import logger

def extract_prompt(prompt):
    # numbers:list of int; target:str
    try:
        prompt_temp=prompt.split('user\nUsing the numbers ')[1].split(', create an equation that equals ')
    except Exception as e:
        logger.error(f'extract_prompt error: {e}')
        print(prompt)
        raise e
    numbers=eval(prompt_temp[0])# list of int
    target=prompt_temp[1].split('. You can use basic arithmetic oper')[0] # str
    return numbers,target
def format_reward_func(completions, target, **kwargs):
    """
    Format: <think>...</think><answer>...</answer>
    Args:
        completions (list[str]): Generated outputs
        target (list[str]): Expected answers
      
      Returns:
          list[float]: Reward scores
    """
    rewards = []

    for completion, gt in zip(completions, target):

      try:
        # add synthetic <think> as its already part of the prompt and prefilled for the assistant to more easily match the regex
        completion = "<think>" + completion        
        # Check if the format is correct
        regex = r"^<think>([^<]*(?:<(?!/?think>)[^<]*)*)<\/think>\n<answer>([\s\S]*?)<\/answer>$"

        match = re.search(regex, completion, re.DOTALL) 
        # if the format is not correct, reward is 0
        if match is None or len(match.groups()) != 2:
            rewards.append(0.0)
        else:
            rewards.append(1.0)
      except Exception:
        rewards.append(0.0)
    return rewards

def equation_reward_func(completions, target, nums, **kwargs):
    """
    Evaluates completions based on:
    2. Mathematical correctness of the answer

    Args:
        completions (list[str]): Generated outputs
        target (list[str]): Expected answers  str
        nums (list[[list[int]]]): Available numbers list of list of int
    
    Returns:
        list[float]: Reward scores
    """
    rewards = []
    for completion, gt, numbers in zip(completions, target, nums):
      try:
        # add synthetic <think> as its already part of the prompt and prefilled for the assistant to more easily match the regex
        completion = "<think>" + completion
        # Check if the format is correct
        match = re.search(r"<answer>(.*?)<\/answer>", completion)
        if match is None:
            rewards.append(0.0)
            continue
        # Extract the "answer" part from the completion
        equation = match.group(1).strip()
        if '=' in equation:
            equation,answer=equation.split('=')  # prompt要求了最终答案部分有=，但是reward却禁止。这导致了mismatch！
            try:
                if abs(float(answer) - float(gt)) > 1e-5:
                    rewards.append(0.0)
                    continue
            except:
                rewards.append(0.0)
                continue
        # Extract all numbers from the equation
        used_numbers = [int(n) for n in re.findall(r'\d+', equation)]
        
        # Check if all numbers are used exactly once
        if sorted(used_numbers) != sorted(numbers):
            rewards.append(0.0)
            continue
        # Define a regex pattern that only allows numbers, operators, parentheses, and whitespace
        allowed_pattern = r'^[\d+\-*/().\s]+$'
        if not re.match(allowed_pattern, equation):
           rewards.append(0.0)
           continue
        
        # Evaluate the equation with restricted globals and locals
        result = eval(equation, {"__builtins__": None}, {})
        # Check if the equation is correct and matches the ground truth
        if abs(float(result) - float(gt)) < 1e-5:
            rewards.append(1.0)
        else:
            rewards.append(0.0)
      except Exception:
            # If evaluation fails, reward is 0
            rewards.append(0.0) 
    return rewards

def load_pickle(path):
    with open(path,'rb') as f:
        pickle_data = pickle.load(f)
    return pickle_data
   
def test_reward():
    correct_sample_1 = """We need to find an equation using the numbers 19, 36, 55, and 7
exactly once, with basic arithmetic operations, that equals 65. One possible
combination is 55 + 36 - 19 + 7... </think>
<answer> 55 + 36 - 7 - 19 </answer>"""

    correct_sample_2 = """ ... </think>
<answer> 55 + 36 - 7 - 19=65. </answer>"""

    wrong_format = """User: Using the numbers [19, 36, 55, 7], create an equation that equals 65."""

    wrong_format_2 = """To find the equation that equals 79 using the numbers 95, 78, 6, 88, I'll start by adding 88 and 95:                      
95 + 88 = 183                                                                                                              
Now, let's subtract 104 from 183 to get 79:
183 - 104 = 79
<think> 183 - 104 = 79 </think><think> 183 - 104 = 79 </think><answer> 183 - 104 = 79 </answer>"""

    wrong_result = """ ... </think>
<answer> 55 + 36 - 7 - 18 </answer>"""

    # 注意</think>和<answer>后面只有一个换行符，不可以手动缩进
    # 此外response不可以加<think>,因为prompt末尾最后一个token就是<think>，再reward function中也会手动添加

    test_rewards = format_reward_func(completions=[correct_sample_1, correct_sample_2, wrong_format, wrong_format_2, wrong_result], target=["65", "65", "65", "65", "65"], nums=[[19, 36, 55, 7]] * 5)
    assert test_rewards == [1.0, 1.0, 0.0, 0.0, 1.0], "Reward function is not working"
    test_rewards = equation_reward_func(completions=[correct_sample_1, correct_sample_2, wrong_format, wrong_format_2, wrong_result], target=["65", "65", "65", "65", "65"], nums=[[19, 36, 55, 7]] * 5)
    assert test_rewards == [1.0, 1.0, 0.0, 0.0, 0.0], "Reward function is not working"

def gather_rewards(path="qwen-r1-aha-moment_save_reward_trial/rewards_save"):
    # 从grpo的在线采样数据中收集rewards，用于检测训练期间的采样情况，监控某个group是否全都是0
    dirs=os.listdir(path)
    for dir in dirs:
        if 'completion' in dir:
            with open(os.path.join(path,dir),'rb') as f:
                completion = pickle.load(f)
        elif 'reward' in dir:
            with open(os.path.join(path,dir),'rb') as f:
                reward = pickle.load(f)
        elif 'prompt' in dir:
            with open(os.path.join(path,dir),'rb') as f:
                prompt = pickle.load(f)
        else:
            continue
    return completion, reward, prompt

def dpo_dataset_format(prompt2comre):
    # 从reference model的sampling结果构造dpo数据集\
    # key:prompt,value:[(completion,reward),..(completion,reward)]
    prompt2comre=load_pickle(path=prompt2comre)
    print(f"len(prompt2comre):{len(prompt2comre)})")
    preference_dataset=[]
    for prompt in prompt2comre:# only format reward
        list_of_chosen=[]
        list_of_rejected=[]
        for completion,reward in prompt2comre[prompt]:
            if reward<1:
                list_of_rejected.append(completion)
            else:
                list_of_chosen.append(completion)
        if len(list_of_chosen)==0 or len(list_of_rejected)==0:
            continue
        list_of_chosen=list(set(list_of_chosen))
        list_of_rejected=list(set(list_of_rejected))
        for chosen in list_of_chosen:
            for rejected in list_of_rejected:
                preference_dataset.append({
                    "prompt":prompt,
                    "chosen":chosen,
                    "rejected":rejected,
                })
                break
            break
    preference_dataset=preference_dataset
    return Dataset.from_list(preference_dataset),preference_dataset

def dpo_dataset_v2(prompt2comre):
    # 从reference model的sampling结果构造dpo数据集\
    # key:prompt,value:[(completion,(format,answer)),..(completion,(format,answer))]
    prompt2comre=load_pickle(path=prompt2comre)
    print(f"len(prompt2comre):{len(prompt2comre)})")
    preference_dataset=[]
    for prompt,values in prompt2comre.items():#
        list_of_chosen=[]
        list_of_rejected=[]
        completions=[x[0] for x in values]
        numbers,target=extract_prompt(prompt)
        numbers_list=[numbers]*len(completions)
        target_list=[target]*len(completions)
        format_rewards=format_reward_func(completions,completions)
        answer_rewards=equation_reward_func(completions,target_list,numbers_list)
        for completion,format,answer in zip(completions,format_rewards,answer_rewards):
            reward=format+answer
            if reward<1: # 0+0
                list_of_rejected.append(completion)
            elif reward>1: # 1+1
                list_of_chosen.append(completion)
            else:
                continue
        if len(list_of_chosen)==0 or len(list_of_rejected)==0:
            continue
        list_of_chosen=list(set(list_of_chosen))
        list_of_rejected=list(set(list_of_rejected))
        for chosen in list_of_chosen:
            for rejected in list_of_rejected:
                preference_dataset.append({
                    "prompt":prompt,
                    "chosen":chosen,
                    "rejected":rejected,
                })
                break
            break #每个prompt只有一个chosen和rejected
    preference_dataset=preference_dataset
    return Dataset.from_list(preference_dataset),preference_dataset

def ref_sampling(model_name="../qwen3b",prompt_path="dataset/train.pkl",bs=16):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    with open(prompt_path,'rb') as f:
        prompts=pickle.load(f)['prompt']
    # batch inference
    batch_size=bs
    batch_prompts=[]
    for i in range(0,len(prompts),batch_size):
        batch_prompts.append(prompts[i:i+batch_size])
    prompt2comre={}
    for batch in batch_prompts: # bs个prompt
        inputs = tokenizer(batch, return_tensors="pt", padding=True,padding_side="left")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        for temp in [0.2,0.4,0.6,0.8,1.0]: # 每个prompt下采样5次，从而尽可能得到accepted和rejected
            outputs = model.generate(
                **inputs,
                max_new_tokens=1024,
                do_sample=True,
                top_p=0.9,
                temperature=temp,
                num_return_sequences=1,
            )
            completions = tokenizer.batch_decode(outputs, skip_special_tokens=True) # bs个com
            # 这里不对，completion是带有prompt的整个句子，所以需要做处理再算rewards
            # assistant\nLet me solve this step by step.\n<think>之后的才是真实的response
            new_completions=[]
            numbers_list=[]
            target_list=[]
            for completion in completions:
                prompt,new_completion=completion.split('assistant\nLet me solve this step by step.\n<think>')
                prompt+="assistant\nLet me solve this step by step.\n<think>"
                new_completions.append(new_completion)
                numbers,target=extract_prompt(prompt)
                numbers_list.append(numbers)
                target_list.append(target)
            completions=new_completions.copy()
            del new_completions
            rewards = format_reward_func(completions,completions) # bs个reward
            answer_rewards = equation_reward_func(completions,target_list,numbers_list)
            for p,c,r,answer in zip(batch,completions,rewards,answer_rewards):
                if p not in prompt2comre:
                    prompt2comre[p]=[(c,r,answer)]
                else:
                    prompt2comre[p].append((c,r,answer))
        if len(prompt2comre)%batch_size==0:
            print(f'current progress: {len(prompt2comre)/len(prompts)}')
            with open(f'dataset/results/temp_sampling_{len(prompt2comre)}.pkl','wb') as f:
                pickle.dump(prompt2comre,f)
    return prompt2comre

def ref_sampling_ds(model_name="../qwen3b",prompt_path="dataset/train.pkl",bs=16):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype='auto', device_map="auto")
    model.generation_config = GenerationConfig.from_pretrained(model_name)
    model.generation_config.pad_token_id = model.generation_config.eos_token_id
    with open(prompt_path,'rb') as f:
        prompts=pickle.load(f)['prompt']
    # batch inference
    batch_size=bs
    batch_prompts=[]
    for i in range(0,len(prompts),batch_size):
        batch_prompts.append(prompts[i:i+batch_size])
    prompt2comre={}
    for batch in batch_prompts: # bs个prompt
        inputs = tokenizer(batch, return_tensors="pt", padding=True,padding_side="left")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        for temp in [0.2,0.4,0.6,0.8,1.0]: # 每个prompt下采样5次，从而尽可能得到accepted和rejected
            outputs = model.generate(
                **inputs,
                max_new_tokens=1024,
                do_sample=True,
                top_p=0.9,
                temperature=temp,
                num_return_sequences=1,
            )
            completions = tokenizer.batch_decode(outputs, skip_special_tokens=True) # bs个com
            # 这里不对，completion是带有prompt的整个句子，所以需要做处理再算rewards
            # \nassistant\nLet me solve this step by step.\n<think>之后的才是真实的response
            new_completions=[]
            numbers_list=[]
            target_list=[]
            for prompt,completion in zip(batch,completions):
                prompt,new_completion=completion.split('assistant\nLet me solve this step by step.\n<think>')
                prompt+="assistant\nLet me solve this step by step.\n<think>"
                new_completions.append(new_completion)
                numbers,target=extract_prompt(prompt)
                numbers_list.append(numbers)
                target_list.append(target)
            completions=new_completions.copy()
            del new_completions
            rewards = format_reward_func(completions,completions) # bs个reward
            answer_rewards = equation_reward_func(completions,target_list,numbers_list)
            for p,c,r,answer in zip(batch,completions,rewards,answer_rewards):
                if p not in prompt2comre:
                    prompt2comre[p]=[(c,(r,answer))]
                else:
                    prompt2comre[p].append((c,(r,answer)))
        if len(prompt2comre)%batch_size==0:
            print(f'current progress: {len(prompt2comre)/len(prompts)}')
            with open(f'dataset/results/temp_sampling_{len(prompt2comre)}.pkl','wb') as f:
                pickle.dump(prompt2comre,f)
    return prompt2comre


def evaluate(path):
    with open(path,'rb') as f:
        prompt2comre=pickle.load(f)
    for prompt,values in prompt2comre.items():
        temp=prompt.split("user\nUsing the numbers ")[1].split(', create an equation that equals ')
        nums=eval(temp[0])
        target=temp[1].split('. You can use basic arithmetic operations ')[0]
        new_value=[]
        for completion,_,_ in values:   ###
            #completion=completion.split('\nassistant\nLet me solve this step by step.\n<think>')[1]
            format_score=format_reward_func([completion],[completion])[0]
            outcome_score=equation_reward_func([completion],[target],[nums])[0]
            new_value.append((completion,format_score,outcome_score))
        prompt2comre[prompt]=new_value
    cnt=0
    format=0
    answer=0
    for key,values in prompt2comre.items():
        for completion,format_score,outcome_score in values:
            format+=format_score
            answer+=outcome_score
            cnt+=1
    print(f'format accuracy: {format/cnt}')
    print(f'answer accuracy: {answer/cnt}')
    with open(path.replace('.pkl','_outcome.txt'),'w') as f:
        f.write(f'format accuracy: {format/cnt}\n')
        f.write(f'answer accuracy: {answer/cnt}\n')
        f.write(f"total number of prompts: {cnt}\n")
    return prompt2comre

def dpo_dataset_main():
    dataset,res=dpo_dataset_v2('dataset/qwen7b_ins_dis/qwen7b_instruct_sampling.pkl')#('dataset/ref_sampling_train.pkl')
    print(type(dataset))
    print('final pairs:',len(res))
    with open('dataset/dpo_dataset_train_dis_qwen.pkl','wb') as f:
        pickle.dump(dataset,f)
    dataset.to_parquet('dataset/dpo_dataset_train_dis_qwen.parquet')

def sampling(model_name):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto"
    )
    logger.info("model done")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    prompt="<|im_start|>system\nYou are a helpful assistant. You first thinks about the reasoning process in the mind and then provides the user with the answer.<|im_end|>\n<|im_start|>user\nUsing the numbers [33, 65, 64, 46], create an equation that equals 50. You can use basic arithmetic operations (+, -, *, /) and each number can only be used once. Show your work in <think> </think> tags. And return the final equation and answer in <answer> </answer> tags, for example <answer> (1 + 2) / 3 = 1 </answer>.<|im_end|>\n<|im_start|>assistant\nLet me solve this step by step.\n<think>"
    logger.info(prompt)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    logger.info("inputs done")
    outputs = model.generate(
        **inputs,
        max_new_tokens=128,
        do_sample=True,
        top_p=0.9,
        temperature=0.2,
        num_return_sequences=1,
    )
    completion = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    logger.info(completion)

def eval_pipeline(model_path,prompt_path,bs=256,save_name='test'):
    prompt2comre=ref_sampling(model_name=model_path,prompt_path=prompt_path,bs=bs) # bs大 512
    with open(f'dataset/{save_name}.pkl','wb') as f:
       pickle.dump(prompt2comre,f)
    evaluate(f'dataset/{save_name}.pkl')

    
if __name__=='__main__':
    eval_pipeline('/mlx_devbox/users/liuhao.200207/playground/files/runs/dpo_full_both2','/mlx_devbox/users/liuhao.200207/playground/files/dataset/test.pkl',bs=512,save_name="dpo_full_both_self_2")
    #dpo_dataset_main()
    # prompt2comre=ref_sampling(model_name='/mlx_devbox/users/liuhao.200207/playground/DeepSeek-Math/evaluation/deepseek-ai/qwen2.5instruct7b',bs=128)
    # with open('/mlx_devbox/users/liuhao.200207/playground/files/dataset/qwen7b_instruct_sampling.pkl','wb') as f:
    #    pickle.dump(prompt2comre,f)




# nohup python utils.py > dpo_full_both.txt &