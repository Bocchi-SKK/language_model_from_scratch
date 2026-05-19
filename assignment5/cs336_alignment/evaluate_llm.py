from load_model_tokenizer import load_model_tokenizer
from vllm import LLM, SamplingParams
from pathlib import Path
import pandas as pd
from typing import Callable, List
from drgrpo_grader import r1_zero_reward_fn
import psutil
import os
import torch

# model, tokenizer = load_model_tokenizer()
data_directory = Path("/mnt/e/Data/cs336_data/Assignment5")
qwen_local_path = Path("/mnt/e/Data/cs336_data/Assignment5/model/Qwen2.5-Math-1.5B")
GSM8K_test_path = Path("/mnt/e/Data/cs336_data/Assignment5/GSM8K/main/test-00000-of-00001.parquet")
GSM8K_train_path = Path("/mnt/e/Data/cs336_data/Assignment5/GSM8K/main/train-00000-of-00001.parquet")
csv_output_path = data_directory/ "results/GSM8K_test_results.csv"

def print_memory_usage(note=""):
    process = psutil.Process(os.getpid())
    mem_mb = process.memory_info().rss / 1024 / 1024
    print(f"{note}Memory usage: {mem_mb:.2f} MB")

def extract_gsm8k_final_answer(answer_text:str) -> str:
    return answer_text.split("####")[-1].strip()

def evaluate_vllm(
        vllm_model: LLM,
        reward_fn: Callable[[str,str], dict[str, float]],
        prompts: List[str],
        eval_sampling_params: SamplingParams,
        answers: List[str],
        csv_output_path: Path
) -> None:
    '''
    Evaluate a language model on a list of prompts,
    compute evaluation metrics, and serialize results to disk.
    '''
    output_lists = []
    outputs = vllm_model.generate(prompts, eval_sampling_params)
    for idx, output in enumerate(outputs):
        prompt = output.prompt

        generated_text = output.outputs[0].text
        ground_truth = extract_gsm8k_final_answer(answers[idx])

        score = reward_fn('<think>' + generated_text, ground_truth)
        format_reward = score["format_reward"]
        answer_reward = score["answer_reward"]
        reward = score["reward"]    

        new_dict = {
            "prompt":prompt,
            "response":generated_text,
            "answer":answers[idx],
            "format_reward":format_reward,
            "answer_reward":answer_reward,
            "reward":reward
        }

        output_lists.append(new_dict)
    
    output_df = pd.DataFrame(output_lists)
    output_df.to_csv(csv_output_path)

if __name__ == "__main__":
    # Initialize vllm
    llm = LLM(model=str(qwen_local_path), dtype=torch.bfloat16, gpu_memory_utilization=0.85,)
    sampling_params = SamplingParams(temperature=1.0, top_p=1.0, max_tokens=768, stop=["</answer>"], include_stop_str_in_output=True)

    # Preparing prompts
    df = pd.read_parquet(GSM8K_test_path)
    prompts = []
    answers = []
    for idx, row in df.iterrows():
        question = row["question"]
        answer = row["answer"]
        r1_zero_prompt = (
            "A conversation between User and Assistant. The User asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer. The reasoning process is enclosed within <think> </think> and answer is enclosed within <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>.\n"
            f"User: {question}\n"
            "Assistant: <think>"
        )
        prompts.append(r1_zero_prompt)
        answers.append(answer)

    evaluate_vllm(vllm_model=llm, reward_fn=r1_zero_reward_fn, prompts=prompts, eval_sampling_params=sampling_params, answers=answers, csv_output_path=csv_output_path)