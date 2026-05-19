from vllm import LLM, SamplingParams
from pathlib import Path
import pandas as pd
from typing import Callable, List
from drgrpo_grader import r1_zero_reward_fn
import psutil
import os
import torch
from transformers import PreTrainedModel, AutoModelForCausalLM, AutoTokenizer
import pandas as pd
import helper_methods
from tqdm import tqdm

try:
    import file_paths
    from load_model_tokenizer import load_model_tokenizer
except:
    try:
        from cs336_alignment import file_paths
        from cs336_alignment.load_model_tokenizer import load_model_tokenizer
    except:
        ImportError

import os
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

GSM8K_train_path = file_paths.GSM8K_train
GSM8K_test_path = file_paths.GSM8K_test
qwen_local_path = file_paths.qwen_model_path

MICRO_BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 1
BATCH_SIZE = MICRO_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS
LEARNING_RATE = 1e-5
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
EPOCHS = 1

def print_memory_usage(note=""):
    process = psutil.Process(os.getpid())
    mem_mb = process.memory_info().rss / 1024 / 1024
    print(f"{note}Memory usage: {mem_mb:.2f} MB")

def extract_gsm8k_final_answer(answer_text:str) -> str:
    return answer_text.split("####")[-1].strip()

def generate_gsm8k_test_prompts_answers(test_file_path:Path):
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
    return prompts, answers

def evaluate_vllm(
        vllm_model: LLM,
        reward_fn: Callable[[str,str], dict[str, float]],
        prompts: List[str],
        eval_sampling_params: SamplingParams,
        answers: List[str],
        csv_output_path: Path | None = None,
        jsonl_output_path: Path | None = None
) -> None:
    '''
    Evaluate a language model on a list of prompts,
    compute evaluation metrics, and serialize results to disk.
    '''
    output_lists = []
    outputs = vllm_model.generate(prompts, eval_sampling_params)
    total_count = 0
    format_count = 0
    correct_count = 0
    for idx, output in enumerate(outputs):
        total_count += 1
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

        if (format_reward == 1.0):
            format_count += 1
        if (reward == 1.0):
            correct_count += 1

        output_lists.append(new_dict)
    
    print(f"format accuracy: {format_count/total_count}, answer correct accuracy: {correct_count/total_count}")
    if csv_output_path:
        output_df = pd.DataFrame(output_lists)
        output_df.to_csv(csv_output_path)

def load_policy_into_vllm_instance(policy: PreTrainedModel, llm: LLM):
    state_dict = policy.state_dict()
    engine = llm.llm_engine

    llm_model = engine.model_executor.driver_worker.model_runner.model
    llm_model.load_weights(state_dict.items())

if __name__ == "__main__":
    # Initialize policy (SFT model's weights)
    policy, tokenizer = load_model_tokenizer(model_name=file_paths.qwen_math, local_model_path=file_paths.qwen_model_path, dtype=torch.bfloat16, device=DEVICE)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=LEARNING_RATE)
    optimizer.zero_grad()

    # Initialize vllm (inference and test)
    llm = LLM(model=str(qwen_local_path), dtype=torch.bfloat16, gpu_memory_utilization=0.125,)
    sampling_params = SamplingParams(temperature=1.0, top_p=1.0, max_tokens=768, stop=["</answer>"], include_stop_str_in_output=True)

    # Load SFT training dataset with pd
    train_df = pd.read_parquet(GSM8K_train_path)
    batch_generator = helper_methods.gsm8k_tokenized_batch_generator(input_df=train_df, batch_size=BATCH_SIZE, tokenizer=tokenizer)

    # Validation steps
    load_policy_into_vllm_instance(policy=policy, llm=llm)
    prompts, answers = generate_gsm8k_test_prompts_answers(GSM8K_test_path)
    evaluate_vllm(vllm_model=llm, reward_fn=r1_zero_reward_fn, prompts=prompts, eval_sampling_params=sampling_params, answers=answers, )

    micro_step = 0
    example_count = 0
    gradient_accumulation_account = 0
    for epoch in range(EPOCHS):
        micro_step = 0

        for batch in batch_generator:
            current_batch_size = batch["input_ids"].shape[0]

            for start in range(0, current_batch_size, MICRO_BATCH_SIZE):
                end = start + MICRO_BATCH_SIZE
                micro_step += 1

                micro_input_ids = batch["input_ids"][start:end]
                micro_labels = batch["labels"][start:end]
                micro_response_mask = batch["response_mask"][start:end]

                output = helper_methods.get_response_log_probs(
                    model=policy,
                    input_ids=micro_input_ids.to(DEVICE),
                    labels=micro_labels.to(DEVICE),
                    return_token_entropy=True
                )

                loss, metrics = helper_methods.sft_microbatch_train_step(
                    policy_log_probs=output["log_probs"],
                    response_mask=micro_response_mask.to(DEVICE),
                    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
                )

                if micro_step % GRADIENT_ACCUMULATION_STEPS == 0:
                    torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad()
                
                example_count += MICRO_BATCH_SIZE

                if example_count in [128, 256, 512, 1024] and micro_step % GRADIENT_ACCUMULATION_STEPS == 0:
                    # Validation steps
                    print(f"validation after training {example_count} examples")
                    load_policy_into_vllm_instance(policy=policy, llm=llm)
                    prompts, answers = generate_gsm8k_test_prompts_answers(GSM8K_test_path)
                    evaluate_vllm(vllm_model=llm, reward_fn=r1_zero_reward_fn, prompts=prompts, eval_sampling_params=sampling_params, answers=answers, )

                if example_count >= 1024:
                    break

            if example_count >= 1024:
                break