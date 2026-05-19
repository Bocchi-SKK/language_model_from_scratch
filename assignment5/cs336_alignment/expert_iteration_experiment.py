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
import torch
import torch
import json
from datetime import datetime

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

# model, tokenizer = load_model_tokenizer()
# data_directory = Path("/mnt/e/Data/cs336_data/Assignment5")
# qwen_local_path = Path("/mnt/e/Data/cs336_data/Assignment5/model/Qwen2.5-Math-1.5B")
# GSM8K_test_path = Path("/mnt/e/Data/cs336_data/Assignment5/GSM8K/main/test-00000-of-00001.parquet")
# GSM8K_train_path = Path("/mnt/e/Data/cs336_data/Assignment5/GSM8K/main/train-00000-of-00001.parquet")
# csv_output_path = data_directory/ "results/GSM8K_test_results.csv"
# log_path = data_directory/'results/GSM8K_expert_iteration_log.jsonl'

qwen_local_path = file_paths.qwen_model_path
GSM8K_test_path = file_paths.GSM8K_test
GSM8K_train_path = file_paths.GSM8K_train
logs_directory = file_paths.logs_directory
csv_output_path = logs_directory / 'GSM8K_test_results.csv'

MICRO_BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 1
BATCH_SIZE = MICRO_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS
LEARNING_RATE = 1e-5
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
EPOCHS = 1
G = 16
SEED = 42
N_EI_STEPS = 5
NUM_SAMPLES = 512
configs = [
    {'G': 1, 'sft_epochs': 1, 'batch_size': 512},
    {'G': 2, 'sft_epochs': 1, 'batch_size': 512},
    {'G': 1, 'sft_epochs': 2, 'batch_size': 512},
    {'G': 1, 'sft_epochs': 1, 'batch_size': 1024},
    {'G': 2, 'sft_epochs': 1, 'batch_size': 1024},
    {'G': 1, 'sft_epochs': 2, 'batch_size': 1024},
    {'G': 1, 'sft_epochs': 1, 'batch_size': 2048},
    {'G': 2, 'sft_epochs': 1, 'batch_size': 2048},
    {'G': 1, 'sft_epochs': 2, 'batch_size': 2048},
]

def extract_gsm8k_final_answer(answer_text:str) -> str:
    return answer_text.split("####")[-1].strip()

def generate_gsm8k_prompts_answers(file_path:Path):
    # Preparing prompts
    df = pd.read_parquet(file_path)
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

    accuracy = correct_count/total_count
    return accuracy

def load_policy_into_vllm_instance(policy: PreTrainedModel, llm: LLM):
    with torch.no_grad():
        state_dict = policy.state_dict()
        engine = llm.llm_engine

        llm_model = engine.model_executor.driver_worker.model_runner.model
        llm_model.load_weights(state_dict.items())

def save_results_to_json(results, filepath="expert_iteration_results.json"):
    """Save all results to a JSON file"""
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {filepath}")

def load_results_from_json(filepath="expert_iteration_results.json"):
    """Load results from a JSON file"""
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

if __name__ == "__main__":
    # Initialize vllm (inference and test)
    policy, tokenizer = load_model_tokenizer()
    llm = LLM(model=str(qwen_local_path), dtype=torch.bfloat16, gpu_memory_utilization=0.135,)
    
    # Load existing results or start fresh
    all_results = load_results_from_json()
    
    for config in configs:
        G = config['G']
        sft_epochs = config['sft_epochs']
        NUM_SAMPLES = config['batch_size']
        
        config_key = f"G{G}_epochs{sft_epochs}_bs{NUM_SAMPLES}"
        print(f"\n{'='*60}")
        print(f"Starting configuration: {config_key}")
        print(f"{'='*60}\n")
        
        # Skip if already completed
        if config_key in all_results:
            print(f"Configuration {config_key} already completed, skipping...\n")
            continue
        
        # Reset policy to base model for each config
        policy = AutoModelForCausalLM.from_pretrained(qwen_local_path, dtype=torch.bfloat16).to(DEVICE)
        tokenizer = AutoTokenizer.from_pretrained(qwen_local_path)
        optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-5)
        optimizer.zero_grad()
        
        # Update sampling params with new G
        sampling_params = SamplingParams(
            temperature=1.0, top_p=1.0, max_tokens=768, 
            min_tokens=4, n=G, stop=["</answer>"], 
            include_stop_str_in_output=True, seed=SEED
        )

        eval_sampling_params = SamplingParams(
            temperature=1.0, top_p=1.0, max_tokens=768, 
            min_tokens=4, n=1, stop=["</answer>"], 
            include_stop_str_in_output=True, seed=SEED
        )
        
        # Load data
        prompts_test, answers_test = generate_gsm8k_prompts_answers(GSM8K_test_path)
        prompts_train, answers_train = generate_gsm8k_prompts_answers(GSM8K_train_path)
        
        # Initialize tracking lists for this configuration
        accuracy_curve = []
        entropy_curve = []
        
        for ei_step in range(N_EI_STEPS):
            print(f"\n--- EI Step {ei_step + 1}/{N_EI_STEPS} ---")
            
            load_policy_into_vllm_instance(policy=policy, llm=llm)
            
            # Generate expert responses
            print(f"Generating {NUM_SAMPLES} samples with G={G} rollouts...")
            outputs = llm.generate(prompts_train[:NUM_SAMPLES], sampling_params)
            
            expert_prompts = []
            expert_responses = []
            
            for idx, output in enumerate(outputs):
                prompt = output.prompt
                for response in output.outputs:
                    response_text = response.text
                    ground_truth = extract_gsm8k_final_answer(answers_train[idx])
                    score = r1_zero_reward_fn('<think>' + response_text, ground_truth)
                    reward = score["reward"]
                    
                    if reward == 1.0:
                        expert_prompts.append(prompt)
                        expert_responses.append(response_text)
            
            print(f"Generated {len(expert_prompts)} correct samples out of {NUM_SAMPLES * G} total")
            
            if len(expert_prompts) == 0:
                print("Warning: No correct samples generated! Skipping SFT training.")
                # Still evaluate to track accuracy
                load_policy_into_vllm_instance(policy=policy, llm=llm)
                accuracy = evaluate_vllm(
                    llm, r1_zero_reward_fn, prompts_test, eval_sampling_params, answers_test
                )
                accuracy_curve.append(accuracy)
                entropy_curve.append(0.0)
                
                # Save intermediate results
                all_results[config_key] = {
                    'config': config,
                    'accuracy_curve': accuracy_curve,
                    'entropy_curve': entropy_curve,
                    'completed_steps': ei_step + 1
                }
                save_results_to_json(all_results)
                continue
            
            # Tokenize expert data
            ids_labels_mask = helper_methods.tokenize_prompt_and_output(
                prompt_strs=expert_prompts,
                output_strs=expert_responses,
                tokenizer=tokenizer
            )
            
            input_ids = ids_labels_mask['input_ids']
            labels = ids_labels_mask['labels']
            response_mask = ids_labels_mask['response_mask']
            
            # SFT training with multiple epochs
            epoch_entropies = []
            
            for epoch in range(sft_epochs):
                print(f"  Training epoch {epoch + 1}/{sft_epochs}")
                batch_entropies = []
                
                expert_examples_count = input_ids.shape[0]
                micro_step = 0
                
                for start in range(0, expert_examples_count, MICRO_BATCH_SIZE):
                    end = min(start + MICRO_BATCH_SIZE, expert_examples_count)
                    micro_input_ids = input_ids[start:end].to(DEVICE)
                    micro_labels = labels[start:end].to(DEVICE)
                    micro_response_mask = response_mask[start:end].to(DEVICE)
                    
                    # Use torch.no_grad() if you are ONLY evaluating, but here we are training.
                    # However, ensure we don't keep old graphs.
                    
                    output = helper_methods.get_response_log_probs(
                        model=policy,
                        input_ids=micro_input_ids,
                        labels=micro_labels,
                        return_token_entropy=True
                    )
                    
                    # Compute average entropy for this batch
                    token_entropy = output['token_entropy']
                    # .item() is CRITICAL to prevent memory leaks
                    avg_batch_entropy = (token_entropy * micro_response_mask).sum() / micro_response_mask.sum()
                    batch_entropies.append(avg_batch_entropy.item()) 
                    
                    loss, metrics = helper_methods.sft_microbatch_train_step(
                        policy_log_probs=output["log_probs"],
                        response_mask=micro_response_mask,
                        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
                    )
                    
                    micro_step += 1
                    if micro_step % GRADIENT_ACCUMULATION_STEPS == 0:
                        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
                        optimizer.step()
                        optimizer.zero_grad()
                    
                    # Explicitly delete temporary tensors to free graph
                    del loss, output, token_entropy, avg_batch_entropy
                    torch.cuda.empty_cache() 

                avg_epoch_entropy = sum(batch_entropies) / len(batch_entropies)
                epoch_entropies.append(avg_epoch_entropy)
                print(f"  Epoch {epoch + 1} average entropy: {avg_epoch_entropy:.4f}")
                
                # Optional: Clear cache after each epoch
                torch.cuda.empty_cache()
            
            # Average entropy across all epochs for this EI step
            avg_entropy = sum(epoch_entropies) / len(epoch_entropies)
            entropy_curve.append(avg_entropy)
            
            # Evaluate after SFT training completes
            print(f"Evaluating after EI step {ei_step + 1}...")

            load_policy_into_vllm_instance(policy=policy, llm=llm)
            accuracy = evaluate_vllm(
                llm, r1_zero_reward_fn, prompts_test, eval_sampling_params, answers_test
            )
            
            accuracy_curve.append(accuracy)
            
            print(f"EI Step {ei_step + 1}: Accuracy={accuracy:.4f}, Entropy={avg_entropy:.4f}")
            
            # Clear cache after each EI step
            torch.cuda.empty_cache()
            
            # Save intermediate results after each EI step
            all_results[config_key] = {
                'config': config,
                'accuracy_curve': accuracy_curve,
                'entropy_curve': entropy_curve,
                'completed_steps': ei_step + 1
            }
            save_results_to_json(all_results)
        
        print(f"\nConfiguration {config_key} completed!")
        print(f"Final accuracy: {accuracy_curve[-1]:.4f}")
        print(f"Accuracy curve: {accuracy_curve}")
        print(f"Entropy curve: {entropy_curve}\n")
        
        # Final cleanup for this configuration
        del policy, optimizer
        torch.cuda.empty_cache()
    
    print("\nAll configurations completed!")
    print("Results saved to expert_iteration_results.json")