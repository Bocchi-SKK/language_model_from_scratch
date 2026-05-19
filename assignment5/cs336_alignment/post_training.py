from typing import Literal
import torch
from pathlib import Path
from vllm import LLM, SamplingParams
from transformers import AutoModelForCausalLM, AutoTokenizer
import helper_methods
from drgrpo_grader import r1_zero_reward_fn
import json
from torch.optim.lr_scheduler import CosineAnnealingLR
import pandas as pd

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

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# data_directory = Path('/mnt/e/Data/cs336_data/Assignment5')
# qwen_model_path = data_directory/'model/Qwen2.5-Math-1.5B'
# gsm8k_train_path = data_directory/'GSM8K/main/train-00000-of-00001.parquet'
# gsm8k_test_path = data_directory/'GSM8K/main/test-00000-of-00001.parquet'

qwen_model_path = file_paths.qwen_model_path
gsm8k_train_path = file_paths.GSM8K_train
gsm8k_test_path = file_paths.GSM8K_test

n_grpo_steps: int = 200 # On-policy
# n_grpo_steps = 100 # Off-policy

sft_learning_rate: float = 5e-5
learning_rate: float = 2e-5
learning_rate_end: float = 1e-6
advantage_eps: float = 1e-6
rollout_batch_size: int = 256
group_size: int = 8
sampling_temperature: float = 1.0
sampling_min_tokens: int = 4
sampling_max_tokens: int = 1024

# epochs_per_rollout_batch: int = 1 # On-policy
# train_batch_size: int = 256 # On-policy
epochs_per_rollout_batch: int = 2 # Off-policy
train_batch_size: int = rollout_batch_size # Off-policy

gradient_clip_value: float = 1.0
cliprange:float = 0.2
gradient_accumulation_steps: int = 128
gpu_memory_utilization: float = 0.130
loss_type: Literal['no_baseline', 'reinforce_with_baseline', 'grpo_clip'] = 'grpo_clip'
patience: int = 5 # set 10 turn off
use_std_normalization: bool = True
use_masked_normalize: bool = True

assert train_batch_size % gradient_accumulation_steps == 0,("train_batch_size must be divisible by gradient_accumulation_steps")
micro_train_batch_size = train_batch_size // gradient_accumulation_steps
assert rollout_batch_size % group_size == 0, ("rollout_batch_size must be divisible by group_size")
n_prompts_per_rollout_batch = rollout_batch_size // group_size
assert train_batch_size >= group_size, ("train_batch_size must be greater than or equal to group_size")
n_microbatches_per_rollout_batch = rollout_batch_size // micro_train_batch_size

num_sft_examples:int = 1024 * 3
sft_micro_batch_size:int = 2
num_sft_batch:int = num_sft_examples // sft_micro_batch_size

sampling_params = SamplingParams(
    temperature=sampling_temperature,top_p=1.0,
    max_tokens=sampling_max_tokens, min_tokens=sampling_min_tokens,
    n=group_size, stop=["</answer>"], include_stop_str_in_output=True)

eval_sampling_params = SamplingParams(
    temperature=sampling_temperature, top_p=1.0,
    max_tokens=sampling_max_tokens, min_tokens=sampling_min_tokens,
    n=1, stop=["</answer>"], include_stop_str_in_output=True)

if __name__ == '__main__':
    
    train_prompts, train_answers = helper_methods.generate_gsm8k_prompts_answers(gsm8k_train_path) # For RL
    test_prompts, test_answers = helper_methods.generate_gsm8k_prompts_answers(gsm8k_test_path) # For test

    policy_model, tokenizer = load_model_tokenizer()
    # tokenizer = AutoTokenizer.from_pretrained(qwen_model_path)
    # policy_model = AutoModelForCausalLM.from_pretrained(qwen_model_path, dtype=torch.bfloat16).to(DEVICE)

    training_logs = []

    # SFHF for model to generate output pass the format test.
    sft_train_dataset_df = pd.read_parquet(gsm8k_train_path)
    sft_ids_labels_mask:list[dict] = helper_methods.gsm8k_tokenized_batch_generator(sft_train_dataset_df, batch_size=sft_micro_batch_size, tokenizer=tokenizer)
    sft_indices = torch.randperm(len(sft_ids_labels_mask))[:num_sft_batch].tolist()
    sft_optimizer = torch.optim.AdamW(policy_model.parameters(), lr=sft_learning_rate)

    # inference_model for RL
    inference_model = LLM(model=str(qwen_model_path), dtype=torch.bfloat16, gpu_memory_utilization=gpu_memory_utilization, )
    helper_methods.load_policy_into_vllm_instance(policy_model, inference_model)

    accuracy = helper_methods.evaluate_vllm(
        vllm_model=inference_model,
        reward_fn=r1_zero_reward_fn,
        prompts=test_prompts,
        eval_sampling_params=eval_sampling_params,
        answers=test_answers
    )

    log_entry = {
        "train_step": 'zero_shot',
        "accuracy": accuracy,
        "learning_rate": sft_learning_rate,
        'loss_type':loss_type,
        'learning_rate':learning_rate,
        'learning_rate_end':learning_rate_end,
        'epochs_per_rollout_batch':epochs_per_rollout_batch,
    }

    training_logs.append(log_entry)

    for step, i in enumerate(sft_indices):
        input_ids:torch.Tensor = sft_ids_labels_mask[i]['input_ids'].to(DEVICE)
        labels:torch.Tensor = sft_ids_labels_mask[i]['labels'].to(DEVICE)
        response_mask:torch.Tensor = sft_ids_labels_mask[i]['response_mask'].to(DEVICE)
        
        log_probs = helper_methods.get_response_log_probs(
            model=policy_model,
            input_ids=input_ids,
            labels=labels,
            return_token_entropy=False
        )['log_probs']

        loss, metadata = helper_methods.sft_microbatch_train_step(
            policy_log_probs=log_probs,
            response_mask=response_mask,
            gradient_accumulation_steps=gradient_accumulation_steps
        )

        if (step+1) % gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(policy_model.parameters(), 1.0)
            sft_optimizer.step()
            sft_optimizer.zero_grad()

        del input_ids, labels, response_mask

    # End SFT and clean the VRAM for RL
    del sft_optimizer, sft_indices, sft_ids_labels_mask, sft_train_dataset_df
    torch.cuda.empty_cache()

    # optimizer and shceduler for RL
    rl_optimizer = torch.optim.AdamW(policy_model.parameters(), lr=learning_rate, weight_decay=0.0, betas=(0.9, 0.95))
    total_steps = n_grpo_steps * epochs_per_rollout_batch
    rl_scheduler = CosineAnnealingLR(rl_optimizer, T_max=total_steps, eta_min=learning_rate_end)

    helper_methods.load_policy_into_vllm_instance(policy_model, inference_model)
    accuracy = helper_methods.evaluate_vllm(
        vllm_model=inference_model,
        reward_fn=r1_zero_reward_fn,
        prompts=test_prompts,
        eval_sampling_params=eval_sampling_params,
        answers=test_answers
    )

    log_entry = {
        "train_step": 0,
        "accuracy": accuracy,
        "learning_rate": sft_learning_rate,
    }
    training_logs.append(log_entry)

    patience_count = 0
    highest_accuracy = 0

    rl_indices = torch.randperm(len(train_prompts))
    num_one_step_example = rollout_batch_size // group_size
    num_examples = len(train_prompts)
    for train_step in range(n_grpo_steps):
        if patience_count >= patience:
            break

        bias = train_step * num_one_step_example
        indices = []
        for k in range(num_one_step_example):
            indices.append(rl_indices[(bias+k)%num_examples])

        # indices = torch.randperm(len(train_prompts))[:n_prompts_per_rollout_batch].tolist()
        
        selected_prompts = [train_prompts[i] for i in indices]
        selected_answers = [train_answers[i] for i in indices]

        # Run inference to generate responses
        outputs = inference_model.generate(selected_prompts, sampling_params)
        prompts = []
        responses = []
        gold_answers = []
        rewards = []
        for idx, output in enumerate(outputs):
            prompt = output.prompt
            for response in output.outputs:
                response_text = response.text
                prompts.append(prompt)
                responses.append(response_text)
                gold_answers.append(helper_methods.extract_gsm8k_final_answer(selected_answers[idx]))

        ids_labels_masks = helper_methods.tokenize_prompt_and_output(prompts, responses, tokenizer)

        # old_log_probs are not used to train the model, only used to calculate the ratio.
        # Used for GRPO 
        old_log_probs_list:list[torch.Tensor] = []
        if loss_type == 'grpo_clip':
            for gradient_accumulation_step in range(gradient_accumulation_steps):
                start_idx = gradient_accumulation_step * micro_train_batch_size
                end_idx = start_idx + micro_train_batch_size
                input_ids = ids_labels_masks['input_ids'][start_idx:end_idx].to(DEVICE)
                labels = ids_labels_masks['labels'][start_idx:end_idx].to(DEVICE)
                with torch.inference_mode():
                    log_probs_and_entropy = helper_methods.get_response_log_probs(
                        model=policy_model,
                        input_ids=input_ids,
                        labels=labels,
                    )
                    old_log_probs:torch.Tensor = log_probs_and_entropy['log_probs'].detach().to('cpu')
                old_log_probs_list.append(old_log_probs)
                del old_log_probs, input_ids, labels

        advantages, raw_rewards, metadata = helper_methods.compute_group_normalized_reward(
            reward_fn=r1_zero_reward_fn,
            rollout_responses=responses,
            repeated_ground_truths=gold_answers,
            group_size=group_size,
            advantage_eps=advantage_eps,
            normalize_by_std=use_std_normalization
        )

        advantages = advantages.unsqueeze(dim=-1)
        raw_rewards = raw_rewards.unsqueeze(dim=-1)

        for train_epoch in range(epochs_per_rollout_batch):
            for gradient_accumulation_step in range(gradient_accumulation_steps):
                start_idx = gradient_accumulation_step * micro_train_batch_size
                end_idx = start_idx + micro_train_batch_size
                micro_input_ids_and_labels = ids_labels_masks['input_ids_and_labels'][start_idx:end_idx].to(DEVICE)
                micro_input_ids = micro_input_ids_and_labels[:, :-1]
                micro_labels = micro_input_ids_and_labels[:, 1:]
                current_log_probs = helper_methods.get_response_log_probs(
                    model=policy_model,
                    input_ids=micro_input_ids,
                    labels=micro_labels,
                    # return_token_entropy=True
                    )['log_probs']
                
                micro_advntages = None
                micro_raw_rewards = None
                micro_old_log_probs = None
                
                micro_mask = ids_labels_masks['response_mask'][start_idx:end_idx].to(DEVICE)
                if loss_type == 'grpo_clip':
                    micro_old_log_probs = old_log_probs_list[gradient_accumulation_step].to(DEVICE)
                if loss_type == 'no_baseline':
                    micro_raw_rewards = raw_rewards[start_idx:end_idx].to(DEVICE)
                if loss_type != 'no_baseline':
                    micro_advntages = advantages[start_idx:end_idx].to(DEVICE)
                
                if not use_masked_normalize:
                    helper_methods.grpo_microbatch_train_step(
                        policy_log_probs=current_log_probs,
                        response_mask=micro_mask,
                        gradient_accumulation_steps=gradient_accumulation_steps,
                        loss_type=loss_type,
                        raw_rewards=micro_raw_rewards,
                        advantages=micro_advntages,
                        old_log_probs=micro_old_log_probs,
                        cliprange=cliprange,
                    )

                elif use_masked_normalize:
                    helper_methods.grpo_microbatch_train_step_normalization(
                        policy_log_probs=current_log_probs,
                        response_mask=micro_mask,
                        gradient_accumulation_steps=gradient_accumulation_steps,
                        loss_type=loss_type,
                        raw_rewards=micro_raw_rewards,
                        advantages=micro_advntages,
                        old_log_probs=micro_old_log_probs,
                        cliprange=cliprange,
                        normalize_constant=sampling_max_tokens
                    )

                del micro_input_ids, micro_labels, micro_mask, micro_raw_rewards, micro_advntages, micro_old_log_probs

            torch.nn.utils.clip_grad_norm_(policy_model.parameters(), max_norm=gradient_clip_value)
            rl_optimizer.step()
            rl_scheduler.step()
            rl_optimizer.zero_grad()
            torch.cuda.empty_cache()
        
        helper_methods.load_policy_into_vllm_instance(policy=policy_model, llm=inference_model)
        if (train_step+1) % 10 == 0:
            accuracy = helper_methods.evaluate_vllm(
                vllm_model=inference_model,
                reward_fn=r1_zero_reward_fn,
                prompts=test_prompts,
                eval_sampling_params=eval_sampling_params,
                answers=test_answers
            )
            print(f'train_step:{train_step+1} get accuracy:{accuracy}')
            log_entry = {
                "train_step": train_step+1,
                "accuracy": accuracy,
                "learning_rate": rl_optimizer.param_groups[0]['lr'],
            }
            training_logs.append(log_entry)
            with open(f'RL_{loss_type}_fine_tunning.json','w') as f:
                json.dump(training_logs, f, indent=4)
            
            # For early stop
            if highest_accuracy > accuracy:
                patience_count += 1
                if patience_count >= patience:
                    break
            else:
                patience_count = 0
                highest_accuracy = accuracy