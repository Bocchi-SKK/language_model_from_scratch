from transformers import PreTrainedTokenizer, PreTrainedModel
from vllm import LLM, SamplingParams
from typing import Callable, List
from pathlib import Path
import pandas as pd
import torch
from einops import rearrange
import json
from typing import Literal

def tokenize_prompt_and_output(prompt_strs:list[str], output_strs:list[str], tokenizer:PreTrainedTokenizer) -> dict:
    input_ids_list = []
    masks = []

    max_prompt_and_output_lens = 0

    for idx, _ in enumerate(prompt_strs):
        prompt_token = tokenizer.encode(prompt_strs[idx])
        output_token = tokenizer.encode(output_strs[idx])
        max_prompt_and_output_lens = max(max_prompt_and_output_lens, (len(prompt_token) + len(output_token)))

    output_length = (max_prompt_and_output_lens - 1)

    for idx, _ in enumerate(prompt_strs):
        prompt_token = tokenizer.encode(prompt_strs[idx])
        output_token = tokenizer.encode(output_strs[idx])

        # Create mask
        output_start = len(prompt_token)
        output_end = len(prompt_token) + len(output_token)
        mask = [0] * output_length
        mask[output_start-1 : output_end-1] = [1] * (output_end - output_start)

        # Combine prompt_token and output_token
        combined = prompt_token + output_token
        pad_id = tokenizer.pad_token_id

        # Truncate if too long
        if len(combined) > max_prompt_and_output_lens:
            # Keep the end
            combined = combined[-max_prompt_and_output_lens:]
        # Pad if too short
        elif len(combined) < max_prompt_and_output_lens:
            # Pad with tokenizer.pad_token_id
            combined = combined + [pad_id] * (max_prompt_and_output_lens - len(combined))

        masks.append(mask)
        input_ids_list.append(combined)

    response_mask = torch.tensor(masks)
    input_ids_and_labels = torch.tensor(input_ids_list)
    input_ids = input_ids_and_labels[:, :-1]
    labels = input_ids_and_labels[:, 1:]

    output = {
        'input_ids_and_labels':input_ids_and_labels,
        'input_ids':input_ids,
        'labels':labels,
        'response_mask':response_mask,
    }

    return output

def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    log_probs = torch.log_softmax(logits, dim=-1)
    prob = torch.softmax(logits, dim=-1)
    entropy = -(log_probs * prob).sum(dim=-1)
    return entropy

def get_response_log_probs(
        model: PreTrainedModel,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        return_token_entropy: bool = False,
) -> dict [str, torch.Tensor]:
    logits = model(input_ids).logits
    log_probs = torch.log_softmax(logits, dim=-1) # shape [batch_size, sequence_length, vocabulary_size]
    log_probs = log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1) # shape [batch_size, sequence_length]
    if return_token_entropy:
        token_entropy = compute_entropy(logits)
        output = {
            'log_probs':log_probs,
            'token_entropy':token_entropy
        }
    else:
        output = {"log_probs" : log_probs}
    return output

def masked_normalize(
        tensor: torch.Tensor,
        mask: torch.Tensor,
        normalize_constant: float,
        dim: int | None = None,
) -> torch.Tensor:
    masked_sum = (tensor * mask).sum(dim=dim)
    maksed_sum_normalized = masked_sum / normalize_constant
    return maksed_sum_normalized

def sft_microbatch_train_step(
        policy_log_probs: torch.Tensor,
        response_mask: torch.Tensor,
        gradient_accumulation_steps: int,
        normalize_constant: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    per_example_log_prob = masked_normalize(policy_log_probs, response_mask, normalize_constant, dim=-1) # negative log-likelihood
    loss = -per_example_log_prob.mean()
    loss = loss / (gradient_accumulation_steps)
    loss.backward()
    metadata = {
        "loss":loss.detach().item(),
        "num_response_tokens":response_mask.sum().item()
    }
    return (loss, metadata)

def log_generations(
        log_path, # .jsonl
        input_prompt,
        response,
        ground_answer,
        reward_information, # including format, answer, and total reward {format:format, answer:answer, total_reward:total_reward}
        average_token_entropy,
        average_response_length,
        average_response_length_for_correct_responses,
        average_response_length_for_incorrect_responses,
):
    output = {
        'prompt': input_prompt,
        'response': response,
        'ground_answer': ground_answer,
        'reward_information': reward_information,
        'average_token_entropy': average_token_entropy,
        'average_response_length': average_response_length,
        'average_response_length_for_correct_responses': average_response_length_for_correct_responses,
        'average_response_length_for_incorrect_responses': average_response_length_for_incorrect_responses
    }
    with open(log_path, 'a') as f:
        f.write(json.dumps(output) + '\n')

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

def extract_gsm8k_final_answer(answer_text:str) -> str:
    return answer_text.split("####")[-1].strip()

def gsm8k_tokenized_batch_generator(input_df, batch_size, tokenizer) -> list[dict]:
    def generate_gsm8k_examples(input_df):
        output_examples = []

        for idx,row in input_df.iterrows():
            question = row["question"]
            answer:str = row["answer"]
            parts = answer.split('\n####')
            think = parts[0]
            gold_answer = parts[-1].strip()
            output_example = {
                'prompts':f'A conversation between User and Assistant. The User asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer. The reasoning process is enclosed within <think> </think> and answer is enclosed within <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>.\nUser: {question}\nAssistant:',
                'targets':f'<think> {think} </think> <answer> {gold_answer} </answer>'
            }
            output_examples.append(output_example)

        return output_examples

    def split_into_batches(input_list, batch_size):
        return [input_list[i:i+batch_size] for i in range(0, len(input_list), batch_size)]

    def generate_gsm8k_batches(input_df, batch_size):
        return(split_into_batches(generate_gsm8k_examples(input_df), batch_size))
    
    batches = generate_gsm8k_batches(input_df, batch_size)
    all_ids_labels_mask = []
    for batch in batches:
        prompts = []
        targets = []
        for prompts_targets in batch:
            prompt = prompts_targets['prompts']
            target = prompts_targets['targets']
            prompts.append(prompt)
            targets.append(target)
        ids_labels_mask = tokenize_prompt_and_output(
            prompt_strs=prompts,
            output_strs=targets,
            tokenizer=tokenizer
        )
        # yield ids_labels_mask
        all_ids_labels_mask.append(ids_labels_mask)
    return all_ids_labels_mask

def load_policy_into_vllm_instance(policy: PreTrainedModel, llm: LLM):
    with torch.no_grad():
        state_dict = policy.state_dict()
        engine = llm.llm_engine

        llm_model = engine.model_executor.driver_worker.model_runner.model
        llm_model.load_weights(state_dict.items())

def evaluate_vllm(
        vllm_model: LLM,
        reward_fn: Callable[[str,str], dict[str, float]],
        prompts: List[str],
        eval_sampling_params: SamplingParams,
        answers: List[str],
        csv_output_path: Path | None = None,
        jsonl_output_path: Path | None = None
):
    '''
    Evaluate a language model on a list of prompts,
    compute evaluation metrics, and serialize results to disk.
    '''
    output_lists = []
    outputs = vllm_model.generate(prompts, eval_sampling_params)
    total_count = 0
    format_count = 0
    answer_count = 0
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
        if (answer_reward == 1.0):
            answer_count += 1
        if (reward == 1.0):
            correct_count += 1

        output_lists.append(new_dict)
    
    print(f"format accuracy: {format_count/total_count}, answer correct accuracy: {answer_count/total_count}")
    if csv_output_path:
        output_df = pd.DataFrame(output_lists)
        output_df.to_csv(csv_output_path)

    accuracy = correct_count/total_count # Final accuracy.
    return accuracy

def compute_group_normalized_reward(
        reward_fn,
        rollout_responses:list[str],
        repeated_ground_truths:list[str],
        group_size:int,
        advantage_eps:float,
        normalize_by_std:bool = True,
):
    raw_rewards = []
    rollout_batch_size = len(rollout_responses)

    for index in range(rollout_batch_size):
        reward = reward_fn(rollout_responses[index], repeated_ground_truths[index])['reward']
        raw_rewards.append(reward)
    
    raw_rewards = torch.tensor(raw_rewards)
    temp_rewards = rearrange(raw_rewards, '(n_prompts_per_rollout_batch group_size) -> n_prompts_per_rollout_batch group_size', group_size=group_size)
    
    means = temp_rewards.mean(dim=-1).unsqueeze(dim=-1)
    stds = temp_rewards.std(dim=-1).unsqueeze(dim=-1)

    if normalize_by_std:
        advantages = (temp_rewards - means) / (stds + advantage_eps)
    else:
        advantages = temp_rewards - means
    
    advantages = rearrange(advantages, 'n_prompts_per_rollout_batch group_size -> (n_prompts_per_rollout_batch group_size)')
    metadata = {
        'normalize_by_std':bool(normalize_by_std),
        'means':means.to('cpu').squeeze(dim=-1).tolist(),
        'stds':stds.to('cpu').squeeze(dim=-1).tolist(),
    }

    return advantages, raw_rewards, metadata

def compute_naive_policy_gradient_loss(
        raw_rewards_or_advantages: torch.Tensor,
        policy_log_probs: torch.Tensor,
) -> torch.Tensor:
    '''
    raw_rewards_or_advantages get shape like (batch_size, 1)
    policy_log_probs shape (batch_size, sequence_length)
    '''
    loss:torch.Tensor = -(raw_rewards_or_advantages * policy_log_probs)
    return loss

def compute_grpo_clip_loss(
        advantages: torch.Tensor,
        policy_log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        cliprange: float = 0.2,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    
    ratio = torch.exp(policy_log_probs - old_log_probs) # input log_prob, loss based on prob, use exp to remove log
    clipped_ratio = torch.clip(ratio, 1-cliprange, 1+cliprange)
    loss = -torch.min(ratio * advantages, clipped_ratio * advantages)

    is_clipped = (ratio != clipped_ratio)
    clipped_fraction = is_clipped.float().mean().item()

    metadata = {
        'clipped_fraction': clipped_fraction,
        'mean_ratio': ratio.mean().item(),
    }

    return loss,metadata

def compute_policy_gradient_loss(
        policy_log_probs: torch.Tensor,
        loss_type: Literal['no_baseline', 'reinforce_with_baseline', 'grpo_clip'],
        raw_rewards: torch.Tensor | None = None,
        advantages: torch.Tensor | None = None,
        old_log_probs: torch.Tensor | None = None,
        cliprange: float | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    metadata = None
    if loss_type == 'no_baseline':
       loss = compute_naive_policy_gradient_loss(raw_rewards_or_advantages=raw_rewards, policy_log_probs=policy_log_probs)
       return loss, metadata
    
    elif loss_type == 'reinforce_with_baseline':
        loss = compute_naive_policy_gradient_loss(raw_rewards_or_advantages=advantages, policy_log_probs=policy_log_probs)
        return loss, metadata
    
    elif loss_type == 'grpo_clip':
        loss, metadata = compute_grpo_clip_loss(
            advantages=advantages,
            policy_log_probs=policy_log_probs,
            old_log_probs=old_log_probs,
            cliprange=cliprange
        )
        return loss, metadata
    
def masked_mean(
        tensor: torch.Tensor,
        mask: torch.Tensor,
        dim: int | None = None,
) -> torch.Tensor:
    # masked_tensor_sum = (tensor * mask).sum(dim=dim)
    # mask_count = mask.sum(dim=dim)
    # masked_tensor_average = masked_tensor_sum/mask_count
    return (tensor * mask).sum(dim=dim) / mask.sum(dim=dim)

def grpo_microbatch_train_step(
        policy_log_probs: torch.Tensor,
        response_mask: torch.Tensor,
        gradient_accumulation_steps: int,
        loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
        raw_rewards: torch.Tensor | None = None,
        advantages: torch.Tensor | None = None,
        old_log_probs: torch.Tensor | None = None,
        cliprange: float | None = None,
)-> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    '''
    Execute a forward-and-backward pass on a microbatch
    Args:
        policy_log_probs (batch_size, sequence_length), per-token log-probabilities from the
        policy being trained.

        response_mask (batch_size, sequence_length), 1 for response tokens, 0 for
        prompt/padding.

        gradient_accumulation_steps Number of microbatches per optimizer step.

        loss_type One of "no_baseline", "reinforce_with_baseline", "grpo_clip".

        raw_rewards Needed when loss_type == "no_baseline"; shape (batch_size, 1).

        advantages Needed when loss_type != "no_baseline"; shape (batch_size, 1).

        old_log_probs Required for GRPO-Clip; shape (batch_size, sequence_length).

        cliprange Clip parameter ϵ for GRPO-Clip.
    Returns:
        tuple[torch.Tensor, dict[str, torch.Tensor]].

            loss scalar tensor. The microbatch loss, adjusted for gradient accumulation. We return
            this so we can log it.

            metadata Dict with metadata from the underlying loss call, and any other statistics you
            might want to log.
    '''
    per_token_loss, metadata = compute_policy_gradient_loss(
        policy_log_probs=policy_log_probs,
        loss_type=loss_type,
        raw_rewards=raw_rewards,
        advantages=advantages,
        old_log_probs=old_log_probs,
        cliprange=cliprange
    )
    loss = masked_mean(per_token_loss, response_mask)
    loss /= gradient_accumulation_steps
    loss.backward()

    return loss, metadata

def grpo_microbatch_train_step_normalization(
        policy_log_probs: torch.Tensor,
        response_mask: torch.Tensor,
        gradient_accumulation_steps: int,
        loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
        raw_rewards: torch.Tensor | None = None,
        advantages: torch.Tensor | None = None,
        old_log_probs: torch.Tensor | None = None,
        cliprange: float | None = None,
        normalize_constant: float | None = None,
)-> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    per_token_loss, metadata = compute_policy_gradient_loss(
        policy_log_probs=policy_log_probs,
        loss_type=loss_type,
        raw_rewards=raw_rewards,
        advantages=advantages,
        old_log_probs=old_log_probs,
        cliprange=cliprange,
    )
    loss = masked_normalize(per_token_loss, response_mask, normalize_constant, dim=-1) # shape (batch_size, )
    loss = loss.mean() # shape (scalar)
    loss /= gradient_accumulation_steps
    loss.backward()

    return loss, metadata