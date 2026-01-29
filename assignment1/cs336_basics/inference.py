import torch
try:
    from nn_utils import softmax
except:
    try:
        from cs336_basics.nn_utils import softmax
    except:
        ImportError

def top_p_sampling(probs, top_p=0.9):
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    accume_p = 0.0
    temp_probs = []
    i = 0
    while (accume_p < top_p):
        value = sorted_probs[i].item()
        temp_probs.append(value)
        accume_p += value
        i += 1
    temp_probs = torch.tensor(temp_probs)
    index = torch.multinomial(temp_probs, num_samples=1).item()
    next_token_id = sorted_indices[index].item()
    return next_token_id

def generate_text(model, tokenizer, prompt, context_length, temperature=0.15, top_p=0.9):
    model.eval()
    with torch.no_grad():
        max_context_length = model.context_length
        if (context_length > max_context_length):
            context_length = max_context_length
        pad_token_id = tokenizer.encode('<|endoftext|>')[0] # Here use '<|endoftext|>' as pad token
        prompt = tokenizer.encode(prompt)
        prompt_length = len(prompt)
        while len(prompt) < context_length:
            prompt.append(pad_token_id)

        input_tensor = torch.tensor(prompt)

        # Inference loop
        while (prompt_length < context_length):
            logits = model(input_tensor)
            probs = softmax(logits/temperature)  # shape: [context_length, vocab_size]

            # Get probabilities for the next token position
            next_token_probs = probs[prompt_length - 1]

            # Sample the next token id from the probability distribution
            # next_token_id = torch.multinomial(next_token_probs, num_samples=1).item()
            next_token_id = top_p_sampling(probs=next_token_probs, top_p=top_p)

            input_tensor[prompt_length] = next_token_id
            if (next_token_id == pad_token_id):
                break
            prompt_length += 1

        prompt = input_tensor.tolist()
        output = tokenizer.decode(prompt).replace('<|endoftext|>','')
        return output