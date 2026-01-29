import torch

def softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    '''
    Computing the softmax for the input tensor, choose the largest number as C(constant)
    '''
    c = x.max(dim=dim, keepdim=True).values
    exp_v = torch.exp(x - c)
    return exp_v / exp_v.sum(dim=dim, keepdim=True)

def cross_entropy(inputs: torch.Tensor, targets: torch.Tensor):
    '''
    Args:
    inputs (Float[Tensor, "batch_size vocab_size"]): inputs[i][j] is the
        unnormalized logit of jth class for the ith example.
    targets (Int[Tensor, "batch_size"]): Tensor of shape (batch_size,) with the index of the correct class.
        Each value must be between 0 and `num_classes - 1`.

    Returns:
        Float[Tensor, ""]: The average cross-entropy loss across examples.
    '''
    inputs = inputs - inputs.max(dim=-1, keepdim=True).values
    # log_sum_exp = torch.log(torch.exp(inputs).sum(dim=-1))
    # target_logits = inputs[torch.arange(inputs.shape[0]), targets]
    # loss = -target_logits + log_sum_exp
    loss = (-inputs[torch.arange(inputs.shape[0]), targets] + torch.log(torch.exp(inputs).sum(dim=-1)))
    return loss.mean()

def gradient_clipping(parameters, max_l2_norm=1.0, eps=1e-6) -> None:
    '''
    Given a set of parameters, clip their combined gradients to have l2 norm at most max_l2_norm.

    Args:
        parameters (Iterable[torch.nn.Parameter]): collection of trainable parameters.
        max_l2_norm (float): a positive value containing the maximum l2-norm.

    The gradients of the parameters (parameter.grad) should be modified in-place.
    '''
    grads = []
    for param in parameters:
        if param.grad is not None:
            grads.append(param.grad.view(-1))
    if not grads:
        return
    g = torch.cat(grads)
    l2_norm = torch.norm(g, p=2)
    if l2_norm > max_l2_norm:
        scale = max_l2_norm / (l2_norm + eps)
        for param in parameters:
            if param.grad is not None:
                param.grad.mul_(scale)
    return