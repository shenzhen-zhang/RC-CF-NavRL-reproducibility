"""Deterministic auxiliary initialization for new controlled training runs."""
from __future__ import annotations
import hashlib
import torch
from torch import nn

@torch.no_grad()
def _orthogonal_from_cpu_generator(tensor, generator, gain=.01):
    """Equivalent QR orthogonalization without newer orthogonal_(generator=).

    Isaac's older Torch versions do not accept generator in nn.init.orthogonal_.
    All randomness is still supplied by the independent CPU generator.
    """
    if tensor.ndim < 2:
        raise ValueError('Orthogonal initialization needs at least two dimensions')
    rows = tensor.shape[0]
    cols = tensor.numel() // rows
    matrix = torch.randn((rows, cols), dtype=tensor.dtype, device='cpu', generator=generator)
    if rows < cols:
        matrix = matrix.t()
    q, r = torch.linalg.qr(matrix, mode='reduced')
    q *= torch.diag(r).sign()
    if rows < cols:
        q = q.t()
    tensor.copy_(q.reshape_as(tensor) * gain)
    return tensor

@torch.no_grad()
def initialize_reviewer_run(policy, seed: int):
    """Preserve loaded nominal tensors; reset all new safety training state.

    A separate CPU generator per named layer makes common critic hidden
    tensors identical when the output head has 1 versus 32 units. This avoids
    shape-dependent consumption of the global random stream.
    """
    policy.initialize_safety_feature_extractor_from_task()
    for name, module in policy.safety_critic.named_modules():
        if isinstance(module, nn.Linear):
            layer_seed = int.from_bytes(hashlib.sha256(f'{seed}:safety_critic:{name}'.encode()).digest()[:8], 'little') % (2**63 - 1)
            generator = torch.Generator(device='cpu').manual_seed(layer_seed)
            weight = torch.empty(module.weight.shape, dtype=torch.float32)
            _orthogonal_from_cpu_generator(weight, generator=generator, gain=.01)
            module.weight.copy_(weight.to(module.weight))
            module.bias.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.weight.fill_(1)
            module.bias.zero_()
    policy.safety_critic.reset_output_prior(float(policy.cfg.method.initial_safety_probability))
    policy.target_safety_critic.load_state_dict(policy.safety_critic.state_dict())
    if policy.safety_feature_extractor is not None:
        policy.target_safety_feature_extractor.load_state_dict(policy.safety_feature_extractor.state_dict())
    policy.actor_head.reset_safety_adapter(float(policy.cfg.method.actor_adapter.initial_gate_bias))
    policy.safety_lambda.fill_(float(policy.cfg.method.initial_safety_lambda))
    policy.training_update_count.zero_()
    policy.sync_training_schedule()
    return {key: hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
            for key, value in policy.safety_critic.state_dict().items()}
