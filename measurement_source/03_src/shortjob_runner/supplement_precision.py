"""Explicit numerical regimes; old TF32-enabled evidence retains its identity."""
POLICIES = {
    'legacy_cudnn_tf32': ('supplement-v1', True),
    'strict_fp32': ('supplement-v2', False),
}
DEFAULT_POLICY = 'legacy_cudnn_tf32'


def policy_spec(policy):
    if policy not in POLICIES:
        raise ValueError(f'Unknown precision policy: {policy}')
    return POLICIES[policy]


def with_precision(tasks, policy):
    protocol, _ = policy_spec(policy)
    converted = []
    for task in tasks:
        item = {**task, 'protocol': protocol}
        if policy == DEFAULT_POLICY:
            item.pop('precision_policy', None)
        else:
            item['precision_policy'] = policy
        converted.append(item)
    return converted


def campaign_precision(tasks):
    identities = {(t.get('protocol'), t.get('precision_policy', DEFAULT_POLICY)) for t in tasks}
    if len(identities) != 1:
        raise ValueError('A campaign must contain one nonempty protocol/precision regime')
    protocol, policy = identities.pop()
    expected_protocol, _ = policy_spec(policy)
    if protocol != expected_protocol:
        raise ValueError('Task protocol does not match its precision policy')
    return protocol, policy


def configure_precision(policy):
    import torch

    _, allow_cudnn_tf32 = policy_spec(policy)
    # Use one API family consistently for the pinned PyTorch 2.12 runtime.
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = allow_cudnn_tf32
