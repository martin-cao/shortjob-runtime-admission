from __future__ import annotations

from math import sqrt
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from shortjob_runner.types import WorkloadSpec


class StepWorkload:
    spec: WorkloadSpec

    # Optional, explicitly-seeded generator for request-input refreshes. Left
    # as None on the timing path (run_shortjob_isolated), where refreshes draw
    # from the global RNG exactly as torch.randn_like would. The correctness
    # gate seeds it (see seed_refresh) so that the eager reference and the graph
    # candidate observe an *identical* refreshed-input sequence, making the
    # input-copy comparison a true like-for-like correctness check rather than
    # comparing outputs computed from different random inputs.
    _refresh_generator: torch.Generator | None = None

    def step(self) -> None:
        raise NotImplementedError

    def enable_compile(self, mode: str | None = None) -> None:
        raise NotImplementedError

    def graph_capture_feasible(self) -> tuple[bool, str | None]:
        if not self.spec.supports_cuda_graph:
            return False, "workload_declares_cuda_graph_ineligible"
        return True, None

    def input_copy_feasible(self) -> tuple[bool, str | None]:
        return True, None

    def seed_refresh(self, seed: int, device: torch.device) -> None:
        """Make subsequent refresh_request_inputs() draws reproducible.

        Used only by the correctness gate so that two independently-constructed
        workloads (the eager reference and the action candidate) refresh their
        request tensors with the *same* values regardless of any RNG the action
        path consumes internally.
        """
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)
        self._refresh_generator = generator

    def _randn_like(self, tensor: torch.Tensor) -> torch.Tensor:
        return torch.randn(
            tensor.shape, dtype=tensor.dtype, device=tensor.device, generator=self._refresh_generator
        )

    def _randint_like(self, tensor: torch.Tensor, low: int, high: int) -> torch.Tensor:
        return torch.randint(
            low, high, tensor.shape, dtype=tensor.dtype, device=tensor.device, generator=self._refresh_generator
        )

    def refresh_request_inputs(self) -> int:
        """Mutate request tensors in-place and return approximate copied bytes."""
        return 0

    def step_micro_batch(self, chunks: int) -> None:
        self.step()

    def correctness_state(self) -> dict[str, torch.Tensor]:
        raise NotImplementedError


def _last_output_state(owner: Any) -> dict[str, torch.Tensor]:
    output = getattr(owner, "last_output", None)
    if output is None:
        raise RuntimeError("workload has no last_output; run at least one step before checking correctness")
    return {"output": output.detach()}


def _training_state(model: nn.Module, loss: torch.Tensor | None) -> dict[str, torch.Tensor]:
    if loss is None:
        raise RuntimeError("training workload has no last_loss; run at least one step before checking correctness")
    state = {"loss": loss.detach()}
    for name, param in model.named_parameters():
        normalized_name = name.removeprefix("_orig_mod.")
        state[f"param:{normalized_name}"] = param.detach()
    return state


class SmallCNN(nn.Module):
    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 4 * 4, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))


class SmallMLP(nn.Module):
    def __init__(self, input_dim: int = 128, hidden_dim: int = 256, num_classes: int = 10) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TinyPrefillBlock(nn.Module):
    def __init__(self, hidden_dim: int = 64, mlp_dim: int = 128) -> None:
        super().__init__()
        self.qkv = nn.Linear(hidden_dim, hidden_dim * 3)
        self.proj = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, hidden_dim),
        )
        self.hidden_dim = hidden_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        scores = torch.matmul(q, k.transpose(-2, -1)) / sqrt(self.hidden_dim)
        attn = F.softmax(scores, dim=-1)
        x = x + self.proj(torch.matmul(attn, v))
        x = self.norm(x)
        return x + self.mlp(x)


class TinyDecodeBlock(nn.Module):
    def __init__(self, hidden_dim: int = 64, mlp_dim: int = 128) -> None:
        super().__init__()
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, hidden_dim),
        )
        self.hidden_dim = hidden_dim

    def forward(self, token: torch.Tensor, key_cache: torch.Tensor, value_cache: torch.Tensor) -> torch.Tensor:
        q = self.q_proj(token)
        scores = torch.matmul(q, key_cache.transpose(-2, -1)) / sqrt(self.hidden_dim)
        attn = F.softmax(scores, dim=-1)
        x = token + self.out_proj(torch.matmul(attn, value_cache))
        x = self.norm(x)
        return x + self.mlp(x)


class FixedShapeInferSmall(StepWorkload):
    spec = WorkloadSpec(
        workload_id="fixed_shape_infer_small",
        task_family="inference",
        workload_class="small_realistic_proxy",
        shape_stability="stable",
        application="small CV proxy",
        bottleneck_hypothesis="small convolution launch overhead",
    )

    def __init__(self, batch_size: int, device: torch.device) -> None:
        self.model: nn.Module = SmallCNN().to(device).eval()
        self.x = torch.randn(batch_size, 3, 32, 32, device=device)
        self.last_output: torch.Tensor | None = None

    @torch.no_grad()
    def step(self) -> None:
        self.last_output = self.model(self.x)

    def enable_compile(self, mode: str | None = None) -> None:
        self.model = torch.compile(self.model, mode=mode)

    def refresh_request_inputs(self) -> int:
        self.x.copy_(self._randn_like(self.x))
        return self.x.numel() * self.x.element_size()

    def step_micro_batch(self, chunks: int) -> None:
        outputs = [self.model(part) for part in self.x.chunk(max(1, chunks), dim=0)]
        self.last_output = torch.cat(outputs, dim=0)

    def correctness_state(self) -> dict[str, torch.Tensor]:
        return _last_output_state(self)


class ShortTrainSmall(StepWorkload):
    spec = WorkloadSpec(
        workload_id="short_train_small",
        task_family="training",
        workload_class="small_realistic_proxy",
        shape_stability="mostly_stable",
        application="small training step",
        bottleneck_hypothesis="backward and optimizer launch sequence",
        supports_training=True,
        supports_mixed_precision=False,
    )

    def __init__(self, batch_size: int, device: torch.device) -> None:
        self.model: nn.Module = SmallMLP().to(device).train()
        self.x = torch.randn(batch_size, 128, device=device)
        self.y = torch.randint(0, 10, (batch_size,), device=device)
        self.lr = 1e-3
        self.last_loss: torch.Tensor | None = None

    def step(self) -> None:
        for param in self.model.parameters():
            param.grad = None
        loss = F.cross_entropy(self.model(self.x), self.y)
        # Keep only a detached scalar for correctness checking. Retaining the
        # graph-attached ``loss`` across iterations keeps a prior-iteration
        # AccumulateGrad node alive on the graph-capture warm-up side stream,
        # which triggers the autograd stream-mismatch warning during CUDA graph
        # capture. ``_training_state`` already detaches, so this is correctness-
        # and timing-neutral for the eager path. See the rerun note in
        # scripts/run_shorttrain_graphfix.sh.
        self.last_loss = loss.detach()
        loss.backward()
        with torch.no_grad():
            for param in self.model.parameters():
                if param.grad is not None:
                    param.add_(param.grad, alpha=-self.lr)

    def enable_compile(self, mode: str | None = None) -> None:
        self.model = torch.compile(self.model, mode=mode)

    def refresh_request_inputs(self) -> int:
        self.x.copy_(self._randn_like(self.x))
        self.y.copy_(self._randint_like(self.y, 0, 10))
        return self.x.numel() * self.x.element_size() + self.y.numel() * self.y.element_size()

    def correctness_state(self) -> dict[str, torch.Tensor]:
        return _training_state(self.model, self.last_loss)


class DynamicShapeInferSmall(StepWorkload):
    spec = WorkloadSpec(
        workload_id="dynamic_shape_infer_small",
        task_family="inference",
        workload_class="controlled_mechanism",
        shape_stability="unstable",
        application="dynamic-shape boundary",
        bottleneck_hypothesis="shape churn and graph ineligibility",
        supports_cuda_graph=False,
    )

    def __init__(self, batch_size: int, device: torch.device) -> None:
        self.model: nn.Module = SmallMLP(input_dim=128).to(device).eval()
        self.device = device
        self.batch_sizes = [max(1, batch_size // 2), batch_size, max(1, batch_size + batch_size // 2)]
        self.cursor = 0
        self.last_output: torch.Tensor | None = None

    @torch.no_grad()
    def step(self) -> None:
        current_batch = self.batch_sizes[self.cursor % len(self.batch_sizes)]
        self.cursor += 1
        self.last_output = self.model(torch.randn(current_batch, 128, device=self.device))

    def enable_compile(self, mode: str | None = None) -> None:
        self.model = torch.compile(self.model, dynamic=True)

    def graph_capture_feasible(self) -> tuple[bool, str | None]:
        return False, "unstable_shape_not_graph_capturable"

    def correctness_state(self) -> dict[str, torch.Tensor]:
        return _last_output_state(self)


class PrefillToy(StepWorkload):
    spec = WorkloadSpec(
        workload_id="prefill_toy",
        task_family="inference",
        workload_class="small_realistic_proxy",
        shape_stability="stable",
        application="attention prefill proxy",
        bottleneck_hypothesis="attention matmul and normalization launch mix",
    )

    def __init__(self, batch_size: int, device: torch.device) -> None:
        self.model: nn.Module = TinyPrefillBlock().to(device).eval()
        self.x = torch.randn(batch_size, 32, 64, device=device)
        self.last_output: torch.Tensor | None = None

    @torch.no_grad()
    def step(self) -> None:
        self.last_output = self.model(self.x)

    def enable_compile(self, mode: str | None = None) -> None:
        self.model = torch.compile(self.model, mode=mode)

    def refresh_request_inputs(self) -> int:
        self.x.copy_(self._randn_like(self.x))
        return self.x.numel() * self.x.element_size()

    def correctness_state(self) -> dict[str, torch.Tensor]:
        return _last_output_state(self)


class DecodeToy(StepWorkload):
    spec = WorkloadSpec(
        workload_id="decode_toy",
        task_family="inference",
        workload_class="small_realistic_proxy",
        shape_stability="stable",
        application="decode-step proxy",
        bottleneck_hypothesis="single-token attention launch overhead",
    )

    def __init__(self, batch_size: int, device: torch.device) -> None:
        self.model: nn.Module = TinyDecodeBlock().to(device).eval()
        self.token = torch.randn(batch_size, 1, 64, device=device)
        self.key_cache = torch.randn(batch_size, 64, 64, device=device)
        self.value_cache = torch.randn(batch_size, 64, 64, device=device)
        self.last_output: torch.Tensor | None = None

    @torch.no_grad()
    def step(self) -> None:
        self.last_output = self.model(self.token, self.key_cache, self.value_cache)

    def enable_compile(self, mode: str | None = None) -> None:
        self.model = torch.compile(self.model, mode=mode)

    def refresh_request_inputs(self) -> int:
        self.token.copy_(self._randn_like(self.token))
        bytes_copied = self.token.numel() * self.token.element_size()
        return bytes_copied

    def correctness_state(self) -> dict[str, torch.Tensor]:
        return _last_output_state(self)


class ConvOnlineModel(nn.Module):
    def __init__(self, num_classes: int = 1000) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        self.blocks = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1, groups=32),
            nn.Conv2d(32, 64, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, groups=64),
            nn.Conv2d(64, 128, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.blocks(self.stem(x)))


class CVOnlineInfer(StepWorkload):
    spec = WorkloadSpec(
        workload_id="cv_online_infer",
        task_family="inference",
        workload_class="realistic_model_proxy",
        shape_stability="stable",
        application="CV online inference",
        bottleneck_hypothesis="cuDNN launch overhead at small batch and compute-bound larger batch",
        preferred_metric="images_per_second",
    )

    def __init__(self, batch_size: int, device: torch.device) -> None:
        self.model: nn.Module = ConvOnlineModel().to(device).eval()
        self.x = torch.randn(batch_size, 3, 224, 224, device=device)
        self.last_output: torch.Tensor | None = None

    @torch.no_grad()
    def step(self) -> None:
        self.last_output = self.model(self.x)

    def enable_compile(self, mode: str | None = None) -> None:
        self.model = torch.compile(self.model, mode=mode)

    def refresh_request_inputs(self) -> int:
        self.x.copy_(self._randn_like(self.x))
        return self.x.numel() * self.x.element_size()

    def step_micro_batch(self, chunks: int) -> None:
        outputs = [self.model(part) for part in self.x.chunk(max(1, chunks), dim=0)]
        self.last_output = torch.cat(outputs, dim=0)

    def correctness_state(self) -> dict[str, torch.Tensor]:
        return _last_output_state(self)


class TinyEncoderClassifier(nn.Module):
    def __init__(self, vocab_size: int = 4096, hidden_dim: int = 128, num_heads: int = 4) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            batch_first=True,
            dropout=0.0,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, 2)

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        x = self.embed(tokens)
        padding_mask = None if mask is None else ~mask.bool()
        x = self.encoder(x, src_key_padding_mask=padding_mask)
        if mask is None:
            pooled = x.mean(dim=1)
        else:
            denom = mask.sum(dim=1, keepdim=True).clamp_min(1).to(x.dtype)
            pooled = (x * mask.unsqueeze(-1).to(x.dtype)).sum(dim=1) / denom
        return self.head(self.norm(pooled))


class ShortTextTransformerPadded(StepWorkload):
    spec = WorkloadSpec(
        workload_id="short_text_transformer_padded",
        task_family="inference",
        workload_class="realistic_model_proxy",
        shape_stability="stable",
        application="short text Transformer inference",
        bottleneck_hypothesis="short-sequence Transformer launch fragmentation and padding cost",
        preferred_metric="texts_per_second",
    )

    def __init__(self, batch_size: int, device: torch.device) -> None:
        self.model: nn.Module = TinyEncoderClassifier().to(device).eval()
        self.tokens = torch.randint(0, 4096, (batch_size, 64), device=device)
        self.mask = torch.ones(batch_size, 64, dtype=torch.bool, device=device)
        self.last_output: torch.Tensor | None = None

    @torch.no_grad()
    def step(self) -> None:
        self.last_output = self.model(self.tokens, self.mask)

    def enable_compile(self, mode: str | None = None) -> None:
        self.model = torch.compile(self.model, mode=mode)

    def refresh_request_inputs(self) -> int:
        self.tokens.copy_(self._randint_like(self.tokens, 0, 4096))
        return self.tokens.numel() * self.tokens.element_size()

    def correctness_state(self) -> dict[str, torch.Tensor]:
        return _last_output_state(self)


class ShortTextTransformerDynamic(StepWorkload):
    spec = WorkloadSpec(
        workload_id="short_text_transformer_dynamic",
        task_family="inference",
        workload_class="realistic_model_proxy",
        shape_stability="unstable",
        application="short text Transformer dynamic length",
        bottleneck_hypothesis="dynamic length recompilation and graph ineligibility",
        preferred_metric="texts_per_second",
        supports_cuda_graph=False,
    )

    def __init__(self, batch_size: int, device: torch.device) -> None:
        self.model: nn.Module = TinyEncoderClassifier().to(device).eval()
        self.device = device
        self.batch_size = batch_size
        self.lengths = (16, 32, 64, 128)
        self.cursor = 0
        self.last_output: torch.Tensor | None = None

    @torch.no_grad()
    def step(self) -> None:
        seq_len = self.lengths[self.cursor % len(self.lengths)]
        self.cursor += 1
        tokens = torch.randint(0, 4096, (self.batch_size, seq_len), device=self.device)
        self.last_output = self.model(tokens, None)

    def enable_compile(self, mode: str | None = None) -> None:
        self.model = torch.compile(self.model, mode=mode, dynamic=True)

    def correctness_state(self) -> dict[str, torch.Tensor]:
        return _last_output_state(self)


class LLMDecodeProxy(StepWorkload):
    spec = WorkloadSpec(
        workload_id="llm_decode_proxy",
        task_family="inference",
        workload_class="realistic_model_proxy",
        shape_stability="stable",
        application="single-token LLM decode proxy",
        bottleneck_hypothesis="static KV-cache replay and launch overhead",
        preferred_metric="tokens_per_second",
    )

    def __init__(self, batch_size: int, device: torch.device) -> None:
        self.model: nn.Module = TinyDecodeBlock(hidden_dim=128, mlp_dim=512).to(device).eval()
        self.token = torch.randn(batch_size, 1, 128, device=device)
        self.key_cache = torch.randn(batch_size, 128, 128, device=device)
        self.value_cache = torch.randn(batch_size, 128, 128, device=device)
        self.last_output: torch.Tensor | None = None

    @torch.no_grad()
    def step(self) -> None:
        self.last_output = self.model(self.token, self.key_cache, self.value_cache)

    def enable_compile(self, mode: str | None = None) -> None:
        self.model = torch.compile(self.model, mode=mode)

    def refresh_request_inputs(self) -> int:
        self.token.copy_(self._randn_like(self.token))
        return self.token.numel() * self.token.element_size()

    def correctness_state(self) -> dict[str, torch.Tensor]:
        return _last_output_state(self)


class DLRMRecommendation(StepWorkload):
    spec = WorkloadSpec(
        workload_id="dlrm_recommendation",
        task_family="inference",
        workload_class="realistic_model_proxy",
        shape_stability="mostly_stable",
        application="DLRM-like recommendation inference",
        bottleneck_hypothesis="embedding lookup and small dense MLP memory-bound behavior",
        preferred_metric="requests_per_second",
    )

    def __init__(self, batch_size: int, device: torch.device) -> None:
        self.embeddings = nn.ModuleList([nn.Embedding(2048, 16) for _ in range(8)]).to(device)
        self.mlp: nn.Module = nn.Sequential(
            nn.Linear(8 * 16 + 16, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        ).to(device).eval()
        self.indices = torch.randint(0, 2048, (batch_size, 8), device=device)
        self.dense = torch.randn(batch_size, 16, device=device)
        self.last_output: torch.Tensor | None = None

    @torch.no_grad()
    def step(self) -> None:
        sparse = [embedding(self.indices[:, idx]) for idx, embedding in enumerate(self.embeddings)]
        self.last_output = self.mlp(torch.cat([*sparse, self.dense], dim=1))

    def enable_compile(self, mode: str | None = None) -> None:
        self.mlp = torch.compile(self.mlp, mode=mode)

    def refresh_request_inputs(self) -> int:
        self.indices.copy_(self._randint_like(self.indices, 0, 2048))
        self.dense.copy_(self._randn_like(self.dense))
        return (
            self.indices.numel() * self.indices.element_size()
            + self.dense.numel() * self.dense.element_size()
        )

    def correctness_state(self) -> dict[str, torch.Tensor]:
        return _last_output_state(self)


class SimpleGNN(nn.Module):
    def __init__(self, input_dim: int = 32, hidden_dim: int = 64) -> None:
        super().__init__()
        self.lin_self = nn.Linear(input_dim, hidden_dim)
        self.lin_neigh = nn.Linear(input_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, 8)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        src, dst = edge_index
        agg = torch.zeros_like(x)
        agg.index_add_(0, dst, x[src])
        deg = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
        deg.index_add_(0, dst, torch.ones_like(dst, dtype=x.dtype))
        agg = agg / deg.clamp_min(1).unsqueeze(-1)
        h = F.relu(self.lin_self(x) + self.lin_neigh(agg))
        return self.out(h).mean(dim=0, keepdim=True)


class GNNIrregular(StepWorkload):
    spec = WorkloadSpec(
        workload_id="gnn_irregular",
        task_family="inference",
        workload_class="irregular_realistic_proxy",
        shape_stability="unstable",
        application="small irregular GNN",
        bottleneck_hypothesis="scatter/gather and dynamic graph-size overhead",
        supports_cuda_graph=False,
        preferred_metric="graphs_per_second",
    )

    def __init__(self, batch_size: int, device: torch.device) -> None:
        self.model: nn.Module = SimpleGNN().to(device).eval()
        self.device = device
        self.num_nodes = max(32, batch_size * 16)
        self.edge_counts = (self.num_nodes * 2, self.num_nodes * 3, self.num_nodes * 4)
        self.cursor = 0
        self.last_output: torch.Tensor | None = None

    @torch.no_grad()
    def step(self) -> None:
        edge_count = self.edge_counts[self.cursor % len(self.edge_counts)]
        self.cursor += 1
        x = torch.randn(self.num_nodes, 32, device=self.device)
        edge_index = torch.randint(0, self.num_nodes, (2, edge_count), device=self.device)
        self.last_output = self.model(x, edge_index)

    def enable_compile(self, mode: str | None = None) -> None:
        self.model = torch.compile(self.model, mode=mode, dynamic=True)

    def correctness_state(self) -> dict[str, torch.Tensor]:
        return _last_output_state(self)


class RLPolicyInfer(StepWorkload):
    spec = WorkloadSpec(
        workload_id="rl_policy_infer",
        task_family="inference",
        workload_class="cpu_boundary_proxy",
        shape_stability="stable",
        application="RL/control policy inference",
        bottleneck_hypothesis="tiny MLP launch overhead where CPU may be rational",
        preferred_metric="actions_per_second",
    )

    def __init__(self, batch_size: int, device: torch.device) -> None:
        self.model: nn.Module = nn.Sequential(
            nn.Linear(17, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 6),
        ).to(device).eval()
        self.obs = torch.randn(batch_size, 17, device=device)
        self.last_output: torch.Tensor | None = None

    @torch.no_grad()
    def step(self) -> None:
        self.last_output = self.model(self.obs)

    def enable_compile(self, mode: str | None = None) -> None:
        self.model = torch.compile(self.model, mode=mode)

    def refresh_request_inputs(self) -> int:
        self.obs.copy_(self._randn_like(self.obs))
        return self.obs.numel() * self.obs.element_size()

    def correctness_state(self) -> dict[str, torch.Tensor]:
        return _last_output_state(self)


class SyntheticKernelChain(StepWorkload):
    spec = WorkloadSpec(
        workload_id="synthetic_kernel_chain",
        task_family="inference",
        workload_class="mechanism_microbenchmark",
        shape_stability="stable",
        application="fusion and launch-overhead mechanism study",
        bottleneck_hypothesis="elementwise and normalization chain fusion opportunity",
        preferred_metric="chains_per_second",
    )

    def __init__(self, batch_size: int, device: torch.device) -> None:
        width = 1024
        self.weight = torch.randn(width, width, device=device) / sqrt(width)
        self.bias = torch.randn(width, device=device)
        self.x = torch.randn(max(1, batch_size), width, device=device)
        self.last_output: torch.Tensor | None = None

    @torch.no_grad()
    def step(self) -> None:
        x = self.x @ self.weight + self.bias
        x = F.gelu(x)
        x = x + torch.sin(x) * 0.1
        self.last_output = F.layer_norm(x, (x.shape[-1],))

    def enable_compile(self, mode: str | None = None) -> None:
        self.step = torch.compile(self.step, mode=mode)  # type: ignore[method-assign]

    def refresh_request_inputs(self) -> int:
        self.x.copy_(self._randn_like(self.x))
        return self.x.numel() * self.x.element_size()

    def correctness_state(self) -> dict[str, torch.Tensor]:
        return _last_output_state(self)


def make_workload(workload_id: str, batch_size: int, device: torch.device) -> StepWorkload:
    if workload_id == "fixed_shape_infer_small":
        return FixedShapeInferSmall(batch_size=batch_size, device=device)
    if workload_id == "short_train_small":
        return ShortTrainSmall(batch_size=batch_size, device=device)
    if workload_id == "dynamic_shape_infer_small":
        return DynamicShapeInferSmall(batch_size=batch_size, device=device)
    if workload_id == "prefill_toy":
        return PrefillToy(batch_size=batch_size, device=device)
    if workload_id == "decode_toy":
        return DecodeToy(batch_size=batch_size, device=device)
    if workload_id == "cv_online_infer":
        return CVOnlineInfer(batch_size=batch_size, device=device)
    if workload_id == "short_text_transformer_padded":
        return ShortTextTransformerPadded(batch_size=batch_size, device=device)
    if workload_id == "short_text_transformer_dynamic":
        return ShortTextTransformerDynamic(batch_size=batch_size, device=device)
    if workload_id == "llm_decode_proxy":
        return LLMDecodeProxy(batch_size=batch_size, device=device)
    if workload_id == "dlrm_recommendation":
        return DLRMRecommendation(batch_size=batch_size, device=device)
    if workload_id == "gnn_irregular":
        return GNNIrregular(batch_size=batch_size, device=device)
    if workload_id == "rl_policy_infer":
        return RLPolicyInfer(batch_size=batch_size, device=device)
    if workload_id == "synthetic_kernel_chain":
        return SyntheticKernelChain(batch_size=batch_size, device=device)
    raise ValueError(f"unknown workload_id: {workload_id}")
