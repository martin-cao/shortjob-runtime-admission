from __future__ import annotations

from dataclasses import dataclass


ACTIONS = (
    "eager",
    "eager_inference_mode",
    "best_eager",
    "cpu_eager",
    "compile_only",
    "compile_reduce_overhead",
    "compile_max_autotune",
    "graphs_only",
    "graphs_input_copy",
    "compile_plus_graphs",
    "compile_reduce_overhead_plus_graphs",
    "amp_fp16",
    "amp_bf16",
    "tf32_eager",
    "micro_batch_2",
    "micro_batch_4",
)
WORKLOADS = (
    "fixed_shape_infer_small",
    "short_train_small",
    "dynamic_shape_infer_small",
    "prefill_toy",
    "decode_toy",
    "cv_online_infer",
    "short_text_transformer_padded",
    "short_text_transformer_dynamic",
    "llm_decode_proxy",
    "dlrm_recommendation",
    "gnn_irregular",
    "rl_policy_infer",
    "synthetic_kernel_chain",
)
SYNC_POLICY = "synchronize_before_and_after_timed_region"


@dataclass(frozen=True)
class WorkloadSpec:
    workload_id: str
    task_family: str
    workload_class: str
    shape_stability: str
    input_mode: str = "synthetic"
    application: str = "controlled mechanism"
    bottleneck_hypothesis: str = "launch overhead / framework overhead"
    preferred_metric: str = "steps_per_second"
    supports_cpu: bool = True
    supports_cuda_graph: bool = True
    supports_training: bool = False
    supports_mixed_precision: bool = True
    supports_micro_batching: bool = True
    notes: str = ""


@dataclass(frozen=True)
class Condition:
    workload_id: str
    action: str
    batch_size: int
    num_steps: int
    repeat: int


class RunnerStageError(RuntimeError):
    def __init__(self, stage: str, cause: BaseException) -> None:
        self.stage = stage
        self.cause = cause
        super().__init__(f"{stage}: {cause}")
