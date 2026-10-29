# Camera-ready metric definitions

## Measurement contracts and eligibility

The controlled comparison uses fixed inputs and excludes `graphs_input_copy`, whose input-refresh contract differs. Canonical eager is the predetermined `best_eager` implementation for inference and literal eager for training. Calibration, oracle selection, policies, and failed-action queue service use the same baseline.

Historical reference-cell screening and runtime feasibility jointly determine eligibility. Expanded training checks validate exact effective-update counts without changing the eligibility of historical training timings. Real-model comparisons match inference mode, input refresh, and strict-FP32 settings, and record initialization, optimization setup, and execution separately.

## Jobs and policies

- **Oracle:** the action with the lowest median time among candidates with the same job contract that are runtime-feasible and pass the applicable correctness screen.
- **Normalized regret:** `(T_selected - T_oracle) / T_oracle`.
- **Invalid requests:** counted separately and excluded from single-job regret; `n_valid` reports the valid sample count.
- **Slow admission:** a valid non-eager choice slower than canonical eager.
- **Missed opportunity:** a valid eager choice despite a faster eligible non-eager candidate. Both decision-error rates divide by all 208 controlled cold conditions.
- **P90 and queue P95:** select index `floor(q * (n - 1))` after sorting, preserving the frozen implementation's non-interpolated definition.
- Calibration excludes all actions and repeats of the target condition. Family holdout additionally excludes the target workload family.

## Prediction and sensitivity

The original eager cost estimator and affine diagnostic use the same three non-target horizons. The affine diagnostic requires nonnegative intercept and slope and replaces only the eager prediction. Actual target time is used for retrospective evaluation.

Retrospective eager-time diagnostics use the full target horizon's mean eager step time in 768 of 832 conditions and a five-step prefix for the remaining 64 training conditions. The `+margin` IQR comes from total-time medians across horizons within the retained bucket, together with a 2% predicted advantage threshold relative to eager.

Repeat sensitivity omits the same repeat index from every action-condition pair, then recomputes medians, oracles, and policies. Heavy-load uniform queue replay retains the arrival streams generated from the full five-repeat analysis.

## Queue replay

A single initially empty worker serves jobs in FCFS order. Each trace contains 1,000 jobs, Poisson arrivals, and a seed from 1-5. Target loads are 0.35, 0.65, and 0.90. Service times are five-repeat medians for each action-condition pair.

| Mix | Relative sampling weights; unlisted weights are 1 |
|---|---|
| Uniform | All weights are 1 |
| Graph-friendly | LLM decode 5, decode toy 4, fixed inference 3, kernel chain 3 |
| Dynamic/irregular | Dynamic inference 5, irregular GNN 5, dynamic text 4 |
| Short-step | Weights 6, 3, 1, and 0.5 for N = 10, 50, 100, and 500 |

Condition sampling weight is workload weight times horizon weight. Arrival rates are normalized using the mix-specific weighted mean matched-oracle service time. Policies share jobs and arrivals within a trace. Invalid actions immediately use canonical eager service with zero additional recovery cost.

Reported P95 values are the mean of the five per-trace P95 values; sample standard deviations are also computed across those five values. Improvement relative to eager is the ratio change between the two mean per-trace P95 values. Throughput is 1,000 divided by trace makespan, averaged across five traces.

Wasted optimization overhead sums a startup proxy over valid non-eager choices slower than canonical eager. The proxy is the maximum of the recorded median compile-call, compile-overhead, capture, input-copy, and warm-up timers.

## Real models and boundary plots

Each session/model/horizon/action combination retains five formal timings. Graph effect is `100 * (median(graph) / median(eager) - 1)`. Its observed envelope is `100 * ([min(graph)/max(eager), max(graph)/min(eager)] - 1)`.

For each device/model/horizon, the plotted point is the median of percentage changes across measurement blocks; the envelope spans the observed bounds of those blocks. CV is the sample standard deviation of five timings divided by their mean. Each heatmap cell counts graph-oracle selections across four batch sizes.

Controlled timing covers optimization setup and finite execution after workload initialization. Real-model timing also includes model, weight, and input initialization after CUDA context creation. Eight-job reuse amortizes one capture over a correlated group. Distinct contracts and sessions are analyzed separately.
