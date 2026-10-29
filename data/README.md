# Experimental data

Obtain formal measurements and results from the GitHub Release asset `shortjob_camera_ready_v0.2.0_data.tar.gz`. See the [reproduction guide](../REPRODUCIBILITY.md) for extraction, verification, and reconstruction commands.

| Collection | Formal records |
|---|---:|
| Four controlled configurations: reference-cell gates, cold, warm, priming, and reuse | 36,480 |
| Eight supplemental sessions | 3,071 |
| Full independent V100 replication, including the ninth real-model session | 9,583 |
| Total | 49,134 |

The nine real-model sessions include 648 correctness checks and 1,620 formal timings. Formal failures and infeasibility outcomes are retained for eligibility and limitation analyses.

The archive's `data/camera_ready_v0.2.0/` contains historical measurements, `supplement/`, `replication/`, asset identities, reference values, a data manifest, and checksums. Its `results/` directory contains current and historical derived results, CSV tables, and figures. Frozen measurement code is stored in this repository's `measurement_source/`.

Images, model weights, prepared input tensors, development smoke runs, installation checks, and local review materials are managed separately. Asset manifests record sources, selection rules, weight identities, and SHA-256 hashes for reconstructing the inputs.

New session projections omit private execution paths and machine identities; historical records retain their original bytes. Sessions and timing contracts are analyzed separately. See [measured environments](../ENVIRONMENTS.md).

This directory contains only this document and the empty-directory placeholders `raw/.gitkeep` and `env/.gitkeep`. Keep measurements, extracted data, and results in an external working directory. Use a new session name for each measurement campaign.
