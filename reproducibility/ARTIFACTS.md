# External Artifacts

Large files are intentionally excluded from Git. This file records the expected layout for external artifact bundles used to reproduce the paper-level analyses. When an artifact bundle is distributed, its release page should provide exact file sizes, SHA256 checksums, and download URLs for the entries below.

## Required Core Artifacts

| Artifact | Expected path | Distribution status |
|---|---|---|
| Query-refined transfer posthoc summary | `analysis_outputs/pure_af_geometry/transport_hybrid_benchmark_a/posthoc_summary/` | External artifact bundle |
| Objective-neutral mobility summaries | `analysis_outputs/pure_af_geometry/` | External artifact bundle |
| JVP mechanism control summaries | `analysis_outputs/pure_af_geometry/` | External artifact bundle |
| Matched intervention summaries | `analysis_outputs/pure_af_geometry/` | External artifact bundle |
| CIFAR-10 model checkpoints | `checkpoints/blackboxbench_cifar10/` or documented equivalent | External checkpoint bundle |

## Optional Audit Artifacts

| Artifact | Expected path | Distribution status |
|---|---|---|
| Full raw trajectory vectors | `analysis_outputs/pure_af_geometry/` | External artifact bundle |
| Full query curves | `analysis_outputs/pure_af_geometry/transport_hybrid_benchmark_a/posthoc_summary/combined_query_curves.csv` | External artifact bundle |
