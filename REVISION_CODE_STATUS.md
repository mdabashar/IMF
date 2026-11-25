# IMF code staging

This repository separates proposed implementations and baseline candidates by dataset. The code is provided for review and further verification.

- `proposed/`: historical IMF training scripts and local Python dependencies for seven datasets.
- `baselines/`: historical scripts under `legacy/`, ITHP adaptations under `ITHP/`, and candidate ARL/AUG implementations copied from the separate mm-mismatch project.

The ARL/AUG copies are candidates for new experiments, not verified reproductions of the results in this paper. Their algorithm fidelity, data paths, splits and model selection must be checked against the paper and official methods before use. Historical script parameters and paths also need validation. Dataset files, environments and results are intentionally excluded.

Known gaps: ROSMAP legacy baseline scripts are adapted copies of BRCA (200 features per modality, two classes), not historical ROSMAP scripts; they require data-path and runtime verification. CrisisMMD3 ITHP was recovered separately from HPC. The other historical baseline entries are not necessarily complete for every task. Do not interpret a directory's presence as evidence that all paper baselines are runnable. Food101 includes alternative proposed training scripts whose correspondence to reported results is unresolved.
