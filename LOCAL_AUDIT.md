# Local audit (in progress)

Syntax parsing and a static check for local `from ... import` modules were performed; these do not verify imports, execution, data compatibility, or model correctness. Dataset paths have been partially standardized and must be verified against the actual dataset archive before use.

- The Food101 legacy `sing_main.py` references `dataset` and `models_baselines`, neither of which was found at the original Food101 project root. Do not substitute similarly named modules without verifying their interfaces.
- ROSMAP ITHP contains `main_meta_new_ms_2bnn.py`, which appears to be an unrelated proposed-model copy and is missing its own model and optimizer dependencies. The ITHP entry point is `main.py`; retain the extra file for provenance until reviewed.
- The current Mac Python lacks PyTorch, scikit-learn and transformers. Downloaded HPC environments contain Linux executables and cannot run on macOS. Model initialization and forward tests have not been performed.
- Legacy job scripts and dataset configuration still require verification against actual data and the historic experiment protocol. CrisisMMD3 and Food101 have multiple candidate scripts; ARL/AUG are candidate adaptations from another project.
- Dataset files have not been inspected yet. Relative paths to unused datasets and alternate data sources remain in historical helper functions, and not all possible data-loading expressions have been rewritten.
