# Dataset layout

Place datasets under this repository's `datasets/` directory using these names:

`BRCA/`, `ROSMAP/`, `mustard/`, `Food101/`, `CrisisMMD1/`, `CrisisMMD2/`, and `CrisisMMD3/`.

The paths in common training scripts resolve from the script location, not the current working directory. Place the actual input files directly in the appropriate directory. The downloaded archive has not yet been checked, and some historical scripts require different feature files or image files. Do not assume a shared split or feature representation without checking the experiment configuration.

Run scripts from their own method directory: `cd proposed/BRCA` or `cd baselines/BRCA/ITHP`, for example. The historical hyperparameters and dependencies still need verification before reproducing the paper.
