# Release scope

This repository contains the author-maintained implementation of the proposed
GEBot pipeline and local implementations of the Random, Attention,
MRT, and CISC prompting baselines.

The following materials are deliberately excluded:

| Material | Reason |
| --- | --- |
| Provider credentials and access identifiers | Security |
| Raw/preprocessed datasets, user IDs, and the selected 1,000-user cohorts | Dataset terms, privacy review, and reproducibility control |
| Local transformer weights | Size and upstream model licensing |
| Results, checkpoints, logs, prompts, and LLM responses | They may contain user records or private experimental evidence |
| `baselines/vendor/` snapshots | Third-party code must be obtained from its original source and used under its own license |

Document the exact dataset release, preprocessing revision, model identifier,
endpoint family, seed, and command for every reported result. This preserves a
clean code history while keeping the paper's experimental record auditable.
