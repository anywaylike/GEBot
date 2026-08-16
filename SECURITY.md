# Security and credential handling

This repository is prepared for private research collaboration. It contains no
provider credentials, access identifiers, local model weights, raw datasets,
or experiment outputs.

- Store provider credentials only in a local `.env` file or your environment.
- Never add a credential to a command, notebook output, issue, commit, or pull
  request.
- Keep `.env`, `data/`, `models/`, `outputs/`, and checkpoints untracked.
- If a credential was ever committed elsewhere, revoke and rotate it at the
  provider before sharing any repository history.

Before each push, inspect `git status --short` and search the staged files for
provider keys and access identifiers. Treat generated LLM responses as private
experiment artifacts unless they have been reviewed for release.
