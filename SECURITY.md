# Security and credential handling

This public repository contains no provider credentials, access identifiers,
local model weights, raw datasets, selected user cohorts, or experiment
outputs.

- Store provider credentials only in a local `.env` file or your environment.
- Never add a credential to a command, notebook output, issue, commit, or pull
  request.
- Keep `.env`, `data/`, `models/`, `outputs/`, and checkpoints untracked.
- If a credential was ever committed elsewhere, revoke and rotate it at the
  provider before sharing any repository history.

Before each push, inspect `git status --short` and search the staged files for
provider keys and access identifiers. Treat generated LLM responses as private
experiment artifacts unless they have been reviewed for release.

Report a suspected credential or private-data exposure to the maintainers
privately before opening a public issue. Revoke and rotate any exposed
credential immediately, even if the affected commit is later removed.
