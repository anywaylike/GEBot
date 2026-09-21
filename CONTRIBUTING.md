# Contributing to GEBot

Thank you for helping improve GEBot. Bug reports, documentation corrections,
and reproducibility questions are welcome. Because the repository is currently
all-rights-reserved, obtain written permission from the copyright holders
before using, modifying, or submitting changes to the code.

## Before opening an issue

1. Check the existing issues and the commands in `docs/EXPERIMENTS.md`.
2. Run `python -m graphreason_bot.evaluation --help` to confirm the supported
   interface.
3. For data-loading problems, report the dataset name and local schema, but do
   not attach user records or account identifiers.

## Development setup

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[dev]"
pytest -q
```

## Authorized pull requests

- Keep each change focused and explain its motivation.
- Add or update tests when behavior changes.
- Update the README or experiment documentation when commands or defaults
  change.
- Preserve the existing public interfaces unless the change explicitly
  documents a migration.
- Confirm that `git diff --check` and `pytest -q` pass.

## Data and credential policy

Never commit or attach:

- TwiBot-22, BotSim-24, or any derived user records;
- the selected 1,000-user cohorts or their account IDs;
- provider keys, tokens, `.env` files, or access identifiers;
- generated prompts, model responses, checkpoints, or experiment outputs that
  may contain user information.

Use synthetic minimal examples when a report needs data to reproduce a bug.
See `SECURITY.md` and `docs/REPOSITORY_SCOPE.md` for the complete release scope.

## Licensing

Submitting an authorized contribution does not change the repository's
licensing status. The project remains all-rights-reserved unless the copyright
holders add an explicit license.
