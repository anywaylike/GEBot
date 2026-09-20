# GEBot

**GEBot: Graph Evidence for Large Language Model-based Social Bot Detection**
is a training-free framework that transforms target-centered local graphs into
compact, decision-relevant structural evidence for frozen-LLM inference. The
pipeline combines:

1. **Graph Structure Enhancement (WGSE)** - controlled graph refinement from
   user and local-topological similarity;
2. **Structural Evidence Organization (SE-HCI)** - structural-entropy-guided
   identification of core-community and bridging evidence; and
3. **Evidence Selection and Structured Prompting (PRKNS)** - budget-aware
   representative-neighbor ranking and prompt construction.

The implementation supports fixed-test evaluation on TwiBot-22 and BotSim-24,
component and feature ablations, sensitivity/policy analyses, and matched
Random, Attention, MRT, and CISC prompt baselines.

## Repository layout

```text
src/graphreason_bot/
├── evaluation.py              # Unified main/ablation evaluation entry
├── wgse.py, se_hci.py, prkns.py, prompt.py
├── data_loader.py, metrics.py, llm.py
└── baselines/                 # Author-maintained prompt baselines
data/README.md                 # Required local dataset layout
docs/EXPERIMENTS.md            # Reproducible commands by experiment family
docs/REPOSITORY_SCOPE.md       # Included and intentionally excluded materials
outputs/                       # Local generated artifacts (ignored)
```

## Quick start

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .
Copy-Item .env.example .env
python -m graphreason_bot.evaluation --dataset all --validate_only
```

Place the prepared datasets locally as described in [data/README.md](data/README.md),
then follow the commands in [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md).

## Data availability

This repository does **not** distribute TwiBot-22, BotSim-24, the balanced
1,000-user evaluation cohorts used in our experiments, their user identifiers,
or any processed records derived from them. Obtain the original datasets from
their official repositories and follow the respective access and license terms:

- [TwiBot-22 official repository](https://github.com/LuoUndergradXJTU/TwiBot-22)
- [BotSim official repository](https://github.com/QQQQQQBY/BotSim), including
  the [BotSim-24 dataset instructions](https://github.com/QQQQQQBY/BotSim/blob/main/BotSim-24-Dataset/Readme.md)

After downloading an authorized copy, prepare the local file layout described
in [data/README.md](data/README.md). The repository includes deterministic
sampling support (default: 1,000 users, balanced as 500 bots and 500 humans,
seed 2026), but the exact cohort files and selected account IDs remain private
and are excluded from Git.

## Security and release status

This is a **public research-code repository**. It contains no credentials,
access identifiers, datasets, selected user cohorts, local model weights,
generated outputs, checkpoints, or third-party vendor snapshots. Credentials
are loaded only from environment variables or a local `.env` file that is
excluded from version control.

See [SECURITY.md](SECURITY.md) before sharing branches or inviting
collaborators. Public access does not grant an open-source license; the project
remains all-rights-reserved unless a license is added explicitly.

## Citation

Please cite the paper **“GEBot: Graph Evidence for Large Language Model-based
Social Bot Detection.”** Complete citation details will be added after the
publication record is final.
