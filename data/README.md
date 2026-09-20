# Data acquisition and local data contract

No dataset, user records, user identifiers, selected 1,000-user cohorts, cached
LLM responses, or model weights are included in this repository or its Git
history.

## Obtain the original datasets

Download the datasets only from their official project pages and comply with
their licenses, access conditions, and platform-data restrictions:

1. **TwiBot-22:** use the download links and instructions in the
   [official TwiBot-22 repository](https://github.com/LuoUndergradXJTU/TwiBot-22).
2. **BotSim-24:** use the [official BotSim repository](https://github.com/QQQQQQBY/BotSim)
   and its [BotSim-24 dataset instructions](https://github.com/QQQQQQBY/BotSim/blob/main/BotSim-24-Dataset/Readme.md).

The upstream repositories are the authoritative sources. This project does
not mirror or redistribute their files.

## Prepare files locally

Place locally prepared files under `data/` with the following layout:

```text
data/
├── Twibot-22/
│   ├── community_format/test.json
│   └── selected_1000.json             # optional; local/private only
└── BotSim-24/
    ├── processed/test.json
    └── selected_1000.json             # optional; local/private only
```

Convert the authorized upstream copies into the JSON structures expected by
`src/graphreason_bot/data_loader.py`, preserving the original labels and graph
relationships. The main paper protocol uses the fixed test files
(`community_format/test.json` and `processed/test.json`). For analyses that
sample from the original pools, the included sampling code defaults to 1,000
users balanced as 500 bots and 500 humans with seed 2026.

The exact paper cohorts, including `selected_1000.json` and their account IDs,
are intentionally not released. Do not commit those files, derived user
records, prompts, cached responses, or outputs. The repository's `.gitignore`
keeps everything under `data/` private except this README.

Set `GRAPHREASON_DATA_ROOT` when the files live outside this repository.
Respect each dataset's original license, terms of use, and any platform-data
restrictions before sharing a prepared copy.
