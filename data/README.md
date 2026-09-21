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
    ├── processed/train.json              # original-pool sampling
    ├── processed/test.json
    └── selected_1000.json                 # optional; local/private only
```

Convert the authorized upstream copies into the JSON structures expected by
`src/graphreason_bot/data_loader.py`, preserving the original labels and graph
relationships. The fixed-test pathway reads `community_format/test.json` and
`processed/test.json`. Original-pool BotSim-24 sampling combines its prepared
`train.json` and `test.json`, because the standard test split contains only 200
bot accounts.

Each top-level JSON value may be a list of records or an object keyed by target
ID. Every record must contain `target_id` (or `ID`). A `bot` or `human` label
may appear at the record's top level or on the target node inside
`community.nodes`. Graph context can be represented either as a `community`
object or as `follower`/`following` first-hop objects. A minimal synthetic
community record looks like this:

```json
{
  "target_id": "synthetic-target",
  "label": "bot",
  "description": "synthetic account",
  "followers_count": 10,
  "following_count": 5,
  "tweet_count": 20,
  "community": {
    "nodes": {
      "synthetic-target": {"is_target": true, "label": "bot"},
      "synthetic-neighbor": {
        "is_target": false,
        "description": "synthetic neighbor",
        "followers_count": 2,
        "following_count": 3,
        "tweet_count": 4
      }
    },
    "edges": [["synthetic-target", "synthetic-neighbor"]]
  }
}
```

The loader also accepts `follower.first_hop` and `following.first_hop` maps;
see `src/graphreason_bot/data_loader.py` for the precise compatibility logic.
An end-to-end converter from every upstream release is not currently included,
so record the upstream release and any local field conversion used. Exact
preprocessing reproduction is not claimed without that information.

## Create a deterministic local cohort

After both authorized datasets have been prepared, generate balanced local
cohorts of 1,000 users (500 bots and 500 humans, seed 2026):

```bash
python -m graphreason_bot.prepare_cohort --dataset all
```

Use `--sample-seed`, `--sample-size`, or `--unbalanced-sample` to change the
protocol. Existing files are protected unless `--overwrite` is supplied. The
generated records remain under `data/` and are ignored by Git.

The exact paper cohorts, including `selected_1000.json` and their account IDs,
are intentionally not released. A newly generated deterministic cohort is a
transparent alternative, not a guarantee of byte-identical reproduction when
the upstream data or preprocessing revision differs. Do not commit those
files, derived user records, prompts, cached responses, or outputs. The
repository's `.gitignore` keeps everything under `data/` private except this
README.

Set `GRAPHREASON_DATA_ROOT` when the files live outside this repository.
Respect each dataset's original license, terms of use, and any platform-data
restrictions before sharing a prepared copy.
