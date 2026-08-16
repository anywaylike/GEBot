# Data contract

No dataset, user records, model-selected subsets, cached LLM responses, or
model weights are included in this repository.

Place locally prepared files under `data/` with the following layout:

```text
data/
├── Twibot-22/
│   ├── community_format/test.json
│   └── selected_1000.json             # optional private subset
└── BotSim-24/
    ├── processed/test.json
    └── selected_1000.json             # optional private subset
```

The main paper protocol uses the fixed test sets (`community_format/test.json`
and `processed/test.json`). The optional selected subsets are supported only
for private internal analyses and should not be used as a substitute for a
prespecified held-out evaluation protocol.

Set `GRAPHREASON_DATA_ROOT` when the files live outside this repository.
Respect each dataset's original license, terms of use, and any platform-data
restrictions before sharing a prepared copy.
