# Experiments

Run commands from the repository root after installing the package and
preparing authorized local dataset copies. All generated files are written
under `outputs/`, which is intentionally ignored by Git.

## Environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[dev]"
Copy-Item .env.example .env
```

Fill in only the provider variables required for the selected model. The
aliases `GPT`, `DeepSeek`, and `ZhipuAI` use the matching provider settings;
other model identifiers are sent to the OpenAI-compatible endpoint configured
by `OPENAI_BASE_URL`.

## Data-source modes

The unified evaluator supports two local sources:

| `--data_source` | Input | Intended use |
| --- | --- | --- |
| `fixed_test` | The configured TwiBot-22 and BotSim-24 test files | Public fixed-split evaluation; `--n_per_class` is capped by the smaller available class |
| `selected_1000` | Local `selected_1000.json` files | The private paper-cohort pathway or a locally generated deterministic alternative |

The standard BotSim-24 8:2 test split contains only 200 bots. Therefore,
`fixed_test --sampling_mode balanced --n_per_class 500` evaluates 200 bots and
200 humans for that dataset. It must not be described as the paper's 500/500
cohort.

To create a deterministic alternative cohort from authorized, prepared source
files, run:

```powershell
python -m graphreason_bot.prepare_cohort --dataset all `
  --sample-seed 2026 --sample-size 1000
```

The generated user records stay below `data/` and are ignored by Git. They are
not guaranteed to be byte-identical to the unpublished paper cohorts when the
upstream release or preprocessing revision differs.

## Validation

```powershell
python -m graphreason_bot.evaluation --dataset all --validate_only
python -m graphreason_bot.evaluation --help
```

`--validate_only` verifies that configured files can be loaded and that both
labels are present. It does not validate every graph field, provider setting,
or the availability of 500 examples per class.

## Main configurations

### Paper-cohort pathway

Authorized maintainers with the original local cohort files can run:

```powershell
python -m graphreason_bot.evaluation `
  --dataset all --data_source selected_1000 --model_name gpt-4o-mini `
  --sampling_mode balanced --n_per_class 500 `
  --neighbor_ratio 0.6 --prkns_method followers `
  --run_tag paper_cohort --result_dir outputs/paper_cohort
```

### Public fixed-test pathway

```powershell
python -m graphreason_bot.evaluation `
  --dataset all --data_source fixed_test --model_name gpt-4o-mini `
  --sampling_mode balanced --n_per_class 500 `
  --neighbor_ratio 0.6 --prkns_method followers `
  --run_tag fixed_test --result_dir outputs/fixed_test
```

Replace `gpt-4o-mini` with the provider model identifier used for the recorded
run. Execute the same command separately for each downstream LLM and preserve
the model identifier in the experiment log.

## Component and feature ablations

Use the same data source, model, seed, and output protocol as the corresponding
full run. The examples below show the private paper-cohort pathway; replace
`selected_1000` with `fixed_test` for public fixed-split evaluation.

```powershell
# Remove WGSE, SE-HCI, and PRKNS one at a time.
python -m graphreason_bot.evaluation `
  --dataset all --data_source selected_1000 --model_name gpt-4o-mini `
  --sampling_mode balanced --n_per_class 500 --neighbor_ratio 0.6 `
  --prkns_method followers --run_ablations `
  --run_tag component_ablation --result_dir outputs/ablations/component

# Retain the structural pipeline and run w/o metadata and w/o text.
python -m graphreason_bot.evaluation `
  --dataset all --data_source selected_1000 --model_name gpt-4o-mini `
  --sampling_mode balanced --n_per_class 500 --neighbor_ratio 0.6 `
  --prkns_method followers --run_feature_ablations `
  --run_tag feature_ablation --result_dir outputs/ablations/features
```

## Sensitivity and ranking-policy analyses

Repeat each value with an unchanged cohort, model, seed, and output protocol.
Do not choose the paper's primary hyperparameter from the held-out test curve.

```powershell
# Neighbor ratio: repeat r in 0.3, 0.4, ..., 0.9.
python -m graphreason_bot.evaluation `
  --dataset Twibot-22 --data_source selected_1000 --model_name gpt-4o-mini `
  --sampling_mode balanced --n_per_class 500 --neighbor_ratio 0.4 `
  --prkns_method followers --run_tag ratio_r04 `
  --result_dir outputs/sensitivity/ratio

# Ranking policy: repeat with pagerank, personalized_pagerank, rwr, degree,
# followers, betweenness, closeness, and other policies shown by --help.
python -m graphreason_bot.evaluation `
  --dataset Twibot-22 --data_source selected_1000 --model_name gpt-4o-mini `
  --sampling_mode balanced --n_per_class 500 --neighbor_ratio 0.6 `
  --prkns_method degree --run_tag policy_degree `
  --result_dir outputs/sensitivity/policy
```

## Matched prompt baselines

The baseline default `--source original-random` creates a deterministic
balanced sample from the prepared original pool using seed 2026. For BotSim-24
this requires both `processed/train.json` and `processed/test.json`. Use an
explicit model identifier rather than `--model all` when recording a single
reproducible run.

Start with low-cost dry runs:

```powershell
# Random prompting
python -m graphreason_bot.baselines.random_twibot22 `
  --model gpt-4o-mini --max-users 10 --dry-run
python -m graphreason_bot.baselines.random_botsim24 `
  --model gpt-4o-mini --max-users 10 --dry-run

# Attention prompting
python -m graphreason_bot.baselines.attention_twibot22 `
  --model gpt-4o-mini --max-users 10 --dry-run
python -m graphreason_bot.baselines.attention_botsim24 `
  --model gpt-4o-mini --max-users 10 --dry-run

# MRT and CISC on both datasets
python -m graphreason_bot.baselines.mrt `
  --dataset all --model gpt-4o-mini --max-users 10 --dry-run
python -m graphreason_bot.baselines.cisc `
  --dataset all --model gpt-4o-mini --traces 3 --max-users 10 --dry-run
```

Remove `--max-users 10 --dry-run` only after checking the prompts and provider
configuration. Use `--source current-paper` only when the intended fixed test
files are present and the resulting class counts are recorded.

The vendor baselines used for broader paper comparisons are intentionally not
vendored. Record their upstream URL, commit, and license separately.

## Output layout

Unified evaluator outputs use the model identifier as a directory:

```text
outputs/<run-family>/<model>/
├── checkpoints/     # resumable per-user JSONL state
├── runs/            # individual run JSON reports
└── *suite*.json     # aggregated suite reports
```

Baseline outputs are grouped by method:

```text
outputs/baselines/
├── random/
├── attention/
├── mrt/
└── cisc/
```

Keep outputs private unless each artifact has been reviewed for credentials,
user records, prompts, and provider responses.
