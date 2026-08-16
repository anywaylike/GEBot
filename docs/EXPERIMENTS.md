# Experiments

Run commands from the repository root after installing the package and placing
the prepared datasets in `data/`. Outputs are written to `outputs/`, which is
intentionally ignored by git.

## Environment and smoke test

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .
Copy-Item .env.example .env
python -m graphreason_bot.evaluation --dataset all --validate_only
```

Set only the provider variables required for the chosen model in `.env`. Use
`--max-users` on the baseline commands for a low-cost smoke test before a full
LLM evaluation.

## Main experiment

The prespecified primary configuration uses both fixed test sets, one balanced
500/500 cohort per dataset, `r=0.6`, and the Followers policy.

```powershell
python -m graphreason_bot.evaluation `
  --dataset all --data_source fixed_test --model_name gpt-4o-mini `
  --sampling_mode balanced --n_per_class 500 `
  --neighbor_ratio 0.6 --prkns_method followers `
  --run_tag main --result_dir outputs/main
```

Replace `gpt-4o-mini` with the provider model identifier used for a recorded
paper run. The same command can be executed once per downstream LLM.

## Component and feature ablations

```powershell
# Remove WGSE, SE-HCI, and PRKNS one at a time.
python -m graphreason_bot.evaluation `
  --dataset all --data_source fixed_test --model_name gpt-4o-mini `
  --sampling_mode balanced --n_per_class 500 --neighbor_ratio 0.6 `
  --prkns_method followers --run_ablations `
  --run_tag component_ablation --result_dir outputs/ablations/component

# Retain the full structural pipeline and run w/o metadata and w/o text.
python -m graphreason_bot.evaluation `
  --dataset all --data_source fixed_test --model_name gpt-4o-mini `
  --sampling_mode balanced --n_per_class 500 --neighbor_ratio 0.6 `
  --prkns_method followers --run_feature_ablations `
  --run_tag feature_ablation --result_dir outputs/ablations/features
```

## Sensitivity and policy analyses

Run each value on the same fixed data, model, seed, and output protocol; never
select the paper's main hyperparameter from the held-out test curve.

```powershell
# Neighbor ratio: repeat r in 0.3, 0.4, ..., 0.9.
python -m graphreason_bot.evaluation `
  --dataset Twibot-22 --data_source fixed_test --model_name gpt-4o-mini `
  --sampling_mode balanced --n_per_class 500 --neighbor_ratio 0.4 `
  --prkns_method followers --run_tag ratio_r04 --result_dir outputs/sensitivity/ratio

# Ranking-policy replacement: repeat with pagerank, personalized_pagerank,
# random_walk, degree, followers, betweenness, and closeness as supported.
python -m graphreason_bot.evaluation `
  --dataset Twibot-22 --data_source fixed_test --model_name gpt-4o-mini `
  --sampling_mode balanced --n_per_class 500 --neighbor_ratio 0.6 `
  --prkns_method degree --run_tag policy_degree --result_dir outputs/sensitivity/policy
```

Run `python -m graphreason_bot.evaluation --help` for the complete interface
and supported PRKNS policies.

## Matched prompt baselines

The following examples use the independent seeded-sample pathway. Add
`--max-users 10 --dry-run` first to validate a setup without provider calls.

```powershell
python -m graphreason_bot.baselines.random_twibot22 --help
python -m graphreason_bot.baselines.attention_botsim24 --help
python -m graphreason_bot.baselines.mrt --dataset all --model gpt-4o-mini
python -m graphreason_bot.baselines.cisc --dataset all --model gpt-4o-mini --traces 3
```

The vendor baselines used for broader paper comparisons are intentionally not
vendored. Record their upstream commit and their license in the experiment log
when reproducing them.
