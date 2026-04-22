# Final report assets: final_report_8h_interactive_20260421_195453

Generated Tue Apr 21 23:12:26 EDT 2026.

## Datasets included
- **gsm8k**: [INDEX](coding/INDEX.md) | [plots](gsm8k/plots/) | [tables](coding/tables/)
- **gsm8k**: [INDEX](coding/INDEX.md) | [plots](gsm8k/plots/) | [tables](coding/tables/)

## Headline assets
For the main body of the report and the poster, use GSM8K as the headline dataset:
- Poster panel: `gsm8k/plots/fig_overview_poster.png`
- Learning curves: `gsm8k/plots/fig_reward_curves.png`, `gsm8k/plots/fig_pass_rate_curves.png`
- Speedup bar: `gsm8k/plots/fig_time_to_threshold.png`
- Pareto: `gsm8k/plots/fig_pareto.png`
- AReaL Fig 5b (staleness tolerance with decoupled objective): `gsm8k/plots/fig5b_decoupled_learning_curves.png`
- AReaL Fig 5c (throughput vs staleness): `gsm8k/plots/fig5c_throughput_vs_staleness.png`

## Secondary (task-robustness) assets
To support the claim that the ordering generalizes beyond GSM8K, cite the matching coding plots:
- `coding/plots/fig_reward_curves.png`
- `coding/plots/fig_pareto.png`
- `coding/tables/mode_aggregate.md`

## Raw inputs
Under each dataset sub-dir:
- `sync/` -- runs produced by the compact sync pass
- `async/` -- runs produced by the async sweep
- Each run dir has `summary.json`, `results.jsonl`, `config.json`, `stdout.log`, `stderr.log`.
