# Report & poster asset index

Experiment root: `results/experiments/final_report_fixed_20260421_154322/gsm8k`

## How to use these in the final report

| Section of report | Asset |
|---|---|
| Headline teaser figure | `plots/fig_overview_poster.png` |
| Section "Learning under staleness" | `plots/fig5a_naive_learning_curves.png`, `plots/fig5b_decoupled_learning_curves.png` |
| Section "Throughput vs staleness" | `plots/fig5c_throughput_vs_staleness.png` |
| Section "Interruptible generation" | `plots/fig6b_interruptible_throughput.png` |
| Section "End-to-end comparison"  | `plots/fig_reward_curves.png`, `plots/fig_pass_rate_curves.png` |
| Section "Speedup analysis" | `plots/fig_time_to_threshold.png`, `plots/fig_updates_per_sec.png` |
| Section "Throughput/accuracy tradeoff" | `plots/fig_pareto.png` |
| Section "Sample efficiency" | `plots/fig_reward_per_ktoken.png` |
| Section "Realized staleness" | `plots/fig_staleness_violin.png` |
| Appendix: full results table | `tables/summary_headline.csv/.md` |
| Appendix: per-config aggregates | `tables/mode_aggregate.csv/.md` |
| Paper-style Table 2 reproduction | `plots/table2_staleness_vs_pass_rate.csv/.md` |

## How to use these on the poster

- **Main panel (top):** `plots/fig_overview_poster.png` (4-in-1 figure, publication-ready).
- **Secondary panel (bottom-left):** `plots/fig5b_decoupled_learning_curves.png` — shows AReaL tolerates staleness when paired with the decoupled objective.
- **Secondary panel (bottom-right):** `plots/fig5c_throughput_vs_staleness.png` — shows throughput scales with allowed staleness.
- **Sidebar stat:** the speedup ratio can be read directly from `plots/fig_time_to_threshold.png`.

## Raw inputs
- `manifest.json` — grid configuration that produced these runs.
- `merged_summary.{json,csv}` — every run summary concatenated.
- `failures.json` — any runs that did not produce a `summary.json`.
- per-run directories hold `summary.json`, `results.jsonl`, `command.txt`, `config.json`, `stdout.log`, `stderr.log`.
