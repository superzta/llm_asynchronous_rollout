# Per-config aggregates (seed-averaged)

| mode | staleness_k | decoupled | n_seeds | pass_rate | effective_tokens_per_sec | avg_tokens_per_sec | wall_clock_sec | accepted_fraction | mean_staleness |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| async_areal_style | 0 | 0 | 2 | 0.225 | 21.59 | 32.25 | 304.82 | 0.67 | 0.0 |
| async_areal_style | 0 | 1 | 2 | 0.1 | 21.56 | 32.21 | 303.33 | 0.67 | 0.0 |
| async_areal_style | 2 | 0 | 2 | 0.146 | 25.87 | 32.06 | 308.83 | 0.807 | 0.24 |
| async_areal_style | 2 | 1 | 2 | 0.042 | 25.74 | 31.9 | 311.03 | 0.807 | 0.24 |
| async_areal_style | 4 | 0 | 2 | 0.375 | 25.92 | 32.14 | 301.48 | 0.807 | 0.24 |
| async_areal_style | 4 | 1 | 2 | 0.083 | 25.9 | 32.1 | 308.96 | 0.807 | 0.24 |
| async_areal_style | 8 | 0 | 2 | 0.333 | 25.89 | 31.96 | 245.54 | 0.81 | 0.24 |
| async_areal_style | 8 | 1 | 2 | 0.042 | 25.99 | 32.22 | 308.1 | 0.807 | 0.24 |
| async_train | 0 | 0 | 2 | 0.2 | 28.49 | 35.52 | 261.66 | 0.802 | 0.0 |
| async_train | 0 | 1 | 2 | 0.025 | 28.07 | 34.99 | 271.62 | 0.802 | 0.0 |
| async_train | 2 | 0 | 2 | 0.271 | 34.74 | 34.74 | 226.25 | 1.0 | 0.24 |
| async_train | 2 | 1 | 2 | 0.042 | 34.91 | 34.91 | 265.59 | 1.0 | 0.24 |
| async_train | 4 | 0 | 2 | 0.271 | 34.06 | 34.06 | 231.29 | 1.0 | 0.24 |
| async_train | 4 | 1 | 2 | 0.042 | 34.53 | 34.53 | 268.53 | 1.0 | 0.24 |
| async_train | 8 | 0 | 2 | 0.271 | 29.9 | 29.9 | 272.01 | 1.0 | 0.24 |
| async_train | 8 | 1 | 2 | 0.042 | 30.24 | 30.24 | 306.64 | 1.0 | 0.24 |
| sync_train | 0 | 1 | 2 | 0.042 | 36.16 | 36.16 | 265.45 | 1.0 | 0.0 |
