 You are working in llm_async_project on PSC.

 Goal:
 Diagnose why the new AReaL-style runner is not showing meaningful staleness-sensitive behavior, fix it, and rerun the experiments.

 Current context:
 - We have three modes in the experiment sweep:
 - sync_train
 - async_train
 - async_areal_style
 - The older async_train baseline behaves as expected:
 - dropped stale count decreases as staleness_k increases
 - accepted fraction increases as staleness_k increases
 - mean staleness increases as staleness_k increases
 - But async_areal_style is suspiciously flat across k:
 - dropped_stale_count is basically 0 for all k
 - mean_staleness is almost always 0
 - pass_rate is stuck around 0.2
 - update_count is almost always 5
 - wall_clock is almost always around 1.4–1.5 sec
 - This strongly suggests the AReaL-style path is not creating enough real overlap, interrupt activity, or policy-version skew to expose the systems tradeoff we want to measure.

 This matters because our project is specifically about reproducing an asynchronous rollout design with:
 - decoupled rollout and training
 - shared buffer / replay behavior
 - interruptible rollout workers
 - bounded staleness
 - throughput vs stability tradeoff

 Files you should inspect first:
 - src/run_async_areal_style.py
 - src/areal_runtime.py
 - src/areal_parameter_service.py
 - src/areal_controller.py
 - src/areal_rollout_worker.py
 - src/areal_trainer_worker.py
 - src/run_async_baseline.py
 - src/model_backends.py
 - scripts/run_experiments.sh
 - scripts/analyze_experiments.sh
 - results/experiments/<latest>/analysis_aggregated.json
 - README_async_areal_style.md

 Diagnosis tasks:
 1. Reproduce the issue locally from the latest experiment directory.
 2. Compare async_train vs async_areal_style behavior and identify why async_areal_style is not reacting to staleness_k.
 3. Add instrumentation to trace:
 - when rollout workers load policy versions
 - when interrupts are triggered
 - when interrupts are observed
 - when a rollout sample is interrupted
 - when a rollout sample is emitted
 - current global policy_version over time
 - replay buffer depth over time
 - trainer batch dispatch timing
 - trainer update commit timing
 - per-worker generated count and interrupted count
 4. Check whether rollout and trainer are truly overlapping in time.
 5. Check whether trainer commits happen too late, too rarely, or too cleanly relative to rollout generation.
 6. Check whether rollout workers refresh weights too early, too often, or only between samples, which could eliminate observable staleness.
 7. Check whether the controller computes staleness at the wrong point in the pipeline.
 8. Check whether accepted/dropped accounting is masking interrupted samples or stale samples.

 Very likely failure modes to investigate:
 - rollout workers generate too fast relative to trainer updates, so samples finish before any useful version skew appears
 - rollout workers only reload at sample boundaries, so “interruptibility” exists in code but is behaviorally inactive
 - generation chunking is too coarse or too small in the wrong way
 - parameter version is not changing often enough during active rollouts
 - staleness is computed against the wrong version or at the wrong time
 - replay buffer dispatch is too immediate or too serialized
 - trainer queue never becomes a bottleneck, so no real async pressure is created

 What I want you to change:

 A. Make async_areal_style produce real pressure
 - Add or tune stress parameters so the AReaL-style runner can create actual overlap:
 - rollout generation chunk delay
 - interrupt polling interval
 - trainer/learner delay
 - replay dispatch delay if needed
 - number of tasks or epochs
 - Ensure at least one default stress configuration makes interrupts and nonzero staleness observable.

 B. Make interruptibility behavior measurable
 Add summary/output fields:
 - interrupted_samples_count
 - interrupt_count_total
 - per_rollout_worker_interrupt_count
 - per_rollout_worker_loaded_versions
 - version_change_events
 - trainer_commit_timestamps
 - rollout_emit_timestamps
 - staleness_histogram

 C. Fix logic bugs if found
 If the issue is logic, fix it directly.
 Examples:
 - wrong staleness computation point
 - worker not reloading newest state
 - interrupt flag never acted on
 - interrupted sample still treated as clean accepted sample
 - policy version not exposed correctly to workers

 D. Preserve existing runners
 - Do NOT break sync_train
 - Do NOT break async_train
 - Keep async_areal_style as the architecture-focused path

 E. Make the experiment suite capable of showing the difference
 Update scripts/run_experiments.sh and related experiment code so that:
 - default experiment sweep can still run PSC-friendly mode
 - there is also a stress-mode sweep specifically for async_areal_style
 - stress-mode sweep should be small but sufficient to expose:
 - nonzero interrupt counts
 - nonzero mean staleness
 - staleness_k-sensitive drop behavior

 Concrete target behavior after the fix:
 - async_areal_style should not remain flat across all k
 - lower k should usually drop more stale samples than higher k
 - mean/max staleness should become nontrivial under stress
 - interrupt counts should become nonzero in the stress setting
 - queue / replay traces should show real overlap

 New scripts / commands to add:
 1. A focused debug run script:
 - scripts/run_async_areal_style_debug.sh
 - should run one short but strongly instrumented configuration

 2. A stress run script:
 - scripts/run_async_areal_style_stress.sh
 - should intentionally create overlap pressure and interrupt events

 3. Optional:
 - scripts/run_experiments_stress.sh
 - a smaller sweep over async_areal_style only, varying k under stress

 Output requirements:
 - print a short root-cause summary
 - list the exact code changes made
 - rerun the relevant experiments
 - regenerate analysis outputs and plots
 - explicitly compare before vs after for async_areal_style
 - say whether the fixed runner now behaves in a staleness-sensitive manner

 Acceptance criteria:
 1. You identify the root cause or the main reasons async_areal_style was flat.
 2. You fix the issue or tune the architecture so that interrupts / overlap / staleness become behaviorally visible.
 3. You rerun the experiments.
 4. The new analysis shows async_areal_style responding meaningfully to staleness_k under at least one stress configuration.
 5. Existing sync_train and async_train still run.

 After making changes, run these:
 - the focused debug run
 - the async_areal_style stress run
 - a compact stress sweep across k values
 - regenerate analysis/plots

 At the end, give me:
 - the root cause
 - files changed
 - commands run
 - before/after metric comparison
 - whether the results are now aligned with the project goal