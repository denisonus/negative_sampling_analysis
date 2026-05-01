# Experiment Suites

Each YAML file is compact: a dataset name plus only values that differ from the
dataset profile in code. Use `--strategies` to keep sweeps focused; otherwise a
single config runs every implemented sampler.

## ML-100K

Location: `config/exp/ml100k/`

This suite is the main rerun package for the thesis comparison on `ml-100k`.
Each YAML file is `dataset: ml-100k` plus only values that differ from the
default ML-100K profile.

Method benchmark:
- `ml100k/10_ml100k_benchmark.yaml`

Main knob sweeps:
- explicit negatives: `ml100k/20_ml100k_neg1.yaml`, `ml100k/21_ml100k_neg10.yaml`
- logQ correction: `ml100k/30_ml100k_logq_off.yaml`
- popularity smoothing: `ml100k/64_ml100k_pop_smoothing_0p40.yaml`, `ml100k/31_ml100k_pop_smoothing_0p50.yaml`, `ml100k/65_ml100k_pop_smoothing_0p60.yaml`, `ml100k/32_ml100k_pop_smoothing_1p00.yaml`
- hard-family candidate pool: `ml100k/44_ml100k_hard_pool_25.yaml`, `ml100k/43_ml100k_hard_pool_50.yaml`, `ml100k/45_ml100k_hard_pool_75.yaml`, `ml100k/10_ml100k_benchmark.yaml`, `ml100k/40_ml100k_hard_pool_200.yaml`
- DNS temperature: `ml100k/10_ml100k_benchmark.yaml`, `ml100k/41_ml100k_dns_temp_0p20.yaml`
- curriculum schedule: `ml100k/10_ml100k_benchmark.yaml`, `ml100k/42_ml100k_curriculum_gentle.yaml`
- debiased `tau_plus`: `ml100k/10_ml100k_benchmark.yaml`, `ml100k/50_ml100k_debiased_tau_0p10.yaml`
- in-batch batch size: `ml100k/10_ml100k_benchmark.yaml`, `ml100k/60_ml100k_inbatch_bs1024.yaml`
- hard-uniform ratio: `ml100k/61_ml100k_hard_uniform_ratio_0p25.yaml`, `ml100k/10_ml100k_benchmark.yaml`, `ml100k/62_ml100k_hard_uniform_ratio_0p75.yaml`
- in-batch + uniform index batch size: `ml100k/10_ml100k_benchmark.yaml`, `ml100k/72_ml100k_mns_index_768.yaml`, `ml100k/70_ml100k_mns_index_1024.yaml`, `ml100k/71_ml100k_mns_index_2048.yaml`
- in-batch + uniform batch interaction: `ml100k/73_ml100k_mns_bs256_index1024.yaml`

Promising interaction follow-ups:
- `ml100k/63_ml100k_hard_uniform_pool50_ratio0p75.yaml`
- `ml100k/66_ml100k_hard_pool50_neg10.yaml`

Feature-aware comparisons:
- direct benchmark: `ml100k/75_ml100k_feature_benchmark.yaml`
- hard-family smaller-pool check: `ml100k/76_ml100k_feature_hard_pool_50.yaml`
- popularity best-point check: `ml100k/77_ml100k_feature_pop_smoothing_0p50.yaml`
- in-batch + uniform index sweep: `ml100k/78_ml100k_feature_mns_index_1024.yaml`, `ml100k/79_ml100k_feature_mns_index_2048.yaml`
- hard-uniform best interaction: `ml100k/80_ml100k_feature_hard_uniform_pool50_ratio0p75.yaml`
- in-batch + uniform batch interaction: `ml100k/81_ml100k_feature_mns_bs256_index1024.yaml`

## Gowalla-1M

Location: `config/exp/gowalla/`

The Gowalla suite mirrors the ML-100K decade-based numbering scheme. It uses
the benchmark split in `dataset/gowalla-1m`, implicit feedback, `topk: [20,
50]`, `NDCG@20`, 30 epochs, and patience 5 from the `gowalla-1m` profile.

Numbering groups (same logic as ML-100K):
- `1x` — benchmark
- `2x` — negative count
- `3x` — popularity / logQ
- `4x` — hard / DNS / curriculum
- `5x` — debiased
- `6x` — ratio / mixed
- `7x` — MNS (mixed in-batch + uniform index)

Core benchmark:
- `gowalla/10_benchmark.yaml`

Paper-linked sweeps:
- BPR/DNS-style single negative: `gowalla/20_neg1_bpr_like.yaml`
- larger sampled-softmax negative set: `gowalla/21_neg10.yaml`
- logQ correction: `gowalla/10_benchmark.yaml`, `gowalla/30_logq_off.yaml`
- popularity smoothing: `gowalla/31_pop_smoothing_0p50.yaml`, `gowalla/10_benchmark.yaml`, `gowalla/32_pop_smoothing_1p00.yaml`
- hard/DNS/curriculum candidate pool: `gowalla/40_hard_pool_100.yaml`, `gowalla/10_benchmark.yaml`, `gowalla/41_hard_pool_600.yaml`
  - Note: `40_hard_pool_100` tests the common default pool on the larger dataset
    (Gowalla preset uses 300).
- DNS temperature: `gowalla/10_benchmark.yaml`, `gowalla/42_dns_temp_0p20.yaml`
- curriculum schedule: `gowalla/10_benchmark.yaml`, `gowalla/43_curriculum_gentle.yaml`
- hard-uniform ratio: `gowalla/10_benchmark.yaml`, `gowalla/60_hard_uniform_ratio_0p75.yaml`
- in-batch + uniform index size: `gowalla/10_benchmark.yaml`, `gowalla/70_mns_index_2048.yaml`
- debiased `tau_plus`: `gowalla/10_benchmark.yaml`, `gowalla/50_debiased_tau_0p10.yaml`

Suggested limited-time order:
- Run the benchmark once with all strategies.
- If time is tight after that, prioritize `gowalla/30_logq_off.yaml`, `gowalla/40_hard_pool_100.yaml`, `gowalla/41_hard_pool_600.yaml`, `gowalla/42_dns_temp_0p20.yaml`, and `gowalla/70_mns_index_2048.yaml`.
- For sweep configs, pass only the affected strategy family, for example
  `--strategies hard dns curriculum` for candidate-pool configs, `--strategies
  popularity` for smoothing, and `--strategies mixed_in_batch_uniform` for MNS.
