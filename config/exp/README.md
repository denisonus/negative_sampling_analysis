# ML-100K Experiment Suite

This suite is the main rerun package for the thesis comparison on `ml-100k`.

Method benchmark:
- `10_ml100k_benchmark.yaml`

Main knob sweeps:
- explicit negatives: `20_ml100k_neg1.yaml`, `21_ml100k_neg10.yaml`
- logQ correction: `30_ml100k_logq_off.yaml`
- popularity smoothing: `64_ml100k_pop_smoothing_0p40.yaml`, `31_ml100k_pop_smoothing_0p50.yaml`, `65_ml100k_pop_smoothing_0p60.yaml`, `32_ml100k_pop_smoothing_1p00.yaml`
- hard-family candidate pool: `44_ml100k_hard_pool_25.yaml`, `43_ml100k_hard_pool_50.yaml`, `45_ml100k_hard_pool_75.yaml`, `10_ml100k_benchmark.yaml`, `40_ml100k_hard_pool_200.yaml`
- DNS temperature: `10_ml100k_benchmark.yaml`, `41_ml100k_dns_temp_0p20.yaml`
- curriculum schedule: `10_ml100k_benchmark.yaml`, `42_ml100k_curriculum_gentle.yaml`
- debiased `tau_plus`: `10_ml100k_benchmark.yaml`, `50_ml100k_debiased_tau_0p10.yaml`
- in-batch batch size: `10_ml100k_benchmark.yaml`, `60_ml100k_inbatch_bs1024.yaml`
- hard-uniform ratio: `61_ml100k_hard_uniform_ratio_0p25.yaml`, `10_ml100k_benchmark.yaml`, `62_ml100k_hard_uniform_ratio_0p75.yaml`
- in-batch + uniform index batch size: `10_ml100k_benchmark.yaml`, `72_ml100k_mns_index_768.yaml`, `70_ml100k_mns_index_1024.yaml`, `71_ml100k_mns_index_2048.yaml`
- in-batch + uniform batch interaction: `73_ml100k_mns_bs256_index1024.yaml`

Promising interaction follow-ups:
- `63_ml100k_hard_uniform_pool50_ratio0p75.yaml`
- `66_ml100k_hard_pool50_neg10.yaml`

Feature-aware comparisons:
- direct benchmark: `75_ml100k_feature_benchmark.yaml`
- hard-family smaller-pool check: `76_ml100k_feature_hard_pool_50.yaml`
- popularity best-point check: `77_ml100k_feature_pop_smoothing_0p50.yaml`
- in-batch + uniform index sweep: `78_ml100k_feature_mns_index_1024.yaml`, `79_ml100k_feature_mns_index_2048.yaml`
- hard-uniform best interaction: `80_ml100k_feature_hard_uniform_pool50_ratio0p75.yaml`
- in-batch + uniform batch interaction: `81_ml100k_feature_mns_bs256_index1024.yaml`

Legacy single feature-aware smoke config:
- `74_ml100k_feature_aware.yaml`
