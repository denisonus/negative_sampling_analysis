# Two-Tower Negative Sampling Experiments

This repository compares negative sampling strategies for a two-tower recommender model.

It includes:
- a two-tower model in `models/two_tower.py`
- multiple negative samplers in `samplers/`
- experiment runner in `run_experiments.py`
- result analysis facade in `analysis.py` with implementation in `_analysis/`

Supported strategies:
- `uniform`
- `popularity`
- `hard`
- `mixed_hard_uniform`
- `mixed_in_batch_uniform`
- `in_batch`
- `dns`
- `curriculum`
- `debiased`

Supported datasets:
- `ml-100k`
- `gowalla-1m`

Configs are compact: each YAML file declares a dataset and only the values that
override the code defaults for that dataset.

## Main Files

```text
config/              Experiment configs
models/              Two-tower model
samplers/            Negative sampling strategies
utils/               Data loading and training code
evaluation/          Ranking metrics
run_experiments.py   Main entry point
analysis.py          Analysis facade and CLI
_analysis/           Plot, table, sweep, and report implementation
```
