# Two-Tower Negative Sampling Experiments

This repository compares negative sampling strategies for a two-tower recommender model.

It includes:
- a two-tower model in `models/two_tower.py`
- multiple negative samplers in `samplers/`
- experiment runner in `run_experiments.py`
- result analysis in `analysis/`

Supported strategies:
- `uniform`
- `popularity`
- `hard`
- `mixed_hard_uniform`
- `mixed_in_batch_uniform`
- `in_batch`
- `dns`
- `curriculum`

Supported datasets:
- `ml-100k`
- `ml-1m`
- `gowalla-1m`

MovieLens datasets are read from the official files (`u.data` for ML-100K and
`ratings.dat` for ML-1M). Gowalla is read from raw LightGCN files under
`dataset/gowalla-1m/raw-lightgcn`.

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
analysis/            Result analysis, plots, tables, sweeps
```
