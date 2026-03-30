# Two-Tower Negative Sampling Experiments

This repository compares negative sampling strategies for a two-tower recommender model.

It includes:
- a two-tower model in `models/two_tower.py`
- multiple negative samplers in `samplers/`
- experiment runner in `run_experiments.py`
- result plots in `analysis.py`

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

## Main Files

```text
config/              Experiment configs
models/              Two-tower model
samplers/            Negative sampling strategies
utils/               Data loading and training code
evaluation/          Ranking metrics
run_experiments.py   Main entry point
analysis.py          Plotting and comparison
```
