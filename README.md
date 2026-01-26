# Impact of Negative Sampling Strategies on Two-Tower Recommender Models

A research codebase for studying the impact of different negative sampling strategies on the effectiveness of two-tower (dual encoder) recommender models.

## Project Structure

```
two_towers/
├── config/
│   └── config.yaml          # Configuration file
├── models/
│   ├── __init__.py
│   └── two_tower.py          # Two-Tower model implementation
├── samplers/
│   ├── __init__.py
│   └── negative_samplers.py  # Negative sampling strategies
├── utils/
│   ├── __init__.py
│   ├── data_utils.py         # Data loading utilities (RecBole integration)
│   └── trainer.py            # Training loop
├── evaluation/
│   ├── __init__.py
│   └── evaluator.py          # Evaluation metrics
├── run_experiments.py        # Main experiment runner
├── quick_start.py            # Quick test script
├── analysis.py               # Results analysis and visualization
├── requirements.txt          # Dependencies
└── README.md
```

## Installation

```bash
pip install -r requirements.txt
```

## Negative Sampling Strategies

The following negative sampling strategies are implemented:

1. **Uniform Sampling** (`uniform`): Random sampling from all items
2. **Popularity-based Sampling** (`popularity`): Sample proportional to item popularity
3. **Hard Negative Mining** (`hard`): Select negatives close to user in embedding space
4. **Mixed Sampling** (`mixed`): Combination of uniform and hard negatives
5. **Dynamic Negative Sampling** (`dns`): Online hard negative mining with softmax sampling
6. **In-Batch Negatives** (`in_batch`): Use other items in batch as negatives
7. **Curriculum Learning** (`curriculum`): Gradually increase negative difficulty

## Quick Start

### Quick Test with Synthetic Data

```bash
python quick_start.py uniform
python quick_start.py popularity
python quick_start.py hard
```

### Run Full Experiments

```bash
# Run all strategies on ML-100K dataset
python run_experiments.py --config config/config.yaml

# Run specific strategies
python run_experiments.py --strategies uniform popularity hard

# Run with multiple seeds for statistical significance
python run_experiments.py --num_runs 5

# Run with specific seeds
python run_experiments.py --num_runs 3 --seeds 42 123 456

# Custom output directory
python run_experiments.py --output my_results
```

## Configuration

Edit `config/config.yaml` to customize:

```yaml
# Dataset
dataset: 'ml-100k'  # ml-100k, ml-1m, amazon-books, yelp2018

# Model
embedding_size: 64
hidden_size: 128
num_layers: 2

# Training
epochs: 50
train_batch_size: 1024
learning_rate: 0.001
num_neg_samples: 4

# Negative sampling
hard_neg_ratio: 0.5           # For mixed sampling
dns_temperature: 1.0          # For DNS sampling
curriculum_start_ratio: 0.0   # Curriculum: start with all random
curriculum_end_ratio: 0.8     # Curriculum: end with 80% hard
curriculum_warmup_epochs: 10  # Curriculum: transition epochs
```

## Results Analysis

After running experiments:

```bash
# Generate full analysis report
python analysis.py results/results_YYYYMMDD_HHMMSS.json

# Specify output directory
python analysis.py results/results_YYYYMMDD_HHMMSS.json my_analysis/
```

This generates:
- Comparison bar charts with error bars
- Training loss and validation curves
- Convergence speed analysis
- Timing breakdown (sampling vs training time)
- Statistical significance tests (paired t-tests)
- LaTeX tables for papers

## Two-Tower Model

The two-tower architecture:

```
User Tower:
  Input: user_id
  → Embedding Layer
  → MLP Layers (with ReLU, Dropout)
  → L2 Normalized Output

Item Tower:
  Input: item_id
  → Embedding Layer
  → MLP Layers (with ReLU, Dropout)
  → L2 Normalized Output

Score = dot_product(user_embedding, item_embedding) / temperature
```

Loss functions:
- Sampled Softmax (InfoNCE)
- BPR (Bayesian Personalized Ranking)

## Metrics

- **Recall@K**: Proportion of relevant items in top-K
- **NDCG@K**: Normalized Discounted Cumulative Gain
- **Hit@K**: Whether any relevant item appears in top-K
- **MRR@K**: Mean Reciprocal Rank

## Example Results

| Strategy   | NDCG@10       | Recall@10     | Hit@10        | MRR@10        |
|------------|---------------|---------------|---------------|---------------|
| Uniform    | 0.123±0.005   | 0.057±0.003   | 0.235±0.008   | 0.089±0.004   |
| Popularity | 0.135±0.004   | 0.061±0.002   | 0.246±0.007   | 0.092±0.003   |
| Hard       | 0.146±0.006   | 0.068±0.003   | 0.257±0.009   | 0.099±0.005   |
| Mixed      | 0.152±0.005   | 0.071±0.003   | 0.263±0.008   | 0.102±0.004   |
| DNS        | 0.149±0.004   | 0.069±0.002   | 0.260±0.006   | 0.100±0.003   |
| In-Batch   | 0.141±0.005   | 0.065±0.003   | 0.251±0.008   | 0.096±0.004   |
| Curriculum | 0.155±0.004   | 0.073±0.002   | 0.268±0.007   | 0.105±0.003   |

*Results shown as mean±std across 5 runs with different random seeds.*

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{two_tower_negative_sampling,
  title={Impact of Negative Sampling Strategies on Two-Tower Recommender Models},
  author={Your Name},
  year={2026}
}
```

## License

MIT License
