# Class-Hardness Simulation Repository

This repository provides a flexible framework for running simulations on deep neural networks using [PyTorch](https://pytorch.org/). Models, datasets, and training procedures are configured through external configuration files, enabling easy experimentation with different architectures and hyperparameters.

## Repository Structure

```
configs/                # YAML configuration files
src/                    # Library code
  data/                 # Dataset utilities and preprocessing
  models/               # Implementations of MLP and CNN architectures
  utils/                # Initialization and metric utilities
results/                # Output metrics for each run (generated)
train.py                # Entry point for running simulations
README.md               # This file
```

## Configuration

Simulation parameters are specified in YAML files under `configs/`. The configuration controls:

- **Dataset**: `mnist`, `cifar10`, `cifar100`, or `gaussian` synthetic data. Datasets can be
  restricted to a subset of classes by providing a `class_map` dictionary mapping original
  labels to new labels. Classes not listed are discarded.
- **Model**:
  - `type`: `mlp` or `cnn`
  - `depth` and `width`
  - `activation`: `relu` or `tanh`
  - `pooling` (CNN only): `max` or `avg`
  - `init`: weight initialization (`kaiming_normal`, `kaiming_uniform`)
- **Training**:
  - `epochs`, `batch_size`, `lr`, `optimizer` (`sgd` or `adam`)
- **Runs**: number of random initializations to repeat
- **Evaluation**:
  - `steps`: explicit list of evaluation steps **or**
  - `num_points`: number of logarithmically spaced checkpoints

The file `configs/gaussian_example.yaml` provides a minimal example with a synthetic dataset.

### Class Subselection Example

To run experiments on a subset of classes, include a `class_map` in the dataset
configuration. The following snippet keeps only digits `0` and `1` from MNIST
and remaps them to labels `0` and `1`:

```yaml
dataset:
  name: mnist
  class_map:
    0: 0
    1: 1
```

Any classes not listed in the mapping are ignored.

## Metrics

At each evaluation step, the following metrics are recorded:

- Global and per-class loss
- Global and per-class accuracy
- Overlap matrix between class representations in the last hidden layer
- Static input overlap between class-mean inputs (computed once)

Metrics are saved in `results/run_<id>/metrics.npz` as NumPy arrays. The saved file contains:

- `steps`: evaluation step indices
- `loss`, `accuracy`
- `loss_per_class`, `accuracy_per_class`
- `overlap_dynamic`: class-overlap matrices over time
- `overlap_static`: class-overlap matrix at initialization

## Running a Simulation

```bash
python train.py --config configs/gaussian_example.yaml
```

Replace the configuration file with your own to run different experiments.

## Extending

New architectures, datasets, or metrics can be added by implementing new modules under `src/` and referencing them from configuration files.
