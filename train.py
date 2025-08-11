"""Command-line entry point for running configurable training simulations."""

import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.utils.config import load_config
from src.data.datasets import get_dataloaders
from src.models.mlp import MLP
from src.models.cnn import SimpleCNN
from src.models.resnet import ResNet50

from src.utils.init import initialize_weights
from src.utils.metrics import evaluate, compute_static_overlap


def get_evaluation_steps(cfg_eval, total_steps):
    """Return a sorted list of training steps where evaluation is performed."""
    steps = cfg_eval.get('steps')
    if steps is None:
        num_points = cfg_eval.get('num_points', 10)
        # Generate logarithmically spaced evaluation steps
        steps = np.unique(
            np.logspace(0, np.log10(total_steps), num=num_points, dtype=int)
        ).tolist()
    # Always include step 0 (initialization)
    steps = sorted(set([0] + steps))
    return steps


def build_model(cfg_model, input_shape, num_classes):
    """Build a neural network based on the configuration dictionary."""
    model_type = cfg_model['type'].lower()
    if model_type == 'mlp':

        input_dim = int(np.prod(input_shape))
        model = MLP(input_dim=input_dim,
                    num_classes=num_classes,
                    depth=cfg_model['depth'],
                    width=cfg_model['width'],
                    activation=cfg_model.get('activation', 'relu'))
    elif model_type == 'cnn':

        model = SimpleCNN(in_channels=input_shape[0],
                          input_height=input_shape[1],
                          input_width=input_shape[2],
                          num_classes=num_classes,
                          depth=cfg_model['depth'],
                          width=cfg_model['width'],
                          activation=cfg_model.get('activation', 'relu'),
                          pooling=cfg_model.get('pooling', 'max'))
        
    elif model_type == 'resnet':
        # ResNet-50 optionally adapted for small inputs (e.g., 32×32 images)
        model = ResNet50(num_classes=num_classes,
                         small_input=cfg_model.get('small_input', True))

    else:
        raise ValueError(f"Unknown model type {cfg_model['type']}")

    # Apply requested weight initialization
    initialize_weights(model, cfg_model.get('init', 'kaiming_normal'))
    return model


def main():
    """Parse arguments, run training for the desired number of runs and save metrics."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Prepare data loaders and compute evaluation schedule
    train_loader, test_loader, train_dataset, input_shape, num_classes = get_dataloaders(cfg)
    total_steps = cfg['training']['epochs'] * len(train_loader)
    eval_steps = get_evaluation_steps(cfg['evaluation'], total_steps)
    criterion = nn.CrossEntropyLoss()

    # Pre-compute class overlap using raw inputs
    static_overlap = compute_static_overlap(train_dataset, device, num_classes)

    os.makedirs('results', exist_ok=True)

    for run in range(cfg.get('runs', 1)):
        # Build a fresh model for each run to ensure different initializations
        model = build_model(cfg['model'], input_shape, num_classes).to(device)

        optimizer_name = cfg['training'].get('optimizer', 'sgd').lower()
        if optimizer_name == 'sgd':
            optimizer = optim.SGD(model.parameters(), lr=cfg['training']['lr'])
        elif optimizer_name == 'adam':
            optimizer = optim.Adam(model.parameters(), lr=cfg['training']['lr'])
        else:
            raise ValueError(f"Unknown optimizer {optimizer_name}")

        # Containers for metrics collected at each evaluation step
        metrics = {
            'steps': [],
            'loss': [],
            'accuracy': [],
            'loss_per_class': [],
            'accuracy_per_class': [],
            'overlap_dynamic': [],
            'overlap_static': static_overlap.numpy(),
        }

        # Evaluate at initialization (step 0)
        step = 0
        eval_result = evaluate(model, test_loader, device, num_classes, criterion)
        metrics['steps'].append(step)
        metrics['loss'].append(eval_result['loss'])
        metrics['accuracy'].append(eval_result['acc'])
        metrics['loss_per_class'].append(eval_result['loss_per_class'].numpy())
        metrics['accuracy_per_class'].append(eval_result['acc_per_class'].numpy())
        metrics['overlap_dynamic'].append(eval_result['overlap'].numpy())

        for epoch in range(cfg['training']['epochs']):
            model.train()
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                step += 1

                # Perform evaluation whenever the current step matches schedule
                if step in eval_steps:
                    eval_result = evaluate(model, test_loader, device, num_classes, criterion)
                    metrics['steps'].append(step)
                    metrics['loss'].append(eval_result['loss'])
                    metrics['accuracy'].append(eval_result['acc'])
                    metrics['loss_per_class'].append(eval_result['loss_per_class'].numpy())
                    metrics['accuracy_per_class'].append(eval_result['acc_per_class'].numpy())
                    metrics['overlap_dynamic'].append(eval_result['overlap'].numpy())

        # Save metrics for this run
        run_dir = os.path.join('results', f'run_{run}')
        os.makedirs(run_dir, exist_ok=True)
        np.savez(os.path.join(run_dir, 'metrics.npz'),
                 steps=np.array(metrics['steps']),
                 loss=np.array(metrics['loss']),
                 accuracy=np.array(metrics['accuracy']),
                 loss_per_class=np.stack(metrics['loss_per_class']),
                 accuracy_per_class=np.stack(metrics['accuracy_per_class']),
                 overlap_dynamic=np.stack(metrics['overlap_dynamic']),
                 overlap_static=metrics['overlap_static'])


def entrypoint():
    """Entry point used by external scripts."""
    main()


if __name__ == '__main__':
    main()
