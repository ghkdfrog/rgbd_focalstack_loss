"""
Default configuration and argparse definitions for Gradient Matching training.
"""

import argparse


def get_parser():
    parser = argparse.ArgumentParser(
        description='Gradient Matching Score Model (SimpleCNN)'
    )

    # Data
    parser.add_argument('--data_dir', type=str, default='../varifocal/data',
                        help='Path to raw EXR data (varifocal/data)')
    parser.add_argument('--generated_data_dir', type=str, default=None,
                        help='Path to generated pred frames (default: ./data)')
    parser.add_argument('--output_dir', type=str, default='runs_gm',
                        help='Root directory for run outputs')

    # Model
    parser.add_argument('--diopter_mode', type=str, default='coc',
                        choices=['spatial', 'coc'],
                        help='Diopter conditioning mode')
    parser.add_argument('--energy_head', type=str, default='fc',
                        choices=['fc', 'conv1x1'],
                        help='Energy output head: fc (Linear 67M, 512x512 only) or conv1x1 (Conv2d 1x1 + sum, resolution-free)')

    # Training
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--num_workers', type=int, default=4)

    # Gradient Matching
    parser.add_argument('--gm_steps', type=int, default=50,
                        help='Number of trajectory steps per sample')
    parser.add_argument('--gm_step_size', type=float, default=0.2,
                        help='Langevin step size (eta_max when using schedule)')
    parser.add_argument('--eta_schedule', type=str, default='constant',
                        choices=['constant', 'cosine', 'linear'],
                        help='Step size schedule: constant (default), cosine, or linear decay')
    parser.add_argument('--eta_min', type=float, default=0.002,
                        help='Minimum step size for eta schedule (ignored when constant)')
    parser.add_argument('--langevin_noise', action='store_true',
                        help='Add Langevin noise term sqrt(2*eta)*z at each step')

    # Dataset
    parser.add_argument('--single_scene_only', action='store_true',
                        help='Only use scene 0 for quick prototyping')
    parser.add_argument('--unmatch_ratio', type=int, default=0,
                        help='Unmatch samples per match (0 = match only)')

    # Checkpointing
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--save_every', type=int, default=5,
                        help='Save checkpoint every N epochs (0 = disable)')

    # Inference (used by infer.py)
    parser.add_argument('--run_dir', type=str, default=None,
                        help='Path to a specific run directory (for inference)')
    parser.add_argument('--scene_idx', type=int, default=0,
                        help='Scene index for inference')
    parser.add_argument('--plane_idx', type=str, default='20',
                        help='Focal plane indices, comma-separated (e.g. 0,20,39 or -1 for all)')

    return parser


def parse_args():
    return get_parser().parse_args()
