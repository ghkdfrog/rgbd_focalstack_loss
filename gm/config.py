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
    parser.add_argument('--arch', type=str, default='simple',
                        choices=['simple', 'deep', 'stride', 'resnet', 'resnet_film', 'resunet', 'convnext', 'convnext_unet', 'dilated', 'interleave_resnet'],
                        help='Model architecture')
    parser.add_argument('--diopter_mode', type=str, default='coc',
                        choices=['spatial', 'coc', 'coc_abs', 'coc_signed'],
                        help='Diopter conditioning mode')
    parser.add_argument('--energy_head', type=str, default='fc',
                        choices=['fc', 'conv1x1'],
                        help='Energy output head: fc (Linear 67M, 512x512 only) or conv1x1 (Conv2d 1x1 + sum, resolution-free)')
    parser.add_argument('--channels', type=int, default=256,
                        help='Number of channels in the model')
    parser.add_argument('--use_film', action='store_true',
                        help='Enable FiLM conditioning on residual/ConvNeXt blocks (resnet, convnext, convnext_unet)')
    parser.add_argument('--long_skip', action='store_true',
                        help='Enable long skip connection from conv_expand to energy head (resnet only)')
    parser.add_argument('--interleave_rate', type=int, default=2,
                        help='Pixel shuffle interleave rate for interleave_resnet (default: 2)')
    parser.add_argument('--sharp_prior', action='store_true',
                        help='Enable Sharpness Prior for in-focus regions (resnet_film only)')
    parser.add_argument('--sharp_lambda_learnable', action='store_true',
                        help='Make sharp_lambda learnable (nn.Parameter). Default: fixed')
    parser.add_argument('--sharp_gamma_learnable', action='store_true',
                        help='Make sharp_gamma learnable (nn.Parameter). Default: fixed')
    parser.add_argument('--sharp_lambda', type=float, default=10.0,
                        help='Sharp Prior lambda (strength). Default: 10.0')
    parser.add_argument('--sharp_gamma', type=float, default=30.0,
                        help='Sharp Prior gamma (CoC decay rate). Default: 30.0')
    parser.add_argument('--activation', type=str, default='relu',
                        choices=['relu', 'silu'],
                        help='Activation function: relu (default) or silu')

    # Execution optimization
    parser.add_argument('--compile', action='store_true',
                        help='Enable torch.compile for model')
    parser.add_argument('--compile_mode', type=str, default='reduce-overhead',
                        choices=['default', 'reduce-overhead', 'max-autotune'],
                        help='torch.compile mode')
    parser.add_argument('--amp', action='store_true',
                        help='Enable Automatic Mixed Precision (AMP)')

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
                        help='Add Langevin noise term at each step')
    parser.add_argument('--noise_method', type=str, default='constant_scale',
                        choices=['constant_scale'],
                        help='Langevin noise method: constant_scale (noise = scale * eta * z)')
    parser.add_argument('--noise_scale', type=float, default=0.1,
                        help='Noise scale factor for constant_scale method (default: 0.1)')

    # Training-time Bypass (Stop-Gradient Sharp Prior)
    parser.add_argument('--train_bypass', action='store_true',
                        help='Enable stop-gradient bypass during training (analytical sharp prior gradient, no autograd graph)')
    parser.add_argument('--bypass_lambda', type=float, default=5.0,
                        help='Bypass sharp prior strength (fixed, not learned). Default: 5.0')
    parser.add_argument('--bypass_gamma', type=float, default=30.0,
                        help='Bypass CoC decay rate (fixed, not learned). Default: 30.0')
    parser.add_argument('--bypass_warmup', type=int, default=0,
                        help='Number of epochs before bypass starts (default: 0)')
    parser.add_argument('--bypass_ramp', type=int, default=10,
                        help='Number of epochs to ramp bypass from 0 to full strength (default: 10)')


    # Dataset
    parser.add_argument('--single_scene_only', action='store_true',
                        help='Only use scene 0 for quick prototyping')
    parser.add_argument('--num_scenes', type=int, default=0,
                        help='Number of scenes to use (0 = all)')
    parser.add_argument('--unmatch_ratio', type=int, default=0,
                        help='Unmatch samples per match (0 = match only)')

    # Checkpointing
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--resume_dir', type=str, default=None,
                        help='Path to a run directory to auto-resume (loads args.json + latest.pth)')
    parser.add_argument('--save_every', type=int, default=5,
                        help='Save checkpoint every N epochs (0 = disable)')

    # Inference (used by infer.py)
    parser.add_argument('--run_dir', type=str, default=None,
                        help='Path to a specific run directory (for inference)')
    parser.add_argument('--scene_idx', type=int, default=0,
                        help='Scene index for inference')
    parser.add_argument('--plane_idx', type=str, default='0,20,39',
                        help='Focal plane indices, comma-separated (e.g. 0,20,39 or -1 for all)')
    parser.add_argument('--ckpt_tag', type=str, default='all',
                        choices=['best', 'best_psnr', 'latest', 'all'],
                        help='Which checkpoint to use: best, best_psnr, latest, or all (default: all)')

    # Inference-time Sharpening (infer.py only)
    parser.add_argument('--infer_sharp', action='store_true',
                        help='Enable inference-time sharpening (post-hoc sharp prior gradient injection)')
    parser.add_argument('--infer_sharp_lambda', type=float, default=5.0,
                        help='Sharpening strength (default: 5.0)')
    parser.add_argument('--infer_sharp_gamma', type=float, default=30.0,
                        help='CoC decay rate for sharpening weight (default: 30.0)')
    parser.add_argument('--infer_sharp_start', type=float, default=0.5,
                        help='Start sharpening at this fraction of total steps (0.0~1.0, default: 0.5)')

    return parser


def parse_args():
    return get_parser().parse_args()
