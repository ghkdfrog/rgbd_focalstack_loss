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
                        choices=['simple', 'deep', 'stride', 'resnet', 'resnet_film', 'dwt_resnet_film', 'resunet', 'convnext', 'convnext_unet', 'dilated', 'interleave_resnet'],
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
    parser.add_argument('--clip_image', action='store_true',
                        help='Use PGD-style clamping [0, 1] after each Langevin step')

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

    # ── Compositional EBM (Multi-Head Sobolev Training) ──
    parser.add_argument('--compositional_ebm', action='store_true',
                        help='Enable multi-head Compositional EBM (Struct, Percep, Phys)')
    # Head on/off toggles
    parser.add_argument('--enable_struct', action='store_true',
                        help='Enable structural head (L2 + SSIM)')
    parser.add_argument('--enable_percep', action='store_true',
                        help='Enable perceptual head (LPIPS)')
    parser.add_argument('--enable_phys', action='store_true',
                        help='Enable physical/optical head')
    # Head-level loss combination weights
    parser.add_argument('--w_struct', type=float, default=1.0,
                        help='Weight for structural Sobolev loss')
    parser.add_argument('--w_percep', type=float, default=1.0,
                        help='Weight for perceptual Sobolev loss')
    parser.add_argument('--w_phys', type=float, default=1.0,
                        help='Weight for physical Sobolev loss')
    # Sobolev anchor (scalar altitude) loss weights — keep very low
    parser.add_argument('--lambda_struct', type=float, default=1e-7,
                        help='Anchor weight for struct energy (E_struct ≈ T_struct)')
    parser.add_argument('--lambda_percep', type=float, default=1e-7,
                        help='Anchor weight for percep energy (E_percep ≈ T_percep)')
    parser.add_argument('--lambda_phys', type=float, default=1e-7,
                        help='Anchor weight for phys energy (E_phys ≈ T_phys)')
    # Structural target internal weights
    parser.add_argument('--alpha_struct', type=float, default=1.0,
                        help='Struct target: L2 (SSE) weight')
    parser.add_argument('--beta_struct', type=float, default=1.0,
                        help='Struct target: (1-SSIM) weight')
    # Physics sub-loss on/off toggles
    parser.add_argument('--enable_phys_blur', action='store_true',
                        help='Enable blur/edge focus loss in T_phys')
    parser.add_argument('--enable_phys_occ', action='store_true',
                        help='Enable occlusion boundary loss in T_phys')
    parser.add_argument('--enable_phys_energy', action='store_true',
                        help='Enable local energy conservation loss in T_phys')
    parser.add_argument('--enable_phys_bokeh', action='store_true',
                        help='Enable high-intensity bokeh loss in T_phys')
    # Physics sub-loss weights
    parser.add_argument('--lambda_blur_edge', type=float, default=1.0,
                        help='Weight for blur/edge focus loss inside T_phys')
    parser.add_argument('--lambda_occlusion', type=float, default=1.0,
                        help='Weight for occlusion boundary loss inside T_phys')
    parser.add_argument('--lambda_energy_conserv', type=float, default=1.0,
                        help='Weight for local energy conservation loss inside T_phys')
    parser.add_argument('--lambda_bokeh', type=float, default=1.0,
                        help='Weight for high-intensity bokeh loss inside T_phys')
    # Physics sub-loss hyperparameters
    parser.add_argument('--phys_gamma', type=float, default=30.0,
                        help='Blur edge: W_focal decay rate exp(-gamma*|CoC|)')
    parser.add_argument('--kappa_occ', type=float, default=5.0,
                        help='Occlusion mask sensitivity: M_occ = tanh(kappa * |grad(D)|)')
    parser.add_argument('--energy_pool_k', type=int, default=31,
                        help='Avg pool kernel size for energy conservation loss')
    parser.add_argument('--bokeh_threshold', type=float, default=0.95,
                        help='Highlight intensity threshold for bokeh mask')
    parser.add_argument('--bokeh_dilate_k', type=int, default=15,
                        help='MaxPool2d kernel size for bokeh mask dilation')

    # Energy Regularization
    parser.add_argument('--enable_energy_dist', action='store_true',
                        help='Enable E(current) ≈ -0.5·scale·mean(|current-GT|²) distance loss')
    parser.add_argument('--weight_energy_dist', type=float, default=0.01,
                        help='Weight for energy distance loss (default: 0.01)')
    parser.add_argument('--energy_dist_scale', type=float, default=1.0,
                        help='Scale constant for target energy: -0.5·scale·mean(|x-gt|²) (default: 1.0)')
    parser.add_argument('--enable_energy_anchor', action='store_true',
                        help='Enable E(GT)²→0 anchor loss')
    parser.add_argument('--weight_energy_anchor', type=float, default=0.01,
                        help='Weight for energy anchor loss (default: 0.01)')

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
    parser.add_argument('--new_run_on_resume', action='store_true',
                        help='Create a new run directory when resuming, instead of overwriting the resume_dir')
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
                        
    # Compositional Backward Compatibility
    parser.add_argument('--force_compositional', action='store_true',
                        help='Force load a non-compositional standard checkpoint into a compositional struct head for backward testing')

    return parser


def parse_args():
    return get_parser().parse_args()
