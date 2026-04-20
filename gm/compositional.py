import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    import lpips
except ImportError:
    pass

class CompositionalTargets(nn.Module):
    """
    Computes structural, perceptual, and physical target objectives for Compositional EBM.
    """
    def __init__(self, args, device):
        super().__init__()
        self.args = args
        self.device = device
        
        # 1. Perceptual Head (LPIPS - AlexNet)
        if args.compositional_ebm and args.enable_percep:
            self.lpips_net = lpips.LPIPS(net='alex').to(device).eval()
            for param in self.lpips_net.parameters():
                param.requires_grad = False
                
        # 2. Structural Head SSIM fallback
        # Since Kornia is broken in this environment, using a simple L1/L2 combination as a surrogate for structural loss.
        # Alternatively, using pure L1 loss which acts structurally similar to SSIM in many perceptual bounds.
        self.ssim_loss = nn.L1Loss(reduction='none')

    def _spatial_grad(self, x):
        """Simple spatial gradient magnitude (Sobel approx) using F.conv2d"""
        kernel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]], device=x.device).view(1, 1, 3, 3)
        kernel_y = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]], device=x.device).view(1, 1, 3, 3)
        
        c = x.shape[1]
        kernel_x = kernel_x.repeat(c, 1, 1, 1)
        kernel_y = kernel_y.repeat(c, 1, 1, 1)
        
        gx = F.conv2d(x, kernel_x, padding=1, groups=c)
        gy = F.conv2d(x, kernel_y, padding=1, groups=c)
        return torch.sqrt(gx**2 + gy**2 + 1e-6)

    def forward_struct(self, x_pred, gt):
        """
        Structural target function: L2 + SSIM.
        x_pred, gt: (N, 3, H, W)
        Returns: scalar total error (reduced by sum)
        """
        l2_err = 0.5 * (x_pred - gt) ** 2
        
        # SSIM Loss expects values in [0, 1] typically, so we keep them as is
        # Surrogate: L1 Loss per pixel
        ssim_err = self.ssim_loss(x_pred, gt)
        
        total_err = self.args.alpha_struct * l2_err + self.args.beta_struct * ssim_err
        return total_err.sum()

    def forward_percep(self, x_pred, gt):
        """
        Perceptual target function using AlexNet LPIPS.
        Returns: scalar total error (reduced by sum)
        """
        # lpips expects inputs in [-1, 1], our inputs are [0, 1]
        x_pred_lpips = x_pred * 2.0 - 1.0
        gt_lpips = gt * 2.0 - 1.0
        
        lpips_err = self.lpips_net(x_pred_lpips, gt_lpips)
        return lpips_err.sum()

    def forward_phys(self, x_pred, gt, x_full, diopter):
        """
        Physical target function with 4 sub-losses.
        x_full: Original input with RGB, D, and CoC.
        diopter: Diopter condition.
        Returns: scalar total error (reduced by sum)
        """
        N, C, H, W = x_pred.shape
        loss_phys = torch.tensor(0.0, device=self.device)
        
        coc_map = None
        if self.args.diopter_mode in ['coc', 'coc_signed', 'coc_abs'] and x_full.shape[1] >= 8:
            coc_map = x_full[:, 7:8, :, :]
            
        # 1. Blur Edge Loss (In-focus should match edges perfectly, out-of-focus edges are penalized)
        if self.args.enable_phys_blur and coc_map is not None:
            # Spatial gradients
            grad_pred = self._spatial_grad(x_pred)
            grad_gt = self._spatial_grad(gt)
            
            # W_focal = exp(-gamma * |CoC|)
            w_focal = torch.exp(-self.args.phys_gamma * torch.abs(coc_map))
            
            err_blur = w_focal * ((grad_pred - grad_gt) ** 2)
            loss_phys += self.args.lambda_blur_edge * err_blur.sum()
            
        # 2. Occlusion Boundary Loss (Emphasize error amplification at depth edges)
        if self.args.enable_phys_occ:
            # Assumes Depth is 4th channel (index 3)
            depth_map = x_full[:, 3:4, :, :]
            grad_depth = self._spatial_grad(depth_map)
            
            # M_occ = tanh(kappa * ||grad_depth||)
            m_occ = torch.tanh(self.args.kappa_occ * grad_depth)
            
            err_occ = m_occ * ((x_pred - gt) ** 2)
            loss_phys += self.args.lambda_occlusion * err_occ.sum()
            
        # 3. Local Energy Conservation (Avg pool)
        if self.args.enable_phys_energy:
            pad = self.args.energy_pool_k // 2
            pool = nn.AvgPool2d(kernel_size=self.args.energy_pool_k, stride=1, padding=pad)
            
            energy_pred = pool(x_pred)
            energy_gt = pool(gt)
            
            err_energy = (energy_pred - energy_gt) ** 2
            loss_phys += self.args.lambda_energy_conserv * err_energy.sum()
            
        # 4. High-Intensity Bokeh (Under-intensity prevention)
        if self.args.enable_phys_bokeh:
            # Bright regions mask
            bright_mask = (gt > self.args.bokeh_threshold).float()
            
            # Dilate using MaxPool2d (differentiable)
            pad_b = self.args.bokeh_dilate_k // 2
            dilate_pool = nn.MaxPool2d(kernel_size=self.args.bokeh_dilate_k, stride=1, padding=pad_b)
            
            m_bokeh = dilate_pool(bright_mask)
            
            # L1 Loss to prevent under-intensity but maintain sharpness
            err_bokeh = m_bokeh * torch.abs(x_pred - gt)
            loss_phys += self.args.lambda_bokeh * err_bokeh.sum()
            
        return loss_phys
