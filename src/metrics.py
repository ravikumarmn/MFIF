import torch
import torch.nn.functional as F
import numpy as np
from skimage.metrics import structural_similarity as ssim
from scipy import ndimage
from sklearn.metrics import mutual_info_score
import cv2


def calculate_ssim(img1, img2):
    """Calculate SSIM between two images"""
    # Convert tensors to numpy arrays
    if isinstance(img1, torch.Tensor):
        img1 = img1.detach().cpu().numpy()
    if isinstance(img2, torch.Tensor):
        img2 = img2.detach().cpu().numpy()

    # Handle batch dimension
    if len(img1.shape) == 4:  # Batch of images
        ssim_values = []
        for i in range(img1.shape[0]):
            # Convert from CHW to HWC format
            img1_hwc = np.transpose(img1[i], (1, 2, 0))
            img2_hwc = np.transpose(img2[i], (1, 2, 0))

            # Calculate SSIM with proper data range
            ssim_val = ssim(
                img1_hwc,
                img2_hwc,
                multichannel=True,
                channel_axis=2,
                data_range=img1_hwc.max() - img1_hwc.min(),
            )
            ssim_values.append(ssim_val)
        return np.mean(ssim_values)
    else:
        # Single image
        img1_hwc = np.transpose(img1, (1, 2, 0))
        img2_hwc = np.transpose(img2, (1, 2, 0))
        return ssim(
            img1_hwc,
            img2_hwc,
            multichannel=True,
            channel_axis=2,
            data_range=img1_hwc.max() - img1_hwc.min(),
        )


def calculate_psnr(img1, img2):
    """Calculate PSNR between two images"""
    mse = F.mse_loss(img1, img2)
    if mse == 0:
        return float("inf")
    return 20 * torch.log10(1.0 / torch.sqrt(mse))


def calculate_l1_loss(pred, target):
    """Calculate L1 loss"""
    return F.l1_loss(pred, target)


def calculate_mse_loss(pred, target):
    """Calculate MSE loss"""
    return F.mse_loss(pred, target)


class SSIMLoss(torch.nn.Module):
    """SSIM Loss implementation"""

    def __init__(self, window_size=11, size_average=True):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average

    def forward(self, img1, img2):
        # Simple SSIM loss approximation using correlation
        mu1 = F.avg_pool2d(img1, self.window_size, 1, padding=self.window_size // 2)
        mu2 = F.avg_pool2d(img2, self.window_size, 1, padding=self.window_size // 2)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = (
            F.avg_pool2d(
                img1 * img1, self.window_size, 1, padding=self.window_size // 2
            )
            - mu1_sq
        )
        sigma2_sq = (
            F.avg_pool2d(
                img2 * img2, self.window_size, 1, padding=self.window_size // 2
            )
            - mu2_sq
        )
        sigma12 = (
            F.avg_pool2d(
                img1 * img2, self.window_size, 1, padding=self.window_size // 2
            )
            - mu1_mu2
        )

        C1 = 0.01**2
        C2 = 0.03**2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
            (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
        )

        if self.size_average:
            return 1 - ssim_map.mean()
        else:
            return 1 - ssim_map.mean(1).mean(1).mean(1)


def evaluate_model(model, data_loader, device):
    """Evaluate model on validation set"""
    model.eval()
    total_l1_loss = 0
    total_ssim = 0
    total_psnr = 0
    num_batches = 0

    with torch.no_grad():
        for batch in data_loader:
            source_a = batch["source_a"].to(device)
            source_b = batch["source_b"].to(device)
            ground_truth = batch["ground_truth"].to(device)

            # Forward pass
            output = model(source_a, source_b)

            # Calculate metrics
            l1_loss = calculate_l1_loss(output, ground_truth)
            ssim_score = calculate_ssim(output, ground_truth)
            psnr_score = calculate_psnr(output, ground_truth)

            total_l1_loss += l1_loss.item()
            total_ssim += ssim_score
            total_psnr += psnr_score.item()
            num_batches += 1

    avg_l1_loss = total_l1_loss / num_batches
    avg_ssim = total_ssim / num_batches
    avg_psnr = total_psnr / num_batches

    return {"l1_loss": avg_l1_loss, "ssim": avg_ssim, "psnr": avg_psnr}


def calculate_qnmi(fused_img, source_a, source_b):
    """
    Calculate Normalized Mutual Information (QNMI) metric
    Measures information preservation from source images to fused image
    """
    if isinstance(fused_img, torch.Tensor):
        fused_img = fused_img.detach().cpu().numpy()
    if isinstance(source_a, torch.Tensor):
        source_a = source_a.detach().cpu().numpy()
    if isinstance(source_b, torch.Tensor):
        source_b = source_b.detach().cpu().numpy()

    # Handle batch dimension
    if len(fused_img.shape) == 4:
        qnmi_values = []
        for i in range(fused_img.shape[0]):
            # Convert to grayscale for MI calculation
            f_gray = np.mean(fused_img[i], axis=0).flatten()
            a_gray = np.mean(source_a[i], axis=0).flatten()
            b_gray = np.mean(source_b[i], axis=0).flatten()

            # Normalize to [0, 255] for histogram calculation
            f_gray = (
                (f_gray - f_gray.min()) / (f_gray.max() - f_gray.min()) * 255
            ).astype(np.uint8)
            a_gray = (
                (a_gray - a_gray.min()) / (a_gray.max() - a_gray.min()) * 255
            ).astype(np.uint8)
            b_gray = (
                (b_gray - b_gray.min()) / (b_gray.max() - b_gray.min()) * 255
            ).astype(np.uint8)

            # Calculate mutual information
            mi_fa = mutual_info_score(f_gray, a_gray)
            mi_fb = mutual_info_score(f_gray, b_gray)

            # Calculate entropy
            h_f = -np.sum(
                np.histogram(f_gray, bins=256)[0]
                / len(f_gray)
                * np.log2(np.histogram(f_gray, bins=256)[0] / len(f_gray) + 1e-10)
            )
            h_a = -np.sum(
                np.histogram(a_gray, bins=256)[0]
                / len(a_gray)
                * np.log2(np.histogram(a_gray, bins=256)[0] / len(a_gray) + 1e-10)
            )
            h_b = -np.sum(
                np.histogram(b_gray, bins=256)[0]
                / len(b_gray)
                * np.log2(np.histogram(b_gray, bins=256)[0] / len(b_gray) + 1e-10)
            )

            # Normalized MI
            nmi_fa = 2 * mi_fa / (h_f + h_a) if (h_f + h_a) > 0 else 0
            nmi_fb = 2 * mi_fb / (h_f + h_b) if (h_f + h_b) > 0 else 0

            qnmi_values.append((nmi_fa + nmi_fb) / 2)

        return np.mean(qnmi_values)
    else:
        # Single image processing
        f_gray = np.mean(fused_img, axis=0).flatten()
        a_gray = np.mean(source_a, axis=0).flatten()
        b_gray = np.mean(source_b, axis=0).flatten()

        # Normalize and calculate as above
        f_gray = ((f_gray - f_gray.min()) / (f_gray.max() - f_gray.min()) * 255).astype(
            np.uint8
        )
        a_gray = ((a_gray - a_gray.min()) / (a_gray.max() - a_gray.min()) * 255).astype(
            np.uint8
        )
        b_gray = ((b_gray - b_gray.min()) / (b_gray.max() - b_gray.min()) * 255).astype(
            np.uint8
        )

        mi_fa = mutual_info_score(f_gray, a_gray)
        mi_fb = mutual_info_score(f_gray, b_gray)

        h_f = -np.sum(
            np.histogram(f_gray, bins=256)[0]
            / len(f_gray)
            * np.log2(np.histogram(f_gray, bins=256)[0] / len(f_gray) + 1e-10)
        )
        h_a = -np.sum(
            np.histogram(a_gray, bins=256)[0]
            / len(a_gray)
            * np.log2(np.histogram(a_gray, bins=256)[0] / len(a_gray) + 1e-10)
        )
        h_b = -np.sum(
            np.histogram(b_gray, bins=256)[0]
            / len(b_gray)
            * np.log2(np.histogram(b_gray, bins=256)[0] / len(b_gray) + 1e-10)
        )

        nmi_fa = 2 * mi_fa / (h_f + h_a) if (h_f + h_a) > 0 else 0
        nmi_fb = 2 * mi_fb / (h_f + h_b) if (h_f + h_b) > 0 else 0

        return (nmi_fa + nmi_fb) / 2


def calculate_qg(fused_img, source_a, source_b):
    """
    Calculate Gradient-based metric (QG)
    Measures edge preservation quality in the fused image
    """
    if isinstance(fused_img, torch.Tensor):
        fused_img = fused_img.detach().cpu().numpy()
    if isinstance(source_a, torch.Tensor):
        source_a = source_a.detach().cpu().numpy()
    if isinstance(source_b, torch.Tensor):
        source_b = source_b.detach().cpu().numpy()

    # Sobel operators for gradient calculation
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    def calculate_gradient_strength(img):
        """Calculate gradient strength using Sobel operators"""
        if len(img.shape) == 4:  # Batch
            gradients = []
            for i in range(img.shape[0]):
                # Convert to grayscale
                gray = np.mean(img[i], axis=0)

                # Calculate gradients
                grad_x = ndimage.convolve(gray, sobel_x)
                grad_y = ndimage.convolve(gray, sobel_y)

                # Gradient magnitude
                grad_mag = np.sqrt(grad_x**2 + grad_y**2)
                gradients.append(grad_mag)
            return np.array(gradients)
        else:
            gray = np.mean(img, axis=0)
            grad_x = ndimage.convolve(gray, sobel_x)
            grad_y = ndimage.convolve(gray, sobel_y)
            return np.sqrt(grad_x**2 + grad_y**2)

    # Calculate gradients
    grad_f = calculate_gradient_strength(fused_img)
    grad_a = calculate_gradient_strength(source_a)
    grad_b = calculate_gradient_strength(source_b)

    if len(fused_img.shape) == 4:  # Batch
        qg_values = []
        for i in range(fused_img.shape[0]):
            # Calculate QG for each image in batch
            gf, ga, gb = grad_f[i], grad_a[i], grad_b[i]

            # Edge preservation weights
            waf = ga / (ga + gb + 1e-10)
            wbf = gb / (ga + gb + 1e-10)

            # QG calculation
            qg = np.sum(waf * (gf * ga) + wbf * (gf * gb)) / np.sum(
                waf * ga**2 + wbf * gb**2 + 1e-10
            )
            qg_values.append(qg)

        return float(np.mean(qg_values))
    else:
        # Single image
        waf = grad_a / (grad_a + grad_b + 1e-10)
        wbf = grad_b / (grad_a + grad_b + 1e-10)

        qg = np.sum(waf * (grad_f * grad_a) + wbf * (grad_f * grad_b)) / np.sum(
            waf * grad_a**2 + wbf * grad_b**2 + 1e-10
        )
        return float(qg)


def calculate_qcb(fused_img, source_a, source_b):
    """
    Calculate Correlation Coefficient-based metric (QCB)
    Measures correlation between fused image and source images
    """
    if isinstance(fused_img, torch.Tensor):
        fused_img = fused_img.detach().cpu().numpy()
    if isinstance(source_a, torch.Tensor):
        source_a = source_a.detach().cpu().numpy()
    if isinstance(source_b, torch.Tensor):
        source_b = source_b.detach().cpu().numpy()

    def correlation_coefficient(x, y):
        """Calculate correlation coefficient between two arrays"""
        x_flat = x.flatten()
        y_flat = y.flatten()

        # Remove mean
        x_centered = x_flat - np.mean(x_flat)
        y_centered = y_flat - np.mean(y_flat)

        # Calculate correlation
        numerator = np.sum(x_centered * y_centered)
        denominator = np.sqrt(np.sum(x_centered**2) * np.sum(y_centered**2))

        return numerator / (denominator + 1e-10)

    if len(fused_img.shape) == 4:  # Batch
        qcb_values = []
        for i in range(fused_img.shape[0]):
            # Convert to grayscale
            f_gray = np.mean(fused_img[i], axis=0)
            a_gray = np.mean(source_a[i], axis=0)
            b_gray = np.mean(source_b[i], axis=0)

            # Calculate correlations
            corr_fa = correlation_coefficient(f_gray, a_gray)
            corr_fb = correlation_coefficient(f_gray, b_gray)

            # QCB is the average correlation
            qcb_values.append((corr_fa + corr_fb) / 2)

        return np.mean(qcb_values)
    else:
        # Single image
        f_gray = np.mean(fused_img, axis=0)
        a_gray = np.mean(source_a, axis=0)
        b_gray = np.mean(source_b, axis=0)

        corr_fa = correlation_coefficient(f_gray, a_gray)
        corr_fb = correlation_coefficient(f_gray, b_gray)

        return (corr_fa + corr_fb) / 2


def calculate_qpiella(fused_img, source_a, source_b):
    """
    Calculate Piella's metric (QPiella)
    Measures structural similarity preservation in fusion
    """
    if isinstance(fused_img, torch.Tensor):
        fused_img = fused_img.detach().cpu().numpy()
    if isinstance(source_a, torch.Tensor):
        source_a = source_a.detach().cpu().numpy()
    if isinstance(source_b, torch.Tensor):
        source_b = source_b.detach().cpu().numpy()

    def local_variance(img, window_size=8):
        """Calculate local variance using sliding window"""
        kernel = np.ones((window_size, window_size)) / (window_size * window_size)
        mean = ndimage.convolve(img, kernel, mode="reflect")
        sqr_mean = ndimage.convolve(img * img, kernel, mode="reflect")
        return sqr_mean - mean * mean

    def calculate_weights(img_a, img_b, window_size=8):
        """Calculate importance weights based on local variance"""
        var_a = local_variance(img_a, window_size)
        var_b = local_variance(img_b, window_size)

        # Normalize weights
        total_var = var_a + var_b + 1e-10
        w_a = var_a / total_var
        w_b = var_b / total_var

        return w_a, w_b

    if len(fused_img.shape) == 4:  # Batch
        qpiella_values = []
        for i in range(fused_img.shape[0]):
            # Convert to grayscale
            f_gray = np.mean(fused_img[i], axis=0)
            a_gray = np.mean(source_a[i], axis=0)
            b_gray = np.mean(source_b[i], axis=0)

            # Calculate importance weights
            w_a, w_b = calculate_weights(a_gray, b_gray)

            # Calculate SSIM between fused and source images
            ssim_fa = ssim(f_gray, a_gray, data_range=f_gray.max() - f_gray.min())
            ssim_fb = ssim(f_gray, b_gray, data_range=f_gray.max() - f_gray.min())

            # Weighted SSIM (Piella's metric)
            qpiella = np.sum(w_a * ssim_fa + w_b * ssim_fb) / np.sum(w_a + w_b)
            qpiella_values.append(qpiella)

        return float(np.mean(qpiella_values))
    else:
        # Single image
        f_gray = np.mean(fused_img, axis=0)
        a_gray = np.mean(source_a, axis=0)
        b_gray = np.mean(source_b, axis=0)

        w_a, w_b = calculate_weights(a_gray, b_gray)

        ssim_fa = ssim(f_gray, a_gray, data_range=f_gray.max() - f_gray.min())
        ssim_fb = ssim(f_gray, b_gray, data_range=f_gray.max() - f_gray.min())

        return float(np.sum(w_a * ssim_fa + w_b * ssim_fb) / np.sum(w_a + w_b))


def evaluate_model_comprehensive(model, data_loader, device, num_samples=None):
    """
    Comprehensive evaluation with all paper metrics
    """
    model.eval()

    # Initialize metric accumulators
    metrics = {
        "l1_loss": 0,
        "ssim": 0,
        "psnr": 0,
        "qnmi": 0,
        "qg": 0,
        "qcb": 0,
        "qpiella": 0,
    }

    num_batches = 0
    samples_processed = 0

    with torch.no_grad():
        for batch in data_loader:
            if num_samples and samples_processed >= num_samples:
                break

            source_a = batch["source_a"].to(device)
            source_b = batch["source_b"].to(device)
            ground_truth = batch["ground_truth"].to(device)

            # Forward pass
            if hasattr(model, "generator"):  # GAN model
                output = model.generator(source_a, source_b)
            else:  # Regular model
                output = model(source_a, source_b)

            # Basic metrics
            metrics["l1_loss"] += calculate_l1_loss(output, ground_truth).item()
            metrics["ssim"] += calculate_ssim(output, ground_truth)
            metrics["psnr"] += calculate_psnr(output, ground_truth).item()

            # Paper-specific metrics
            metrics["qnmi"] += calculate_qnmi(output, source_a, source_b)
            metrics["qg"] += calculate_qg(output, source_a, source_b)
            metrics["qcb"] += calculate_qcb(output, source_a, source_b)
            metrics["qpiella"] += calculate_qpiella(output, source_a, source_b)

            num_batches += 1
            samples_processed += source_a.size(0)

    # Average all metrics
    for key in metrics:
        metrics[key] /= num_batches

    return metrics
