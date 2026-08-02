import argparse
import logging
import os
import json
import torch
import numpy as np
import torch.nn as nn
from scipy import linalg
from torch.utils.data import DataLoader
from torchmetrics.functional import structural_similarity_index_measure as ssim
from torchmetrics.functional import peak_signal_noise_ratio as psnr
from skimage.metrics import peak_signal_noise_ratio as psnr_sk
from lpips import LPIPS
from torchvision import transforms
from torch.nn import functional as F
from .fid.inception import InceptionV3
from .dataset import EvalDataset

LOGGER = logging.getLogger(__name__)

class PSNR(nn.Module):
    def __init__(self, max_val):
        super(PSNR, self).__init__()

        base10 = torch.log(torch.tensor(10.0))
        max_val = torch.tensor(max_val).float()

        self.register_buffer('base10', base10)
        self.register_buffer('max_val', 20 * torch.log(max_val) / base10)

    def __call__(self, a, b):
        mse = torch.mean((a.float() - b.float()) ** 2)

        if mse == 0:
            return 0

        return self.max_val - 10 * torch.log(mse) / self.base10

def fid_calculate_activation_statistics(act):
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma


def calculate_frechet_distance(activations_pred, activations_target, eps=1e-6):
    mu1, sigma1 = fid_calculate_activation_statistics(activations_pred)
    mu2, sigma2 = fid_calculate_activation_statistics(activations_target)

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        LOGGER.warning(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-2):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) +
            np.trace(sigma2) - 2 * tr_covmean)

class FIDScore:
    def __init__(self, dims=2048, eps=1e-6, device='cpu'): 
        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
        if not hasattr(FIDScore, '_MODEL') or FIDScore._MODEL is None:
            FIDScore._MODEL = InceptionV3([block_idx]).eval()
            FIDScore._MODEL.to(device) 
        self.model = FIDScore._MODEL
        self.eps = eps
        self.device = device 

    def get_activations(self, batch):
        with torch.no_grad():
            activations = self.model(batch)[0]
            if activations.shape[2] != 1 or activations.shape[3] != 1:
                activations = F.adaptive_avg_pool2d(activations, output_size=(1, 1))
            activations = activations.squeeze(-1).squeeze(-1)
            return activations.cpu().numpy() # Return numpy array for calculate_frechet_distance

    def calculate_fid(self, activations_pred, activations_target):
        return calculate_frechet_distance(activations_pred, activations_target, eps=self.eps)

def eval_score(fid, lpips, ssim, psnr,
               fid_max=100, lpips_max=1.0,
               ssim_min=0.0, psnr_min=0.0,
               weights=(0.3, 0.2, 0.2, 0.3)): 

    fid_score = max(0.0, fid_max - min(fid, fid_max)) / fid_max
    lpips_score = max(0.0, lpips_max - min(lpips, lpips_max)) / lpips_max
    ssim_score = max(0.0, ssim - ssim_min) / (1.0 - ssim_min)
    psnr_score = max(0.0, psnr - psnr_min) / (50.0 - psnr_min)  # 通常 PSNR > 50 不太常见

    w_fid, w_lpips, w_ssim, w_psnr = weights

    eps = 1e-6
    score = 1.0 / (
            (w_fid / (fid_score + eps)) +
            (w_lpips / (lpips_score + eps)) +
            (w_ssim / (ssim_score + eps)) +
            (w_psnr / (psnr_score + eps))
    )
    return score


class Evaluator:
    def __init__(self, device='cuda', num_bins=6, lpips_net='alex', withmask=False):
        self.device = device
        self.num_bins = num_bins
        self.lpips_fn = LPIPS(net=lpips_net).to(device).eval()
        self.withmask = withmask

        # Initialize custom FIDScore for all FID calculations
        self.fid_calculator = FIDScore(dims=2048, device=device)

        self.reset_metrics()

    def reset_metrics(self):
        # Reset lists for other metrics
        self.metrics = {i: {'lpips': [], 'ssim': [], 'psnr': [], 'l1': []}
                        for i in range(self.num_bins)}
        self.total = {'lpips': [], 'ssim': [], 'psnr': [], 'l1': []}

        # Reset FID activations for custom FIDScore (for total FID)
        self.fid_activations_total_real = []
        self.fid_activations_total_pred = []

        # Reset FID activations for bin-wise FID (using custom FIDScore)
        self.fid_activations_bins_real = {i: [] for i in range(self.num_bins)}
        self.fid_activations_bins_pred = {i: [] for i in range(self.num_bins)}

        self.bin_counts = {i: 0 for i in range(self.num_bins)}
        self.total_count = 0

    def psnr_metric(self, pred, target):
        pred_np = pred.permute(0,2,3,1).cpu().numpy()
        target_np = target.permute(0,2,3,1).cpu().numpy()
        return np.mean([psnr_sk(t, p, data_range=1.0) #
                        for t, p in zip(target_np, pred_np)])

    def get_bin_ids(self, masks):
        # print("masks in eval :", type(masks), torch.unique(masks), masks.float().mean())
        
        ratios = masks.flatten(1).float().mean(1)  # B,
        bin_ids = torch.clamp((ratios * 10).long(), 0, self.num_bins - 1)
        return bin_ids.cpu(), ratios.cpu()

    def compute_lpips_batchwise(self, img1, img2):
        with torch.no_grad():
            img1 = img1 * 2 - 1 # LPIPS expects images in [-1, 1]
            img2 = img2 * 2 - 1
            score = self.lpips_fn(img1, img2).squeeze()
            if score.dim() == 0:
                score = score.unsqueeze(0)
            return score.cpu().mean().item()

    def compute_metric_batchwise(self, metric_fn, img1, img2):
        with torch.no_grad():
            score = metric_fn(img1, img2)
            if score.dim() == 0:
                score = score.unsqueeze(0)
            return score.cpu().mean().item()

    @torch.no_grad()
    def forward(self, dataloader: DataLoader):
        self.reset_metrics()

        for i, batch in enumerate(dataloader):
            if batch is None:
                continue

            image = batch['image']
            output = batch['output']
            self.total_count += image.size(0)
            if self.withmask:
                mask = batch['mask']

            # Move batch data to device
            image_gpu = image.to(self.device)
            output_gpu = output.to(self.device)

            # --- Accumulate activations for TOTAL FID using custom FIDScore ---
            self.fid_activations_total_real.append(self.fid_calculator.get_activations(image_gpu))
            self.fid_activations_total_pred.append(self.fid_calculator.get_activations(output_gpu))

            # --- Calculate other total metrics for the current batch ---
            self.total['lpips'].append(self.compute_lpips_batchwise(image_gpu, output_gpu))
            self.total['ssim'].append(self.compute_metric_batchwise(ssim, output_gpu, image_gpu))
            self.total['psnr'].append(self.psnr_metric(output_gpu, image_gpu))
            self.total['l1'].append(F.l1_loss(output_gpu, image_gpu).item())

            if self.withmask:
                bin_ids, _ = self.get_bin_ids(mask)

                for bin_id in range(self.num_bins):
                    idx = (bin_ids == bin_id)
                    count = idx.sum().item()
                    self.bin_counts[bin_id] += count
                    if idx.sum() == 0:
                        continue

                    gt_bin = image_gpu[idx]
                    pred_bin = output_gpu[idx]

                    if gt_bin.shape[0] > 0:
                        # --- Accumulate activations for BIN-WISE FID using custom FIDScore ---
                        self.fid_activations_bins_real[bin_id].append(self.fid_calculator.get_activations(gt_bin))
                        self.fid_activations_bins_pred[bin_id].append(self.fid_calculator.get_activations(pred_bin))

                        # Calculate other metrics for the current bin batch slice
                        lp = self.compute_lpips_batchwise(gt_bin, pred_bin)
                        self.metrics[bin_id]['lpips'].append(lp)
                        self.metrics[bin_id]['ssim'].append(self.compute_metric_batchwise(ssim, pred_bin, gt_bin))
                        self.metrics[bin_id]['psnr'].append(self.psnr_metric(pred_bin, gt_bin))
                        self.metrics[bin_id]['l1'].append(F.l1_loss(pred_bin, gt_bin).item())

            del image, output, image_gpu, output_gpu
            if self.withmask:
                del mask
            torch.cuda.empty_cache()

        # After processing all batches, compute final scores
        return self.compute_scores(mask_present=(self.withmask))

    def compute_scores(self, mask_present=True):
        results = {}

        # --- Compute Bin-wise FID using custom FIDScore ---
        if mask_present:
            for bin_id in range(self.num_bins):
                m = self.metrics[bin_id]

                fid_score_bin = np.nan
                try:
                    if self.fid_activations_bins_real[bin_id] and self.fid_activations_bins_pred[bin_id]:
                        all_activations_bin_real = np.concatenate(self.fid_activations_bins_real[bin_id], axis=0)
                        all_activations_bin_pred = np.concatenate(self.fid_activations_bins_pred[bin_id], axis=0)
                        fid_score_bin = self.fid_calculator.calculate_fid(all_activations_bin_pred, all_activations_bin_real)
                except Exception as e:
                    LOGGER.warning(f"Warning: FID computation failed for bin {bin_id}: {e}. Setting FID to NaN.")


                lpips_avg = np.mean(m['lpips']) if m['lpips'] else np.nan
                ssim_avg = np.mean(m['ssim']) if m['ssim'] else np.nan
                psnr_avg = np.mean(m['psnr']) if m['psnr'] else np.nan
                l1_avg = np.mean(m['l1']) if m['l1'] else np.nan

                results[bin_id] = {
                    'FID': fid_score_bin,
                    'LPIPS': lpips_avg,
                    'SSIM': ssim_avg,
                    'PSNR': psnr_avg,
                    'L1': l1_avg,
                    'count': self.bin_counts[bin_id]
                }

        # --- Compute Total FID using custom FIDScore ---
        m_total = self.total
        fid_score_total = np.nan
        try:
            if self.fid_activations_total_real and self.fid_activations_total_pred:
                # Concatenate all collected activations
                all_activations_real = np.concatenate(self.fid_activations_total_real, axis=0)
                all_activations_pred = np.concatenate(self.fid_activations_total_pred, axis=0)
                fid_score_total = self.fid_calculator.calculate_fid(all_activations_pred, all_activations_real)
        except Exception as e:
            LOGGER.warning(f"Warning: Total custom FID computation failed: {e}. Setting FID to NaN.")


        lpips_avg_total = np.mean(m_total['lpips']) if m_total['lpips'] else np.nan
        ssim_avg_total = np.mean(m_total['ssim']) if m_total['ssim'] else np.nan
        psnr_avg_total = np.mean(m_total['psnr']) if m_total['psnr'] else np.nan
        l1_avg_total = np.mean(m_total['l1']) if m_total['l1'] else np.nan

        results['mean'] = {
            'FID': fid_score_total,
            'LPIPS': lpips_avg_total,
            'SSIM': ssim_avg_total,
            'PSNR': psnr_avg_total,
            'L1': l1_avg_total,
            'F4Score': self.compute_eval_scsore(fid_score_total, lpips_avg_total, ssim_avg_total,
                                                psnr_avg_total, l1_avg_total)
        }
        return results

    def compute_eval_scsore(self, fid, lpips, ssim_score, psnr_score, l1_score):
        return eval_score(fid, lpips, ssim_score, psnr_score)

    def print_metrics(self, results):
            output = "\n--- Evaluation Metrics ---\n"
            header = f"{'ratios':<12}{'FID':<12}{'LPIPS':<12}{'SSIM':<12}{'PSNR':<12}{'L1loss':<12}{'Count':<18}"
            output += header + "\n"

            if any(isinstance(val, dict) for val in results.values() if val != results.get('mean')):
                for bin_id in range(self.num_bins):
                    res = results.get(bin_id)
                    if res is None or (not res['LPIPS'] and not res['SSIM'] and not res['PSNR'] and not res['L1'] and np.isnan(res['FID'])):
                        line = f"{bin_id * 10}-{(bin_id + 1) * 10}% {'-':<12}{'-':<12}{'-':<12}{'-':<12}{'-':<12}{'-':<18}\n"
                    else:
                        fid_str = f"{res['FID']:<12.4f}" if not (np.isnan(res['FID']) or np.isinf(res['FID'])) else f"{'-':<12}"
                        lpips_str = f"{res['LPIPS']:<12.4f}" if not np.isnan(res['LPIPS']) else f"{'-':<12}"
                        ssim_str = f"{res['SSIM']:<12.4f}" if not np.isnan(res['SSIM']) else f"{'-':<12}"
                        psnr_str = f"{res['PSNR']:<12.4f}" if not np.isnan(res['PSNR']) else f"{'-':<12}"
                        l1_str = f"{res['L1']:<12.4f}" if not np.isnan(res['L1']) else f"{'-':<12}"

                        count_str = f"{res.get('count', 0):<18}"  
                        line = f"{bin_id * 10}-{(bin_id + 1) * 10}% {fid_str}{lpips_str}{ssim_str}{psnr_str}{l1_str}{count_str}\n"
                    output += line

            mean = results['mean']
            fid_str_mean = f"{mean['FID']:<12.4f}" if not (np.isnan(mean['FID']) or np.isinf(mean['FID'])) else f"{'-':<12}"
            lpips_str_mean = f"{mean['LPIPS']:<12.4f}" if not np.isnan(mean['LPIPS']) else f"{'-':<12}"
            ssim_str_mean = f"{mean['SSIM']:<12.4f}" if not np.isnan(mean['SSIM']) else f"{'-':<12}"
            psnr_str_mean = f"{mean['PSNR']:<12.4f}" if not np.isnan(mean['PSNR']) else f"{'-':<12}"
            l1_str_mean = f"{mean['L1']:<12.4f}" if not np.isnan(mean['L1']) else f"{'-':<12}"
            f4_str_mean = f"{mean['F4Score']:<18.6f}" if not np.isnan(mean['F4Score']) else f"{'-':<18}"

            mean_line = f"{'Mean':<12}{fid_str_mean}{lpips_str_mean}{ssim_str_mean}{psnr_str_mean}{l1_str_mean}{f4_str_mean}\n"
            
            output += mean_line

            if self.device == torch.device('cuda'):
                torch.cuda.empty_cache()

            return mean.get('F4Score', 0), output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ground_truth', type=str, default='/home/student01/MxT-main/datasets/CelebA-HQ/test_256',
                        help='ground truth path ')
    parser.add_argument('--output', type=str, default='/home/student01/MxT-main/datasets/CelebA-HQ/test_256',
                        help='output path ')
    parser.add_argument('--mask', type=str, default=None,
                        help='mask path ')
    parser.add_argument('--metrics', type=str, default=None,
                        help='metrics path ')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for DataLoader')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of workers for DataLoader')
    args = parser.parse_args()

    # Define the transformation
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),  # Converts [0, 255] to [0.0, 1.0]
    ])
    # Create dataset
    dataset = EvalDataset(
        gt_folder=args.ground_truth,
        output_folder=args.output,
        mask_folder=args.mask,
        transform=transform

    )

    if len(dataset) == 0:
        print("No images found in the specified output folder. Please check your paths.")
    else:
        # Create DataLoader
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )
        withmask = args.mask is not None
        evaluator = Evaluator(num_bins=6, device='cuda' if torch.cuda.is_available() else 'cpu', withmask=withmask)
        metrics = evaluator.forward(dataloader)
        _, metric_lines = evaluator.print_metrics(metrics)
        print("\n",metric_lines)

        with open(args.metrics, "a", encoding="utf-8") as f:
            json.dump(metrics, f, indent=4)
            f.write("\n") 
