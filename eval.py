
import torch
import argparse
import json
from torchvision import transforms
from src.dataset import EvalDataset
from torch.utils.data import DataLoader
from src.metrics import Evaluator



def main(args):

    # Define the transformation

    # Create dataset
    dataset = EvalDataset(
        gt_folder=args.ground_truth,
        output_folder=args.output,
        mask_folder=args.mask,
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
            f.write("\n" + metric_lines + "\n")


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
    main(args)
