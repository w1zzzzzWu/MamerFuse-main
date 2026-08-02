import os
import re
import time
import logging

import numpy as np
from tqdm import tqdm
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr

import torch
from torch.utils.data import DataLoader

from .dataset import Dataset, EvalDataset
from .models import InpaintingModel
from .utils import Progbar, create_dir, stitch_images, imsave
from .metrics import PSNR, Evaluator


LOGGER = logging.getLogger(__name__)

class MamerFuse():
    def __init__(self, config):
        self.config = config

        if config.MODEL == 2:
            model_name = 'inpaint'
        self.results_path = config.RESULTS
        self.debug = False
        self.model_name = model_name

        self.inpaint_model = InpaintingModel(config).to(config.DEVICE)

        self.psnr = PSNR(255.0).to(config.DEVICE)

        # train mode
        if self.config.MODE == 1:
            if self.config.MODEL == 2:
                self.train_dataset = Dataset(config, config.TRAIN_INPAINT_IMAGE_FLIST, config.TRAIN_MASK_FLIST,
                                             augment=True, training=True)
                self.val_dataset = Dataset(
                    self.config,
                    self.config.VAL_INPAINT_IMAGE_FLIST,
                    self.config.VAL_MASK_FLIST,
                    augment=False,
                    training=True
                )

        # test mode
        if self.config.MODE == 2:
            if self.config.MODEL == 2:
                self.test_dataset = Dataset(self.config, config.TEST_INPAINT_IMAGE_FLIST, config.TEST_MASK_FLIST,
                                            augment=False, training=False)

        if config.DEBUG is not None and config.DEBUG != 0:
            self.debug = True

        self.log_file = os.path.join(self.results_path, 'log_' + model_name + '.log')

    def load(self):        
        if self.config.MODE ==1 and self.config.RESUMEPATH is None:
            LOGGER.info(f"Train without Resume model")
        else:
            self.inpaint_model.load()

    def save(self, iteration=None, score=None):
        """
        Save model only if it ranks among the top 5 best scores (higher is better).
        """
        if iteration is None:
            iteration = self.inpaint_model.iteration
        if score is None:
            raise ValueError("Must provide lpips_fid100_f1_total score to save selectively.")

        # Create directories if they do not exist
        save_dir = os.path.join(self.config.RESULTS, 'models')
        create_dir(save_dir)
        discriminator_dir = os.path.join(save_dir, 'discriminator')
        create_dir(discriminator_dir)

        # Gather all existing saved models excluding 'last.pth'
        model_files = [f for f in os.listdir(save_dir) if f.endswith('.pth') and 'last' not in f]
        scored_models = []
        for fname in model_files:
            try:
                match = re.search(r'_f1_([0-9.]+)', fname)
                if match:
                    model_score = float(match.group(1))
                    scored_models.append((model_score, fname))
            except Exception as e:
                LOGGER.warning(f"Failed to parse score from {fname}: {e}")
                continue

        # Determine if the current score qualifies for top-5
        scored_models.sort(reverse=True)  # Higher scores first
        is_top5 = len(scored_models) < 5 or score > scored_models[-1][0]

        if not is_top5:
            # If not in top-5, only save as 'last'
            model_name = 'last'
            model_path = os.path.join(save_dir, model_name)
            self.inpaint_model.save(path=save_dir, name=model_name)
            LOGGER.info(f"Saved model as 'last': {model_path}")
            return

        # Save with score-based filename
        model_name = f'iter_{iteration:07d}_f1_{score:.6f}'
        model_path = os.path.join(save_dir, model_name)
        self.inpaint_model.save(path=save_dir, name=model_name)

        # Add current model to scored list and sort again
        scored_models.append((score, f'{model_name}_gen.pth'))
        scored_models.sort(reverse=True)

        # Keep only top-5, delete others
        for item in scored_models[5:]:
            gen_file = os.path.join(save_dir, item[1])
            prefix_match = item[1].replace('_gen.pth', '')
            dis_file = os.path.join(discriminator_dir, f'{prefix_match}_dis.pth')

            if os.path.exists(gen_file):
                os.remove(gen_file)
                LOGGER.info(f"Removed old generator model: {gen_file}")
            if os.path.exists(dis_file):
                os.remove(dis_file)
                LOGGER.info(f"Removed old discriminator model: {dis_file}")

        LOGGER.info(f"Saved top-5 model: {model_path}")


    def train(self):

        train_loader = DataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.BATCH_SIZE,
            num_workers=4,
            drop_last=True,
            shuffle=True
        )

        epoch = 0
        keep_training = True
        max_iteration = int(float(self.config.MAX_ITERS))
        total = len(self.train_dataset)

        while (keep_training):
            epoch += 1
            LOGGER.info('\n\nTraining epoch: %d' % epoch)

            progbar = Progbar(total, width=20, stateful_metrics=['epoch', 'iter'])

            for items in train_loader:

                self.inpaint_model.train()

                images, masks = self.cuda(*items)

                outputs_img, gen_loss, dis_loss, logs, gen_gan_loss, gen_l1_loss, gen_content_loss, gen_style_loss = self.inpaint_model.process(
                    images, masks)
                outputs_merged = (outputs_img * masks) + (images * (1 - masks))

                psnr = self.psnr(self.postprocess(images), self.postprocess(outputs_merged))
                mae = (torch.sum(torch.abs(images - outputs_merged)) / torch.sum(images)).float()

                logs.append(('psnr', round(psnr.item(), 4)))
                logs.append(('mae', round(mae.item(), 4)))

                self.inpaint_model.backward(gen_loss, dis_loss)
                current_lr = self.inpaint_model.gen_optimizer.param_groups[0]['lr']
                iteration = self.inpaint_model.iteration

                if iteration >= max_iteration:
                    keep_training = False
                    break

                logs = [
                           ("epoch", epoch),
                           ("iter", iteration),
                           ("lr", current_lr)
                       ] + logs

                progbar.add(len(images),
                            values=logs if self.config.VERBOSE else [x for x in logs if not x[0].startswith('l_')])

                epoch_dir = f'epoch_{epoch:04d}' 
                sample_path = os.path.join(self.results_path, 'sample')
                if self.config.TRAINSAVE_INTERVAL and iteration % self.config.TRAINSAVE_INTERVAL == 0:
                    # create_dir(self.results_path)
                    inputs = (images * (1 - masks))
                    images_joint = stitch_images(
                        self.postprocess(images),
                        self.postprocess(inputs),
                        self.postprocess(outputs_img),
                        self.postprocess(outputs_merged),
                        img_per_row=1
                    )

                    path_joint = os.path.join(sample_path, epoch_dir, 'train')
                    name = f'iter_{iteration:07d}.png'
                    create_dir(path_joint)
                    images_joint.save(os.path.join(path_joint, name))

                    LOGGER.info(f" Saving training samples: {os.path.join(path_joint, name)} ")

                # log model at checkpoints
                if self.config.LOG_INTERVAL and iteration % self.config.LOG_INTERVAL == 0:
                    self.log(logs, self.log_file)

                # save model at checkpoints
                if self.config.SAVE_INTERVAL and iteration % self.config.SAVE_INTERVAL == 0:

                    LOGGER.info(f"\n--- Running training val at epoch {epoch}  iteration {iteration} ---")
                    self.inpaint_model.eval()  # Set model to evaluation mode

                    val_loader = DataLoader(
                        dataset=self.val_dataset,
                        batch_size=1,
                    )
                    LOGGER.info(f"Val dataset size: {len(self.val_dataset)}")

                    output_batches = {
                        "image": [],
                        "mask": [],
                        "output": []
                    }
                    with torch.no_grad():
                        for i, items_test in enumerate(val_loader):
                            
                            images_test, masks_test = self.cuda(*items_test)

                            outputs_img_test = self.inpaint_model(images_test, masks_test)
                            outputs_merged_test = (outputs_img_test * masks_test) + (images_test * (1 - masks_test))

                            output_batches["image"].append(images_test.detach().cpu())
                            output_batches["mask"].append(masks_test.detach().cpu())
                            output_batches["output"].append(outputs_merged_test.detach().cpu())

                        for key in output_batches:
                            output_batches[key] = torch.cat(output_batches[key], dim=0)  # (B, C, H, W)

                        path_joint = os.path.join(sample_path, epoch_dir, 'val')
                        create_dir(path_joint)

                        B = output_batches['output'].shape[0]
                        num_samples = 6

                        selected_indices = torch.linspace(0, B - 1, steps=num_samples).long().tolist()

                        gt = output_batches['image'][selected_indices]
                        mask = output_batches['mask'][selected_indices]
                        inpainted = output_batches['output'][selected_indices]
                        masked = gt * (1 - mask)

                        vis_img = stitch_images(
                            self.postprocess(gt),
                            self.postprocess(masked),
                            self.postprocess(inpainted),
                            img_per_row=1
                        )

                        vis_img.save(os.path.join(path_joint, f'iter_{iteration:07d}.png'))

                    dataset = EvalDataset(in_memory_batch=output_batches)
                    dataloader = DataLoader(
                        dataset,
                        batch_size=16,
                        shuffle=False,
                        num_workers=4,
                        pin_memory=True,
                    )

                    evaluator = Evaluator(device='cuda' if torch.cuda.is_available() else 'cpu', withmask=True)
                    metrics = evaluator.forward(dataloader)
                    F4scores, metric_lines = evaluator.print_metrics(metrics)
                    LOGGER.info("Evalautor done")
                    LOGGER.info(metric_lines)
                    del output_batches

                    self.save(iteration, F4scores)
                    LOGGER.info(f"In-training val done")

                    # scheduler
                    self.inpaint_model.gen_scheduler.step(F4scores)
                    current_lr = self.inpaint_model.gen_optimizer.param_groups[0]['lr']
                    LOGGER.info(f"Updated learning rate: {current_lr:.8f}")
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
                    LOGGER.info("Validation complete, clearing GPU cache.")

                    self.inpaint_model.train()  # Set model back to train mode

        LOGGER.info('\nEnd training....')

    def test(self):
        LOGGER.info(f"--- Test  ---")

        self.inpaint_model.eval()
        test_loader = DataLoader(
            dataset=self.test_dataset,
            batch_size=1,
        )
        LOGGER.info(f":test lenth {len(self.test_dataset)}")

        index = 0
        total_inference_time = 0  
        num_images = 0  

        LOGGER.info(f"Result path: {self.results_path}")
        for idx, items in enumerate(tqdm(test_loader, desc="Testing", leave=False)):
            images, masks = self.cuda(*items)
            index += 1
            num_images += images.shape[0] 

            torch.cuda.synchronize()
            tsince = time.time()
            outputs_img = self.inpaint_model(images, masks)
            torch.cuda.synchronize()
            ttime_elapsed = (time.time() - tsince) * 1000  
            total_inference_time += ttime_elapsed

            outputs_merged = (outputs_img * masks) + (images * (1 - masks))

            path_masked = os.path.join(self.results_path, 'masked')
            path_result = os.path.join(self.results_path, 'result')

            name = self.test_dataset.load_name(index - 1)[:-4] + '.png'

            create_dir(path_masked)
            create_dir(path_result)

            masked_images = self.postprocess(images * (1 - masks) + masks)[0]
            images_result = self.postprocess(outputs_merged)[0]

            imsave(masked_images, os.path.join(path_masked, name))
            imsave(images_result, os.path.join(path_result, name))


        avg_time = total_inference_time / num_images
        LOGGER.info(f'Average inference time per image: {avg_time:.2f} ms over {num_images} images.')

        LOGGER.info('--- End Testing ---')

    def eval(self):
        LOGGER.info(f"--- Eval  ---")

    def log(self, logs, path):
        with open(path, 'a') as f:
            f.write('%s\n' % ' '.join([str(item[1]) for item in logs]))


    def cuda(self, *args):
        return (item.to(self.config.DEVICE) for item in args)

    def postprocess(self, img):
        img = img * 255.0
        img = img.permute(0, 2, 3, 1)
        img = img.int()
        return img

    def metric(self, gt, pre):
        pre = pre.clamp_(0, 1) * 255.0
        pre = pre.permute(0, 2, 3, 1)
        pre = pre.detach().cpu().numpy().astype(np.uint8)[0]

        gt = gt.clamp_(0, 1) * 255.0
        gt = gt.permute(0, 2, 3, 1)
        gt = gt.cpu().detach().numpy().astype(np.uint8)[0]

        psnr = min(100, compare_psnr(gt, pre))

        ssim = compare_ssim(gt, pre, multichannel=True, channel_axis=-1, data_range=255)

        return psnr, ssim
