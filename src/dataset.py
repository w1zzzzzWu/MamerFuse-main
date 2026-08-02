import logging
import os
import glob
import torch
import random
import numpy as np

from typing import Dict, Optional
from torchvision import transforms
from PIL import Image
import cv2

LOGGER = logging.getLogger(__name__)


class EvalDataset(torch.utils.data.Dataset):
    def __init__(self,
                 gt_folder: Optional[str] = None,
                 output_folder: Optional[str] = None,
                 mask_folder: Optional[str] = None,
                 in_memory_batch: Optional[Dict[str, torch.Tensor]] = None
    ):

        self.in_memory_mode = (in_memory_batch is not None)
        if self.in_memory_mode:
            self.images = in_memory_batch['image']  # (B, C, H, W)
            self.outputs = in_memory_batch['output']
            self.masks = in_memory_batch.get('mask', None)
            self.num_samples = self.images.shape[0]
        else:
            self.gt_folder = self.load_flist(gt_folder)
            self.output_folder = self.load_flist(output_folder)
            if mask_folder is not None:
                self.mask_folder = self.load_flist(mask_folder)
            else:
                self.mask_folder = None        

    
    def load_flist(self, flist):
        if isinstance(flist, list):
            return flist

        # flist: image file path, image directory path, text file flist path
        if isinstance(flist, str):
            if os.path.isdir(flist):
                flist = list(glob.glob(flist + '/*.jpg')) + list(glob.glob(flist + '/*.png'))
                flist.sort()
                return flist

            if os.path.isfile(flist):
                try:
                    return np.genfromtxt(flist, dtype=str, encoding='utf-8')
                except Exception as e:
                    print(e)
                    return [flist]

        return None    

    def __len__(self):
        return self.num_samples if self.in_memory_mode else len(self.output_folder)


    def __getitem__(self, idx):
        if self.in_memory_mode:
            sample = {
                'image': self.images[idx],
                'output': self.outputs[idx]
            }
            if self.masks is not None:
                sample['mask'] = self.masks[idx]
 
            return sample
        else:
            return self._load_from_disk(idx)

    def _load_from_disk(self, idx):

        out_path = self.output_folder[idx]
        gt_path = self.gt_folder[idx]


        gt_img = Image.open(gt_path).convert('RGB')
        out_img = Image.open(out_path).convert('RGB')

        sample = {'image': gt_img, 'output': out_img}

        if self.mask_folder is not None:
            mask_path = self.mask_folder[idx]
            mask_img = Image.open(mask_path).convert('L')

            mask_tensor = transforms.ToTensor()(mask_img)  
            mask_tensor = (mask_tensor > 0.5).float()      

            sample['mask'] = mask_tensor
            print("masks in evaldata:", type(sample['mask']), torch.unique(sample['mask']), sample['mask'].float().mean()) 

        sample['image'] = transforms.ToTensor()(sample['image'])
        sample['output'] = transforms.ToTensor()(sample['output'])

        return sample


class Dataset(torch.utils.data.Dataset):
    def __init__(self, config, flist, mask_flist, augment=True, training=True):
        super(Dataset, self).__init__()
        self.config = config
        self.augment = augment
        self.training = training

        self.data = self.load_flist(flist)
        self.mask_data = self.load_flist(mask_flist)

        self.input_size = config.INPUT_SIZE
        self.mask = config.MASK

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        item = self.load_item(index)
        return item

    def load_name(self, index):
        name = self.data[index]
        return os.path.basename(name)

    def load_item(self, index):

        size = self.input_size
        # load image using PIL
        img = Image.open(self.data[index]).convert('RGB')  
        # resize if needed
        if size != 0:
            img = img.resize((size, size), resample=Image.BICUBIC)
            
        img_tensor = transforms.ToTensor()(img)
        # load mask
        mask = self.load_mask(img, index)
        mask_tensor = transforms.ToTensor()(mask)  
        mask_tensor = (mask_tensor > 0.5).float()    
 
        return img_tensor, mask_tensor

    def load_mask(self, img, index):
        imgw, imgh = img.size  
        mask_type = self.mask

        if not self.augment:
            mask_type = 4

        if mask_type == 0:
            return np.zeros((self.config.INPUT_SIZE, self.config.INPUT_SIZE))

        #  hybrid random mask
        if mask_type == 1:
            if random.random() < 0.5:
                mask_index = random.randint(0, len(self.mask_data) - 1)
                mask = Image.open(self.mask_data[mask_index]).convert('L')
                mask = mask.resize((imgw, imgh), resample=Image.NEAREST)
                return mask
            else:
                return self.get_mask_with_limit(imgw, imgh)

        # center mask
        if mask_type == 2:
            return create_mask(imgw, imgh, imgw // 2, imgh // 2, x=imgw // 4, y=imgh // 4)

        # external
        if mask_type == 3:
            mask_index = random.randint(0, len(self.mask_data) - 1)
            mask = Image.open(self.mask_data[mask_index]).convert('L')
            mask = mask.resize((imgw, imgh), resample=Image.NEAREST)

            return mask

        # test mode: load mask non random
        if mask_type == 4:
            mask = Image.open(self.mask_data[index % len(self.mask_data)]).convert('L')
            mask = mask.resize((imgw, imgh), resample=Image.NEAREST)
            
            return mask

    def load_flist(self, flist):
        if isinstance(flist, list):
            return flist

        # flist: image file path, image directory path, text file flist path
        if isinstance(flist, str):
            if os.path.isdir(flist):
                flist = list(glob.glob(flist + '/*.jpg')) + list(glob.glob(flist + '/*.png'))
                flist.sort()
                return flist

            if os.path.isfile(flist):
                try:
                    return np.genfromtxt(flist, dtype=str, encoding='utf-8')
                except Exception as e:
                    print(e)
                    return [flist]

        return []


    def get_mask_with_limit(self, imgw, imgh, max_ratio=0.6):

        # mask1: Ensure it's 0 or 255, uint8
        mask1 = create_mask(imgw, imgh, imgw // 2, imgh // 2)
        mask1 = (mask1 > 0).astype(np.uint8) * 255

        # Generate stroke mask with retry mechanism to keep within area limit
        total_area = imgw * imgh
        max_mask_pixels = int(total_area * max_ratio)
        attempt = 0
        max_attempts = 10  # Avoid infinite loop

        max_parts = 5
        maxLength = 100
        maxBrushWidth = 15

        while attempt < max_attempts:
            attempt += 1
            safeBrushWidth = max(int(maxBrushWidth), 11)
            safemax_parts = max(int(max_parts), 2)

            mask2_float = generate_stroke_mask(
                [imgh, imgw],
                max_parts=safemax_parts,
                maxVertex=25,
                maxLength=int(maxLength),
                maxBrushWidth=safeBrushWidth,
                maxAngle=360
            )
            mask2_float = np.squeeze(mask2_float, axis=-1)  # From (H, W, 1) to (H, W)
            mask2_resized = np.array(Image.fromarray(mask2_float).resize((imgh, imgw)))
            mask2 = (mask2_resized > 0).astype(np.uint8) * 255

            combined = np.maximum(mask1, mask2)
            mask_area = np.sum(combined == 255)
            if maxBrushWidth > 10:
                maxBrushWidth *= 0.8
            elif maxLength > 30:
                maxLength *= 0.8
            elif max_parts > 2:
                max_parts -= 1
            if mask_area <= max_mask_pixels:
                return combined
            # return combined


def create_mask(width, height, mask_width, mask_height, x=None, y=None):
    mask = np.zeros((height, width))
    mask_height = random.randint(0, height - mask_height)
    mask_width = random.randint(0, width - mask_width)
    mask_x = x if x is not None else random.randint(0, width - mask_width)
    mask_y = y if y is not None else random.randint(0, height - mask_height)
    mask[mask_y:mask_y + mask_height, mask_x:mask_x + mask_width] = 1
    return mask


def generate_stroke_mask(im_size, max_parts=10, maxVertex=25, maxLength=100, maxBrushWidth=24, maxAngle=360):
    mask = np.zeros((im_size[0], im_size[1], 1), dtype=np.float32)
    parts = random.randint(2, max_parts)
    for i in range(parts):
        mask = mask + np_free_form_mask(maxVertex, maxLength, maxBrushWidth, maxAngle, im_size[0], im_size[1])
    mask = np.minimum(mask, 1.0)
    # mask = np.concatenate([mask, mask, mask], axis = 2)
    return mask


def np_free_form_mask(maxVertex, maxLength, maxBrushWidth, maxAngle, h, w):
    mask = np.zeros((h, w, 1), np.float32)
    numVertex = np.random.randint(maxVertex + 1)
    startY = np.random.randint(h)
    startX = np.random.randint(w)
    brushWidth = 0
    for i in range(numVertex):
        angle = np.random.randint(maxAngle + 1)
        angle = angle / 360.0 * 2 * np.pi
        if i % 2 == 0:
            angle = 2 * np.pi - angle
        length = np.random.randint(maxLength + 1)
        brushWidth = np.random.randint(10, maxBrushWidth + 1) // 2 * 2
        nextY = startY + length * np.cos(angle)
        nextX = startX + length * np.sin(angle)
        nextY = np.maximum(np.minimum(nextY, h - 1), 0).astype(int)
        nextX = np.maximum(np.minimum(nextX, w - 1), 0).astype(int)
        cv2.line(mask, (startY, startX), (nextY, nextX), 1, brushWidth)
        cv2.circle(mask, (startY, startX), brushWidth // 2, 2)
        startY, startX = nextY, nextX
    cv2.circle(mask, (startY, startX), brushWidth // 2, 2)
    return mask


def image_transforms(load_size):
    return transforms.Compose([

        transforms.Resize(size=load_size, interpolation=Image.BILINEAR),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
