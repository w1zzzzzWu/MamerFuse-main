import os
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from MAE.util import misc

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

# define the utils
imagenet_mean = np.array([0.485, 0.456, 0.406])
imagenet_std = np.array([0.229, 0.224, 0.225])

def save_image(image_tensor, title, save_dir, name_prefix):
    assert image_tensor.shape[2] == 3
    image_np = torch.clip((image_tensor * imagenet_std + imagenet_mean) * 255, 0, 255).int().cpu().numpy()
    # image_np = torch.clip(image_tensor * 255, 0, 255).int().cpu().numpy()
    image_np = image_np.astype(np.uint8)
    save_path = os.path.join(save_dir, f"{name_prefix}_{title}.png")
    cv2.imwrite(save_path, cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))
    print(f"Saved: {save_path}")

def prepare_model(chkpt_dir, arch='mae_vit_base_patch16', random_mask=False, finetune=False):
    model = misc.get_mae_model(arch, random_mask=random_mask, finetune=finetune)
    checkpoint = torch.load(chkpt_dir, map_location='cpu')
    msg = model.load_state_dict(checkpoint['model'], strict=False)
    print(msg)
    return model

def run_one_image(img, mask, model, save_dir, name_prefix):
    os.makedirs(save_dir, exist_ok=True)

    x = torch.tensor(img, dtype=torch.float32).cuda()
    mask = torch.tensor(mask, dtype=torch.float32).cuda()

    x = x.unsqueeze(dim=0)
    x = torch.einsum('nhwc->nchw', x)
    mask = mask.reshape(1, 1, mask.shape[0], mask.shape[1])

    y, mask2 = model.forward_return_image(x.float(), mask)
    y = torch.einsum('nchw->nhwc', y).detach()

    mask = mask.detach().repeat(1, 3, 1, 1)
    mask = torch.einsum('nchw->nhwc', mask).detach()

    mask2 = mask2.detach().unsqueeze(-1).repeat(1, 1, model.patch_embed.patch_size[0]**2 * 3)
    mask2 = model.unpatchify(mask2)
    mask2 = torch.einsum('nchw->nhwc', mask2).detach()

    x = torch.einsum('nchw->nhwc', x)

    im_masked = x * (1 - mask)
    im_masked2 = x * (1 - mask2)
    im_paste = x * (1 - mask) + y * mask

    x, y, im_masked, im_masked2, im_paste = x.cpu(), y.cpu(), im_masked.cpu(), im_masked2.cpu(), im_paste.cpu()

    # 保存图像
    save_image(x[0], "original", save_dir, name_prefix)
    save_image(im_masked[0], "masked", save_dir, name_prefix)
    save_image(im_masked2[0], "enlarged_masked", save_dir, name_prefix)
    save_image(y[0], "reconstruction", save_dir, name_prefix)
    save_image(im_paste[0], "reconstruction_visible", save_dir, name_prefix)

if __name__ == "__main__":
    # load model
    chkpt_dir = '/home/student01/MxT-main/ckpts/mae_wo_cls_wo_pixnorm/checkpoint-last.pth'
    model_mae = prepare_model(chkpt_dir, random_mask=False, finetune=False).cuda()
    print('Model loaded.')

    # load an image/mask
    img_path = '/home/student01/MxT-main/datasets/CelebA-HQ/test_256/00039.jpg'
    mask_path = '/home/wh/DIR/dataset/test_face/g6/image_0001_mask.png'

    img = cv2.imread(img_path)[:,:,::-1]
    img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
    img = np.array(img) / 255.
    img = img - imagenet_mean
    img = img / imagenet_std

    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    mask = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_NEAREST)
    mask = np.array(mask) / 255.

    # save directory & name
    save_dir = "./outputs"
    name_prefix = "sample_00039"

    run_one_image(img, mask, model_mae, save_dir, name_prefix)
