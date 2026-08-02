import logging
import os
import cv2
import random
import numpy as np
import torch
import argparse
from shutil import copyfile
from src.MamerFuse import MamerFuse
from src.config import Config
import datetime

from src.utils import create_dir

LOGGER = logging.getLogger(__name__)

def main(mode=None, args_list=None):
    r"""starts the model

    Args:
        mode (int): 1: train, 2: test, 3: eval, reads from config file if not specified
    """
    config = load_config(mode, args_list)
    if config.RESULTS is not None:
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H_%M_%S')
    if config.MODE == 1:
        results_path = os.path.join(config.RESULTS, f'train_res_{timestamp}')
        log_path = os.path.join(results_path, 'train.log')
    elif config.MODE == 2:
        results_path = config.RESULTS
        log_path = os.path.join(results_path, 'test.log')
    create_dir(results_path)
    copyfile(config.CONFIG_PATH, os.path.join(results_path, 'config.yml'))
    config.RESULTS = results_path

    logging.basicConfig(
        level=logging.INFO,  
        format='[%(asctime)s][%(name)s][%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_path, mode='a'),  
            logging.StreamHandler()  
        ]
    )

    # cuda visble devices
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(e) for e in config.GPU)
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    # init device
    if torch.cuda.is_available():
        LOGGER.info('Cuda is available')
        config.DEVICE = torch.device("cuda")
        torch.backends.cudnn.benchmark = True   # cudnn auto-tuner
    else:
        LOGGER.info('Cuda is unavailable, use cpu')
        config.DEVICE = torch.device("cpu")

    LOGGER.info(f'Current device:{torch.cuda.current_device()}')
    LOGGER.info(f'Device name: {torch.cuda.get_device_name(torch.cuda.current_device())}')

    # set cv2 running threads to 1 (prevents deadlocks with pytorch dataloader)
    cv2.setNumThreads(0)
    # initialize random seed
    torch.manual_seed(config.SEED)
    torch.cuda.manual_seed_all(config.SEED)
    np.random.seed(config.SEED)
    random.seed(config.SEED)

    # build the model and initialize
    model = MamerFuse(config)
    model.load()

    # model training
    if config.MODE == 1:
        config.print()
        model.train()

    # model test
    elif config.MODE == 2:
        model.test()

    # eval mode
    else:
        model.eval()


def load_config(mode=None, args_list=None):
    r"""loads model config
    Args:
        mode (int): 1: train, 2: test, reads from config file if not specified
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='./config/config_celeba.yml',
                        help='config_path')

    # test mode
    if mode == 2:
        parser.add_argument('--input', type=str, help='path to the input images directory or an input image')
        parser.add_argument('--mask', type=str, help='path to the masks directory or a mask file')
        parser.add_argument('--output', type=str, help='path to the output directory')
        parser.add_argument('--mae_path', type=str, help='path to the maemodel directory')
        parser.add_argument('--model_path', type=str, help='path to the model directory')

    if args_list is not None:
        args = parser.parse_args(args_list)
    else:
        args = parser.parse_args()

    if os.path.exists(args.config_path):
        config_path = args.config_path
    else:
        LOGGER.error(f"Config_path is None or Wrong: {args.config_path}")

    # load config file
    config = Config(config_path)
    LOGGER.info(f"config_path: {config_path}")

    # train mode
    if mode == 1:
        config.MODE = 1

    # test mode
    elif mode == 2:
        config.MODE = 2
        config.MODEL = 2
        if args.mae_path is not None and os.path.exists(args.mae_path):
            if args.model_path is not None:
                config.MAE_MODELPATH = args.mae_path
                config.MODELPATH = args.model_path
            else:
                LOGGER.error(f"model path is None or not exist: {args.model}")
                raise ValueError("model path is None or not exist")
        else:
            LOGGER.error(f"MAE model path is None or not exist: {args.maemodel}")
            raise ValueError("MAE model path is None or not exist")
        if args.input is not None:
            config.TEST_INPAINT_IMAGE_FLIST = args.input

        if args.mask is not None:
            config.TEST_MASK_FLIST = args.mask

        if args.output is not None:
            config.RESULTS = args.output

    # eval mode
    elif mode == 3:
        config.MODE = 3
        config.MODEL = 2

    return config


if __name__ == "__main__":
    main()
