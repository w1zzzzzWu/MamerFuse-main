import os
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, default='/home/student01/MxT-main/results_celeba/iter_0312000_f1_0.806525/g6/result/',help='')
parser.add_argument('--output', type=str, default='/home/student01/MxT-main/results_celeba/iter_0312000_f1_0.806525/results_60.flist',help='')
args = parser.parse_args()

ext = {'.jpg', '.png', '.jpeg', '.bmp', '.tif', '.tiff'}

images = []
for root, dirs, files in os.walk(args.path):
    print('loading ' + root)
    for file in files:
        if os.path.splitext(file)[1].lower() in ext:
            images.append(os.path.join(root, file))

images = sorted(images)

with open(args.output, 'a', encoding='utf-8') as f:
    for img in images:
        f.write(f"{img}\n")


