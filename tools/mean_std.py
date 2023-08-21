
import argparse
import glob
import os
import torch
from torchvision.transforms.v2 import functional as F
from PIL import Image
from joblib import Parallel, delayed


def load_image(path):
    image = Image.open(path).convert('RGB')
    image = F.to_image_tensor(image)
    image = F.convert_image_dtype(image)
    value = image.mean(dim=(-1, -2))
    return value


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('folder')
    return parser.parse_args()


def main():
    args = get_args()
    csv_files = glob.glob(os.path.join(args.folder, '*.csv'))

    all_image_paths = []
    for csv_file in csv_files:
        with open(csv_file, 'r') as fp:
            csv_lines = fp.read().strip().split('\n')
        image_paths = [line.split(',')[0] for line in csv_lines]
        all_image_paths.extend(image_paths)

    values = Parallel(n_jobs=-1, verbose=2)(delayed(load_image)(image_path) for image_path in all_image_paths)
    values = torch.stack(values)
    print(values.mean(dim=0))
    print(values.std(dim=0))


if __name__=='__main__':
    main()
