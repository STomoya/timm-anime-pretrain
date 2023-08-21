import glob
import os
from typing import Callable

import numpy as np
from torch.utils.data import Dataset
from PIL import Image


class MultiLabelCSV(Dataset):
    """multi-label dataset.

    Args:
        dataset_folder (str): path directory of CSV file(s) containing image paths and labels.
            File format:
                <image path 0>,<label 0>,<label 1>,<label 2>,...,<label n>
                <image path 1>,<label 0>,<label 1>,<label 2>,...,<label m>
        label_mapping (str): path to CSV file containing label name to index mapping information.
            File format:
                <label name 0>,<label index 0>
                <label name 1>,<label index 1>
        transform (Callable): callable for transforming images.
        label_is_indexed (bool, option): If True, labels in the dataset folder is already indexed. Default: False.
    """
    def __init__(self,
        dataset_folder: str, label_mapping: str, transform: Callable, label_is_indexed: bool=False
    ):
        super().__init__()
        csv_files = glob.glob(os.path.join(dataset_folder, '*.csv'))
        self.samples = []
        for csv_file in csv_files:
            with open(csv_file, 'r') as fp:
                self.samples.extend(fp.read().strip().split('\n'))
        with open(label_mapping, 'r') as fp:
            label_mapping = [line.split(',') for line in fp.read().strip().split('\n')]
        self.label_mapping = {name: int(index) for name, index, *_ in label_mapping}
        self.label_names = sorted(list(self.label_mapping.keys()), key=lambda key: self.label_mapping[key])
        self.label_name_set = set(self.label_mapping.keys())

        self.transform = transform
        self.label_is_indexed = label_is_indexed
        self.num_classes = len(self.label_mapping)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index].split(',')
        image_path = sample[0]
        labels = sample[1:]

        if self.label_is_indexed:
            labels = [int(label) for label in labels]
        else:
            labels = [self.label_mapping[label] for label in labels if label in self.label_name_set]
        multihot = np.zeros(self.num_classes, dtype=float)
        multihot[np.array(labels.copy(), dtype=int)] = 1.0

        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)

        return dict(image=image, label=multihot)
