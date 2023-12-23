import argparse
import glob
import json
import os
from collections import Counter
from itertools import chain


def load_jsonl(path):
    with open(path, 'r') as fp:
        data = fp.read().strip().split('\n')
    data = list(map(json.loads, data))
    return data


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('folder')
    parser.add_argument('--output', default='./data/dataset')
    parser.add_argument('--num-labels', default=2048, type=int)
    parser.add_argument('--all', default=False, action='store_true')
    parser.add_argument('--vis', default=False, action='store_true')
    return parser.parse_args()


def main():
    args = get_args()

    counter = Counter()

    metadata_files = glob.glob(os.path.join(args.folder, '*.json'))
    total = 0
    for metadata_file in metadata_files:
        metadata_list = load_jsonl(metadata_file)
        if not args.all:
            metadata_list = list(filter(lambda x: x['rating'] == 's', metadata_list))
        total += len(metadata_list)
        tags = [x['tag_string_general'].split(' ') for x in metadata_list]
        counter.update(chain(*tags))

    most_common = [(x[1][0], x[0], x[1][1]) for x in enumerate(counter.most_common(args.num_labels))]
    label_mapping = [f'{x[0]},{x[1]},{x[2]}' for x in most_common]

    if not os.path.exists(args.output):
        os.makedirs(args.output, exist_ok=True)

    with open(os.path.join(args.output, 'label_mapping.csv'), 'w') as fp:
        fp.write('\n'.join(label_mapping))

    if args.vis:
        import matplotlib.pyplot as plt

        label_names = [x[0] for x in most_common][:512]
        counts = [x[-1] / total for x in most_common][:512]
        x = range(len(counts))

        plt.figure(figsize=(20, 12))
        plt.bar(x, counts)
        plt.xlabel('tags')
        plt.xticks(x, label_names, rotation=90)
        plt.ylabel('frequency')
        plt.savefig('frequency')
        plt.tight_layout()
        plt.close()


if __name__ == '__main__':
    main()
