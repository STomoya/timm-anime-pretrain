import argparse
import glob
import json
import os
import shutil
from collections import Counter
from functools import partial

import tqdm
from sklearn.model_selection import train_test_split


def load_jsonl(path):
    with open(path, 'r') as fp:
        data = fp.read().strip().split('\n')
    data = list(map(json.loads, data))
    return data


def load_labels(path):
    with open(path, 'r') as fp:
        label_mapping_string = fp.read().strip().split('\n')
    labels = {name for name, *_ in (x.split(',') for x in label_mapping_string)}
    return labels


def split(iter, chunk_size):
    for i in range(0, len(iter), chunk_size):
        yield iter[i : i + chunk_size]


def save_chunks(csv_entries, data_split, folder, chunk_size=100_000):
    folder = os.path.join(folder, data_split)
    shutil.rmtree(folder)
    os.makedirs(folder, exist_ok=True)

    filenames = []
    for i, chunk in enumerate(split(csv_entries, chunk_size)):
        filename = os.path.join(folder, f'{data_split}-{i:04}.csv')
        with open(filename, 'w') as fp:
            fp.write('\n'.join(chunk))
        filenames.append(filename)

    return filenames


def convert_metadata_to_csv(metadata, image_root, label_set, safe=True, image_ext='png'):
    rating = metadata['rating']
    if safe and rating != 's':
        return

    id = metadata['id']
    id_split = int(id) % 1000
    created_at = metadata['created_at']  # datetime
    created_year = created_at[:4]
    image_path = os.path.join(image_root, created_year, f'{id_split:04}', f'{id}.{image_ext}')

    tags = metadata['tag_string_general'].split(' ')
    tags = list(set(tags) & label_set)

    if len(tags) == 0:
        return

    return ','.join([image_path, *tags])


def limit_tag_appearance(csv_lines, limit):
    import matplotlib.pyplot as plt

    def _visualize(most_common, total):
        label_names = [x[0] for x in most_common][:512]
        counts = [x[-1] / total for x in most_common][:512]
        x = range(len(counts))
        plt.bar(x, counts)
        plt.xticks(x, label_names, rotation=90)

    tag_counter = Counter()
    new_tag_counter = Counter()
    new_csv_lines = []
    for line in tqdm.tqdm(csv_lines, bar_format='{l_bar}{bar:15}{r_bar}', desc='Filtering training data'):
        tags = line.split(',')[1:]
        tag_counter.update(tags)
        if all(tag_counter[tag] <= limit for tag in tags):
            new_csv_lines.append(line)
            new_tag_counter.update(tags)

    plt.figure(figsize=(20, 12))
    _visualize(tag_counter.most_common(), len(csv_lines))
    _visualize(new_tag_counter.most_common(), len(new_csv_lines))
    plt.xlabel('tags')
    plt.ylabel('frequency')
    plt.savefig('frequency')
    plt.tight_layout()
    plt.close()

    return new_csv_lines


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('folder')
    parser.add_argument('label_mapping')
    parser.add_argument('--output', default='./data/dataset')
    parser.add_argument('--image-root', default='./data/images')
    parser.add_argument('--all', default=False, action='store_true')
    parser.add_argument('--random-state', default=3407, type=int)
    parser.add_argument('--val-size', default=10_000, type=float)
    parser.add_argument('--test-size', default=50_000, type=float)
    parser.add_argument('--chunk-size', default=100_000, type=int)
    parser.add_argument('--limit', default=None, type=int)
    return parser.parse_args()


def main():
    args = get_args()

    label_set = load_labels(args.label_mapping)
    convert_to_csv = partial(
        convert_metadata_to_csv, image_root=args.image_root, label_set=label_set, safe=not args.all
    )

    metadata_files = glob.glob(os.path.join(args.folder, '*.json'))
    all_csv_entries = []
    for metadata_file in metadata_files:
        metadata_list = load_jsonl(metadata_file)
        csv_entries = map(convert_to_csv, metadata_list)
        csv_entries = filter(lambda x: x is not None and os.path.exists(x.split(',')[0]), csv_entries)
        all_csv_entries.extend(csv_entries)

    total_size = len(all_csv_entries)
    val_size = int(total_size * args.val_size) if args.val_size < 1 else int(args.val_size)
    test_size = int(total_size * args.test_size) if args.test_size < 1 else int(args.test_size)
    val_test_size = val_size + test_size

    train, val_test = train_test_split(all_csv_entries, test_size=val_test_size, random_state=args.random_state)
    val, test = train_test_split(val_test, test_size=test_size, random_state=args.random_state)

    if args.limit:
        train = limit_tag_appearance(train, args.limit)

    print('Train samples     :', len(train))
    print('Validation samples:', len(val))
    print('Test samples      :', len(test))

    if not os.path.exists(args.output):
        os.makedirs(args.output, exist_ok=True)

    save_chunks(train, 'train', args.output, args.chunk_size)
    save_chunks(val, 'validation', args.output, args.chunk_size)
    save_chunks(test, 'test', args.output, args.chunk_size)


if __name__ == '__main__':
    main()
