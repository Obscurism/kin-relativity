import csv
import os
import numpy as np
import torch

from nsml import GPU_NUM
from progress import create_progressbar, finish_progressbar

from torch import FloatTensor, LongTensor, Size, zeros
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import Dataset


train_data_name = 'train_data'
train_label_name = 'train_label'
test_data_name = 'test_data'
word2vec_mapped_size = 50


def pad_sequence(sequence, max_len, vocab_size):
    pad_len = max_len - len(sequence)
    padding_sequence = np.zeros((pad_len, vocab_size))

    return np.concatenate([np.array(sequence), padding_sequence])


class KinDataset(Dataset):
    def __init__(self, dataset_path, preprocessor):
        print("[Dataset] Initializing Dataset...")

        self.loaded_data = []
        self.preprocessor = preprocessor

        data_path = os.path.join(dataset_path, 'train', train_data_name)  # Train data only
        data_label = os.path.join(dataset_path, 'train', train_label_name)

        with open(data_path, 'rt', 50 * 1024 * 1024, encoding='utf-8') as data_csvfile,\
                open(data_label, 'rt') as data_label:  # maximum buffer size == 50 MB
            # the fastest way of counting the number of total file lines.
            self.total_data_length = sum(1 for _ in data_csvfile)
            print("[Dataset] Found %d sequences. " % self.total_data_length)

            # reset file pointer to 0
            data_csvfile.seek(0)
            data_label.seek(0)
            read_line = csv.reader(data_csvfile)
            data_label = csv.reader(data_label)

            print("[Dataset] Preprocessing sequences...")
            progbar_preprocess = create_progressbar(self.total_data_length)

            for idx, (data, label) in enumerate(zip(read_line, data_label)):
                data = preprocessor.parse_sentence(data[0])
                label = int(label[0])
                self.loaded_data.append([data, label])

                progbar_preprocess.update(idx + 1)

            finish_progressbar(progbar_preprocess)

        print("[Dataset] Making dictionary from sequences...")
        preprocessor.make_dict(self)

        print("[Dataset] Padding sequences...")
        progbar_pad = create_progressbar(max_value=self.total_data_length)

        # loaded_data: Dataset
        # seqs: Data
        pad_lens = [max((len(seqs[0][0]), len(seqs[0][1]))) for seqs in self.loaded_data]

        for (idx, datum) in enumerate(self.loaded_data):
            mapped_vectors = self.preprocessor.map_vector(datum[0])
            self.loaded_data[idx] = [
                list(map(
                    lambda x: pad_sequence(x, pad_lens[idx], word2vec_mapped_size),
                    mapped_vectors
                )),
                datum[1]
            ];

            progbar_pad.update(idx + 1)

        finish_progressbar(progbar_pad)

        print("[Dataset] Done generating datasets.")

    def __len__(self):
        return self.total_data_length

    def __getitem__(self, idx):
        return self.loaded_data[idx]


def load_batch_input_to_memory(batch_input, has_targets=True):
    if has_targets:
        batch_input = [seq[0] for seq in batch_input]
    else:
        batch_input = batch_input

    # Get the length of each seq in your batch
    tensor_lens = LongTensor(list(map(lambda x: max((len(x[0]), len(x[1]))), batch_input)))

    # Zero-padded long-Matirx size of (B, T)
    tensor_seqs = zeros((len(batch_input), 2, tensor_lens.max(), 50)).float()
    for idx, seq in enumerate(batch_input):
        tensor_seqs[idx, :2, :tensor_lens[idx]] = FloatTensor(np.array(seq))

    return tensor_seqs, tensor_lens


def custom_collate_fn(data):
    """Creates mini-batch tensors from the list of list [batch_input, batch_target].
    We should build a custom collate_fn rather than using default collate_fn,
    because merging sequences (including padding) is not supported in default.
    Seqeuences are padded to the maximum length of mini-batch sequences (dynamic padding).
    Args:
        data: list of list [batch_input, batch_target].

    Refs:
        Thanks to yunjey. (https://github.com/yunjey/seq2seq-dataloader/blob/master/data_loader.py)
    """

    # data sampler returns an indices to select into mini-batch.
    # than the collate_function gather the selected data.
    # in this example, we use pack_padded_sequence so always ordering with descending
    # so the targets should be ordered same here.
    # sort the sequences as descending order to use 'pack_padded_sequence' before loading.
    data.sort(key=lambda x: len(x[0]), reverse=True)
    tensor_targets = LongTensor([pair[1] for pair in data])
    return [data, tensor_targets]


def get_dataloaders(dataset_path, batch_size, ratio_of_validation=0.2, shuffle=True, preprocessor=None):
    num_workers = 0  # Don't use multithread because of konlpy
    train_kin_dataset = KinDataset(dataset_path=dataset_path, preprocessor=preprocessor)

    num_train = len(train_kin_dataset)
    split_point = int(ratio_of_validation * num_train)

    indices = list(range(num_train))
    if shuffle:
        np.random.shuffle(indices)

    train_idx, val_idx = indices[split_point:], indices[: split_point]
    # Random sampling at every epoch without replacement in given indices
    train_sampler = SubsetRandomSampler(train_idx)
    val_sampler = SubsetRandomSampler(val_idx)

    # 'pin_memory=True' allows for you to use fast memory buffer with way of calling '.cuda(async=True)' function.
    pin_memory = False
    if GPU_NUM:
        pin_memory = True

    train_loader = torch.utils.data.DataLoader(
        train_kin_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers,
        pin_memory=pin_memory, collate_fn=custom_collate_fn)
    val_loader = torch.utils.data.DataLoader(
        train_kin_dataset, batch_size=batch_size, sampler=val_sampler, num_workers=num_workers,
        pin_memory=pin_memory, collate_fn=custom_collate_fn)

    return [train_loader, val_loader]


def read_test_file(dataset_path, preprocessor):
    data_path = os.path.join(dataset_path, 'test', test_data_name)
    loaded_data = []
    with open(data_path, 'rt', 50 * 1024 * 1024) as data_csvfile:  # maximum buffer size == 50 MB
        data_csvfile.seek(0)
        read_line = csv.reader(data_csvfile)
        for idx, data in enumerate(read_line):
            data = preprocessor.preprocess_test(data[0])
            loaded_data.append([data])
    return loaded_data
