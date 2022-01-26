# FIXME: move to code snippets dir

import torch
import math
import random

parent_class = torch.utils.data.IterableDataset
class ChunkingDataset(parent_class):

    def __init__(self, data, shuffle=False, subset_len=None):
        print(f"__init__ shuffle {shuffle}, subset_len {subset_len}")
        self.all_data = [ x for x in data]
        self.n_calls = 0
        self.shuffle = shuffle
        assert subset_len is not None
        self.subset_len = subset_len
        self.subset = []

    def __iter__(self):
        n_starts = int(math.floor(len(self.all_data)/self.subset_len))
        section_index = self.n_calls % n_starts
        datapoint_start = section_index * self.subset_len
        datapoint_end = (section_index+1) * self.subset_len

        #print(f"__iter__ called; n_calls {self.n_calls}, n_starts {n_starts}, section_index {section_index}, datapoint_start {datapoint_start}, datapoint_end {datapoint_end}")
        self.subset = self.all_data[datapoint_start:datapoint_end]
        if self.shuffle:
            random.shuffle(self.subset)
        self.n_calls += 1
        return iter(self.subset)

class ChunkingDataLoader(torch.utils.data.DataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


def command_line_test():

    def make_dataset():
        dataset = ChunkingDataset(range(20), shuffle=True, subset_len=5)
        return dataset

    def run_epochs(X):
        for e in range(7):
            print("epoch", e, ":")
            for x in X:
                print(int(x), end=' ')
            print('.')

    print("iterating dataset..")
    dataset = make_dataset()
    run_epochs(dataset)
    print("done. iterating loader..")
    dataset = make_dataset()
    loader = torch.utils.data.DataLoader(
        dataset,
        shuffle=False, # shuffle is implemented in the dataset
        num_workers=4,
        pin_memory=False
    )
    run_epochs(loader)
    print("done.")

if __name__ == '__main__':
    command_line_test()