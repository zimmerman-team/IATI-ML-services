# FIXME: move to code snippets dir

import torch
import math
import random
import logging

class ChunkingDataset(torch.utils.data.IterableDataset):

    def __init__(self, data, shuffle=False, subset_len=None):
        logging.debug(f"ChunkingDataset __init__ shuffle {shuffle}, subset_len {subset_len}")
        self.all_data = data
        self.n_calls = 0
        self.shuffle = shuffle
        assert subset_len is not None
        self.subset_len = subset_len
        self.subset = []

    def __iter__(self):
        all_data_len = len(self.all_data)
        logging.debug(f"ChunkingDataset __iter__ id {id(self)}, all_data_len {all_data_len}, subset_len {self.subset_len}")

        n_starts = int(math.ceil(all_data_len/self.subset_len))
        logging.debug(f"ChunkingDataset __iter__ n_starts {n_starts}, n_calls {self.n_calls}")
        section_index = self.n_calls % n_starts
        logging.debug(f"ChunkingDataset __iter__ section_index {section_index}")
        datapoint_start = section_index * self.subset_len
        datapoint_end = (section_index+1) * self.subset_len

        logging.debug(f"ChunkingDataset __iter__ , datapoint_start {datapoint_start}, datapoint_end {datapoint_end}")
        self.subset = self.all_data[datapoint_start:datapoint_end]
        if self.shuffle:
            random.shuffle(self.subset)
        self.n_calls += 1
        return iter(self.subset)

def command_line_test():

    logging.basicConfig( level=logging.DEBUG )
    def make_dataset():
        dataset = ChunkingDataset(range(30), shuffle=True, subset_len=15)
        return dataset

    def run_epochs(X):
        for e in range(7):
            print("epoch", e, ":")
            i=0
            for x in X:
                print(int(x), end=' ')
                i+=1
            print(f' i={i}.')

    print("iterating dataset..")
    dataset = make_dataset()
    run_epochs(dataset)
    print("done. iterating loader..")
    dataset = make_dataset()
    loader = torch.utils.data.DataLoader(
        dataset,
        shuffle=False, # shuffle is implemented in the dataset
        num_workers=0 # num_workers needs to be 0 to
    )
    run_epochs(loader)
    print("done.")

if __name__ == '__main__':
    command_line_test()