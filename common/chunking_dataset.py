# FIXME: move to code snippets dir

import torch
import math
import random
import logging

class ChunkingDataset(torch.utils.data.IterableDataset):
    """
    Enables splitting the entire dataset sweep into multiple epochs
    """

    def __init__(self, data, shuffle=False, subset_len=None):
        """
        :param data: list-like dataset
        :param shuffle: by using this dataset type, the shuffling has to be done within the dataset instead of the data loader
        :param subset_len: length of each chunk
        """
        logging.debug(f"ChunkingDataset __init__ shuffle {shuffle}, subset_len {subset_len}")
        self.all_data = list(data) # FIXME: maybe an iterator implementation is memory saving
        self.n_calls = 0 # used to calculate the current epoch's dataset chunk's indexes
        self.shuffle = shuffle # shuffles the datapoints within the chunk
        assert subset_len is not None
        self.subset_len = int(subset_len) # forcing int cast because cmdline arg may enter as string
        self.subset = []

    def __iter__(self):
        """
        This dataset type is iterable instead of being indexable.
        This provides some disadvantages, for example the data loader will not be able to
        shuffle the dataset, and this needs to be performed within the dataset (see shuffle init parameter)
        :return:
        """
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

def _command_line_test():
    """
    called if the python module is run directly.
    Simply showcases the basic functioning of ChunkingDataset
    """
    logging.basicConfig( level=logging.DEBUG )
    def make_dataset():
        # a dataset of size 30 and a subset_len of 15 are enough to show how
        # ChunkingDataset splits the dataset sweeps into two epochs
        dataset = ChunkingDataset(range(30), shuffle=True, subset_len=15)
        return dataset

    def run_epochs(X):
        for e in range(7): # 7 is a number of epochs
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
    _command_line_test()