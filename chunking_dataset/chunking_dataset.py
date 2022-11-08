# FIXME: move to code snippets dir

import torch
import math
import random
import logging
import mlflow

class ChunkingDataset(torch.utils.data.IterableDataset):
    """
    Enables splitting the entire dataset sweep into multiple epochs
    """

    def __init__(self, data, shuffle=False, chunk_len=None, log_mlflow=False):
        """
        :param data: list-like dataset
        :param shuffle: by using this dataset type, the shuffling has to be done within the dataset instead of the data loader
        :param chunk_len: length of each chunk
        """
        logging.debug(f"ChunkingDataset __init__ shuffle {shuffle}, chunk_len {chunk_len}")
        self.log_mlflow = log_mlflow
        self.all_data = list(data) # FIXME: maybe an iterator implementation is memory saving

        # used to calculate the current epoch's dataset chunk's indexes
        # it's a constantly increasing index of how many items __iter__ is being called
        self.n_calls = 0

        self.shuffle = shuffle # shuffles the datapoints within the chunk
        assert chunk_len is not None
        self.chunk_len = int(chunk_len) # forcing int cast because cmdline arg may enter as string
        self.chunk = []
        self._create_slicing_indexes()
        self.log_param("chunking_dataset_shuffle",self.shuffle)
        self.log_param("chunking_dataset_chunk_len",self.chunk_len)
        self.log_param("chunking_dataset_all_data_len",self.all_data_len)
        self.log_param("chunking_dataset_n_chunks",self.n_chunks)

    def _create_slicing_indexes(self):
        self._slicing_indexes = []
        for chunk_index in range(self.n_chunks):
            datapoint_start = chunk_index * self.chunk_len
            datapoint_end = (chunk_index + 1) * self.chunk_len
            self._slicing_indexes.append((datapoint_start, datapoint_end))

    def log_param(self,name,val):
        """
        param logging wrapper around mlflow
        """
        if self.log_mlflow:
            mlflow.log_param(name,val)

    def log_metric(self,name,val):
        """
        metric logging wrapper around mlflow
        """
        if self.log_mlflow:
            mlflow.log_metric(name,val)

    @property
    def all_data_len(self):
        """
        :return: amount of datapoints
        """
        return len(self.all_data)

    @property
    def n_chunks(self):
        """
        :return: amount of chunks
        """
        return int(math.ceil(self.all_data_len / self.chunk_len))

    def __iter__(self):
        """
        This dataset type is iterable instead of being indexable.
        This provides some disadvantages, for example the data loader will not be able to
        shuffle the dataset, and this needs to be performed within the dataset (see shuffle init parameter)
        :return:
        """
        logging.debug(f"ChunkingDataset __iter__ id {id(self)}, all_data_len {self.all_data_len}, chunk_len {self.chunk_len}")

        logging.debug(f"ChunkingDataset __iter__ n_chunks {self.n_chunks}, n_calls {self.n_calls}")
        self.log_metric("chunking_dataset_n_calls",self.n_calls)
        chunk_index = self.n_calls % self.n_chunks
        if chunk_index == 0:
            # shuffle the which chunks to select at every entire dataset sweep
            random.shuffle(self._slicing_indexes)
        self.log_metric("chunking_dataset_chunk_index",chunk_index)
        logging.debug(f"ChunkingDataset __iter__ chunk_index {chunk_index}")
        datapoint_start, datapoint_end = self._slicing_indexes[chunk_index]
        self.log_metric("chunking_dataset_datapoint_start",datapoint_start)
        self.log_metric("chunking_dataset_datapoint_end",datapoint_end)
        progress = float(chunk_index)/float(self.n_chunks)
        self.log_metric("chunking_dataset_progress",progress)

        logging.debug(f"ChunkingDataset __iter__ , datapoint_start {datapoint_start}, datapoint_end {datapoint_end}")
        self.chunk = self.all_data[datapoint_start:datapoint_end]
        if self.shuffle:
            random.shuffle(self.chunk)
        self.n_calls += 1
        return iter(self.chunk)

def _command_line_test():
    """
    called if the python module is run directly.
    Simply showcases the basic functioning of ChunkingDataset
    """
    logging.basicConfig( level=logging.DEBUG )
    def make_dataset():
        # a dataset of size 30 and a chunk_len of 15 are enough to show how
        # ChunkingDataset splits the dataset sweeps into two epochs
        dataset = ChunkingDataset(range(30), shuffle=True, chunk_len=15)
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