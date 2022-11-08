# Chunking dataset

Chunking_dataset is a library based on pytorch-lightning that allows for 
splitting a training epoch in shorter epochs (chunks).

The purpose of this arrangement is when launching experiments on
differently-sized datasets. In that context it's important to be able to
compare training curves to evaluate the efficacy of certain models or
hyperparameter settings at the same epoch number.
That's because if a dataset is larger that would naturally lead
to a better training accuracy, defeating the purpose of comparison
over epochs.
 
The `chunking_dataset.ChunkingDataset` class extends the `IterableDataset`
object from the `torch` library.

Important constructor parameters are:

 * `shuffle` (boolean) will shuffle the datapoints within the chunk

 * `chunk_len` (integer) determines the length of each chunk
