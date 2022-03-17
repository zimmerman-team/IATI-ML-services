import numpy as np

from common import utils
from common import persistency

import logging

class Tsets(utils.Collection): # FIXME: rename in splits?
    tsets_names = ('train', 'test')

    def load_data(self):
        """
        loads the data for this tset from mongodb(+gridfs)
        :return:
        """
        total_n_datapoints = 0
        for which_tset in self.tsets_names:
            # FIXME: maybe make a Tset object so I can get rid of these underscores
            gridfs_fileid = self[f"{which_tset}_npa_file_id"]

            self[which_tset] = persistency.load_npa(file_id=gridfs_fileid)
            if self[which_tset] is None:
                raise Exception(f"Didn't find {gridfs_fileid}")

            total_n_datapoints += self[which_tset].shape[0]

        if self.cap not in (None, False, 0):
            # the dataset_cap option in the configuration file allows to use
            # a smaller amount of datapoints in order to quickly debug a new model.
            for which_tset in self.tsets_names:
                orig = self[which_tset].shape[0]
                tset_fraction = float(orig)/float(total_n_datapoints)
                tset_cap = int(tset_fraction*self.cap)
                self[which_tset] = self[which_tset][:tset_cap, :]

    def __init__(
            self,
            spec,
            **kwargs
    ):
        self.spec = spec
        kwargs.update(dict.fromkeys(self.tsets_names, None))
        self.creation_time = kwargs['creation_time']
        super().__init__(**kwargs)

        self.load_data()

        for which_tset in self.tsets_names:

            # IMPORTANT: The input data is always considered as having the first column
            # having the set index. What `.with_set_index` does is to remove it from further
            # consideration in the Tsets
            self[which_tset + "_without_set_index"] = self[which_tset][:, 1:]

            if self.with_set_index is False:
                # removes the set_index column from the glued-up tensor
                self[which_tset] = self[which_tset+"_without_set_index"]


            # makes a list of tensors, each of which contains the data of a field of
            # the relation
            sections = spec.divide(
                self[which_tset],
                with_set_index=self.with_set_index
            )
            logging.debug("divided:"+str([s.shape for s in sections]))
            if self.with_set_index is True:
                # enriches the Tsets object with train_set_index and test_set_index
                # properties that contain a 1D-vector of set ids - positional, as
                # it refers to the item-rows of the data tensors
                self[which_tset+"_set_index"] = sections[0].squeeze(1)
            self[which_tset+"_sections"] = sections  # FIXME: needed to keep this?

            if which_tset == 'train':
                # the preprocessing scaling functions are trained only
                # on the training data, of course
                self._make_and_fit_scalers(sections, self.with_set_index)

            # WARNING: this relies on the fact that the training
            # set is going to be processed before the others
            (
                self[which_tset+"_scaled"],
                self[which_tset+"_scaled_without_set_index"]
            ) = self._scale(sections, self.with_set_index)

    def print_shapes(self):
        for which_tset in self.tsets_names:
            logging.debug(which_tset+".shape"+str(self[which_tset].shape))
            logging.debug(which_tset+"_scaled.shape"+str(self[which_tset+"_scaled"].shape))

    @property
    def item_dim(self):
        return self.train_without_set_index.shape[1] # because rows = items and cols = features

    def n_sets(self, which_tset):
        """
        :param which_tset: "train" or "test"
        :return: the number of sets contained in that dataset split
        """
        assert self.with_set_index, "to use this attribute with_set_index needs to be True"

        # WARNING: this is based on the assumption that the set indexes
        # are presented as ordered, hence the last item should have the
        # last set's id, hence the largest id
        return self[which_tset+"_set_index"][-1]

    def sets_intervals(self, which_tset):
        """
        returns a numpy array in which the rows are index intervals of item rows,
        represented by two numbers:
        the index of the start item for a set and the index for the ending item
        for that same set.
        Hence, the length of the intervals list is going to be the number of sets.
        This is used, for example, by a data loader that needs to return all
        all the contiguous items that belong to one (or, eventually multiple) set.
        :param which_tset: "train" or "test"
        :return: numpy array of shape (n_sets,2) containing the item intervals of each set
        """
        assert self.with_set_index, "to use this attribute with_set_index needs to be True"
        indexes = self[which_tset+'_set_index']
        ret = []
        curr = indexes[0]
        prev = curr
        all_count = 0
        curr_start = 0
        for i, curr in enumerate(indexes):
            if curr != prev:
                ret.append([curr_start, all_count])
                curr_start = all_count
            all_count += 1
            prev = curr
        ret.append([curr_start, all_count])
        return np.array(ret)

    def _make_and_fit_scalers(self, sections, with_set_index):
        if with_set_index:
            # but the index does not need to be scaled
            sections = sections[1:]
        for field, section in zip(self.spec.fields, sections):
            # FIXME: is having the trained scaler in the field a good idea??
            field.make_and_fit_scaler(section)

    def _scale(self, sections, with_set_index):
        """
        Uses the previously-trained scalers to scale the given data,
        which is presented as a list of per-field tensors (sections)
        :param sections: list of per-field tensors
        :return: the glued-up tensor which contains the horizontally-stacked
        scaled tensors of each field.
        """
        scaled = []
        if with_set_index:

            # adding the index to the scaled quantities
            scaled.append(sections[0])

            # but the index does not need to be scaled
            sections = sections[1:]

        for field, section in zip(self.spec.fields, sections):
            assert field.n_features == section.shape[1], \
                f"mismatch between field n_features and n columns of section: {field.n_features} != {section.shape[1]}"
            # FIXME: is having the trained scaler in the field a good idea??
            logging.debug("scaling field"+str(field)+" section.shape:"+str(section.shape))
            section_scaled = field.scaler.transform(section)
            logging.debug('resulting section_scaled.shape:'+str(section_scaled.shape))
            scaled.append(section_scaled)
        logging.debug("scaled sections:"+str([s.shape for s in scaled]))

        ret = utils.glue(scaled)
        ret_without_set_index = utils.glue(scaled[1:]) # exclude first section, which is the set index
        return ret,ret_without_set_index
