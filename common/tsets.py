import numpy as np
import collections

from common import utils
from common import persistency


class Tsets(utils.Collection):
    tsets_names = ('train', 'test')

    def load_data(self):
        total_n_datapoints = 0
        for which_tset in self.tsets_names:
            # FIXME: maybe make a Tset object so I can get rid of these underscores
            gridfs_filename = f"{self.rel.name}_{which_tset}"

            self[which_tset] = persistency.load_npa(filename=gridfs_filename)
            if self[which_tset] is None:
                raise Exception(f"Didn't find {gridfs_filename}")

            total_n_datapoints += self[which_tset].shape[0]

        if self.cap is not None:
            # the dataset_cap option in the configuration file allows to use
            # a smaller amount of datapoints in order to quickly debug a new model.
            for which_tset in self.tsets_names:
                orig = self[which_tset].shape[0]
                tset_fraction = float(orig)/float(total_n_datapoints)
                tset_cap = int(tset_fraction*self.cap)
                self[which_tset] = self[which_tset][:tset_cap,:]


    def __init__(
            self,
            rel,
            **kwargs
    ):
        self.rel = rel
        kwargs.update(dict.fromkeys(self.tsets_names, None))
        super().__init__(**kwargs)

        self.load_data()

        for which_tset in self.tsets_names:

            if self.with_set_index is False:
                # removes the set_index column from the glued-up tensor
                self[which_tset] = self[which_tset][:, 1:]

            # makes a list of tensors, each of which contains the data of a field of
            # the relation
            sections = rel.divide(
                self[which_tset],
                with_set_index=self.with_set_index
            )
            print("divided:",[s.shape for s in sections])
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
            self[which_tset+"_scaled"] = self._scale(sections, self.with_set_index)

    def print_shapes(self):
        for which_tset in self.tsets_names:
            print(which_tset+".shape",self[which_tset].shape)
            print(which_tset+"_scaled.shape",self[which_tset+"_scaled"].shape)

    @property
    def item_dim(self):
        return self.train.shape[1]

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
        for field, section in zip(self.rel.fields, sections):
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

        for field, section in zip(self.rel.fields, sections):
            assert field.n_features == section.shape[1], \
                f"mismatch between field n_features and n columns of section: {field.n_features} != {section.shape[1]}"
            # FIXME: is having the trained scaler in the field a good idea??
            print("scaling field",field,"section.shape:",section.shape)
            section_scaled = field.scaler.transform(section)
            print('resulting section_scaled.shape:',section_scaled.shape)
            scaled.append(section_scaled)
        print("scaled sections:",[s.shape for s in scaled])

        ret = self.rel.glue(scaled)
        return ret
