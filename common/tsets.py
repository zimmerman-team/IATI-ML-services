import numpy as np

from common import utils
from common import persistency


class Tsets(utils.Collection):
    tsets_names = ('train', 'test')

    def __init__(
            self,
            rel,
            **kwargs
    ):
        self.rel = rel
        kwargs.update(dict.fromkeys(self.tsets_names, None))
        super().__init__(**kwargs)
        for which_tset in self.tsets_names:

            # FIXME: maybe make a Tset object so I can get rid of these underscores
            gridfs_filename = f"{rel.name}_{which_tset}"

            buf = persistency.load_npa(filename=gridfs_filename)
            if buf is not None:
                self[which_tset] = buf
                if self.with_set_index is False:
                    self[which_tset] = self[which_tset][:, 1:]
                sections = rel.divide(
                    self[which_tset],
                    with_set_index=self.with_set_index
                )
                if self.with_set_index is True:
                    self[which_tset+"_set_index"] = sections[0].squeeze(1)
                self[which_tset+"_sections"] = sections  # FIXME: needed to keep this?
                if which_tset == 'train':
                    self._make_and_fit_scalers(sections)

                self[which_tset+"_scaled"] = self._scale(sections)
            else:
                raise Exception(f"Didn't find {gridfs_filename}")


    def n_sets(self, which_tset):
        assert self.with_set_index, "to use this attribute with_set_index needs to be True"
        return self[which_tset+"_set_index"][-1]


    def sets_intervals(self, which_tset):
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

    def _make_and_fit_scalers(self, sections):
        for field, section in zip(self.rel.fields, sections):
            field.make_and_fit_scaler(section)


    def _scale(self, sections):
        scaled = []
        for field, section in zip(self.rel.fields, sections):
            section_scaled = field.scaler.transform(section)
            scaled.append(section_scaled)
        ret = self.rel.glue(scaled)
        return ret
