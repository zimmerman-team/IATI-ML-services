import numpy as np

import utils

class Tsets(utils.Collection):
    tsets_names = ('train', 'test')
    def __init__(
            self,
            rel,
            **kwargs
    ):
        self.rel = rel
        kwargs.update(dict.fromkeys(self.tsets_names,None))
        super().__init__(**kwargs)
        for curr in self.tsets_names:

            # FIXME: maybe make a Tset object so I can get rid of these underscores
            npa_fieldname = curr+"_npa"

            buf = kwargs.get(npa_fieldname,None)
            if buf is not None:
                self[curr] = self._deserialize(buf)
                if self.with_set_index is False:
                    self[curr] = self[curr][:,1:]
                sections = rel.divide(
                    self[curr],
                    with_set_index=self.with_set_index
                )
                self[curr+"_sections"] = sections # FIXME: needed to keep this?
                if curr == 'train':
                    self._make_and_fit_scalers(sections)

                self[curr+"_scaled"] = self._scale(sections)
            else:
                raise Exception(f"Didn't find {npa_fieldname}")

    def _deserialize(self, buf):
        return utils.deserialize(buf).astype(np.float32)

    def _make_and_fit_scalers(self,sections):
        for field, section in zip(self.rel.fields, sections):
            field.make_and_fit_scaler(section)

    def _scale(self,sections):
        scaled = []
        for field, section in zip(self.rel.fields, sections):
            section_scaled = field.scaler.transform(section)
            scaled.append(section_scaled)
        ret = self.rel.glue(scaled)
        return ret


