import functools
import sys
import os
from collections import defaultdict

from dataspecs.dataspecs import Spec

sys.path.append(
    os.path.abspath(
        os.path.dirname(
            os.path.abspath(__file__)
        )+"/.."
    )
)

from common import dataset_persistency

functools.lru_cache(maxsize=None)
def get_codelists():
    """
    returns a codelist from the mongo db
    :return:
    """
    with dataset_persistency.MongoDB() as db:
        ret = defaultdict(lambda: list())
        for curr in db['codelists'].find({}):
            ret[curr['name']] = curr['codelist']
        return ret


class Activity(Spec):
    """
    Specifies the fields of an activity.
    """
    @property
    def prefixed_fields_names(self):
        return self.fields_names

    @property
    def extract_field_regex(self):
        """
        regex to extract the field name within this rel.
        In the case of Activity non-relation fields,
        the field name corresponds to the IATI.cloud
        field name.
        :return:
        """
        return '(.*)'

    def extract_from_field_data(self, v):
        if type(v) in (list, tuple):
            # if it's a list, which is unlikely, then just
            # get the first element
            v = v[0]
        return v


