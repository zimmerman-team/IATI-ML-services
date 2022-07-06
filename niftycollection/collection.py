import copy


class Collection(dict):
    """
    In a utils.Collection, which inherits from `dict`,dynamic properties
    can be set as `c['the_property'] = the_value`
    and accessed as in `c.the_property`.
    Moreover the `.names` property returns the properties that
    have been set.
    """

    def __init__(self, *args, **kwargs):
        if len(args) > 0:
            lst = args[0]
            assert type(lst) is list, \
                "first argument of the collection has to be a list of elements"
            for curr in lst:
                if curr.name in self.names:
                    raise Exception(f"element {curr.name} already in this Collection")
                self[curr.name] = curr
        super().__init__(**kwargs)

    def __iter__(self):
        """
        Iterating on the utils.Collection results in the values
        of the properties set, not the names of the properties.
        :return: iterator on the values of the properties that have been set
        """
        return iter(self.values())

    @property
    def names(self):
        """
        :return: list of string names of the properties set
        """
        return self.keys()

    def __getattr__(self, name):
        """
        in a utils.Collection, dynamic properties can be set as
        `c['the_property'] = the_value`
        and accessed as in `c.the_property`
        :param name: the name of the property being accessed
        :return: the value of the property
        """
        if name[0] == '_':
            return self.__getattribute__(name)
        assert name in self.names, f"{name} not in collection's names"
        return self[name]

    def __add__(self, addendum):

        # new object is going to be returned,
        # without altering the source one
        new = copy.deepcopy(self)
        for curr in addendum:
            new.add(curr)
        return new

    def add(self, item):
        self[item.name] = item
