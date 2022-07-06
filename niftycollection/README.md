Nifty Collection
================

Extends the `dict` type by enabling item access as they were instance variables.

Added items may have a `name` attribute that is going to be used as key.

Iteration on this object iterates over the values of the dictionary.

Examples
--------

```
>>> from niftycollection.collection import Collection
>>> foo = Collection(quuz='baz')
>>> print(foo.quuz)
baz
>>> print(foo.names)
dict_keys(['quuz'])
```

