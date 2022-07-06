DataSpecs
=========

Data mapping module for easy specification definition of data that needs to be
downloaded (especially from a Solr endpoint), pre-processed and converted into 
a machine learning-friendly format.

Every field type has its own class, and specifies machine learning aspecs
such as activation functions, normalizations, feature dimensionality counting,
feature intervals in input dimension space, creating the glued-up arrays and
splitting them up.