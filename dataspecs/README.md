DataSpecs
=========

Data mapping module for easy specification definition of data that needs to be
downloaded (especially from a Solr endpoint), pre-processed and converted into 
a machine learning-friendly format.

Every field type has its own class, and specifies machine learning aspecs
such as activation functions, normalizations, feature dimensionality counting,
feature intervals in input dimension space, creating the glued-up arrays and
splitting them up.

Example usage:

```python
rels = SpecsCollection([
    Rel("activity_date", [
        CategoryField("type","ActivityDateType"),
        DatetimeField("iso_date")
    ], download=True),
    Rel("budget", [
        CategoryField(
            "value_currency",
            'Currency',
            prevent_constant_prediction='USD'
        ),
        CategoryField("type", 'BudgetType'),
        CategoryField("status", 'BudgetStatus'),
        DatetimeField("period_start_iso_date"),
        DatetimeField("period_end_iso_date"),
        NumericalField("value")
    ], download=True),
    Rel("result", [
        CategoryField("type", 'ResultType'),
        TextField("title_narrative"),
    ])
])
```