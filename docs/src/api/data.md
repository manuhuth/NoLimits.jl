# Data Binding API

Binding a model to a `DataFrame`, and reading the resulting `DataModel` back.

!!! note "Some entries need an optional dependency"
    Parts of this API are implemented in package extensions and become available only once the
    corresponding package is loaded. Calling one without its package raises an error naming what
    to install. See [Optional Dependencies](../installation.md#Optional-Dependencies).

## DataModel

```@docs
DataModel
```

## DataModel Accessors

```@docs
get_individuals
get_individual
get_batches
get_batch_ids
get_primary_id
get_df
get_model
get_row_groups
get_re_group_info
get_re_indices
get_closed_form_plan
```

## Summaries

```@docs
ModelSummary
DataModelSummary
DescriptiveStats
summarize
```

## Where to go next

- [Data Model Construction guide](../data-model-construction.md) - the prose behind these names.
- [Estimation](estimation.md) - the next API page.
- [API overview](../api.md) - all seven reference pages.
