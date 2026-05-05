# Loading data

polar-high doesn't have its own data importer. Variables, parameters,
and constraint index frames are all **polars DataFrames**, so you
build them with whichever polars-supported source your data lives
in. There's no special format to learn and no separate import layer
to plumb.

## What polar-high needs

Three shapes:

| Where it goes | Frame columns | Example |
|---|---|---|
| `Var` index | `(*dims,)` plus any extras you carry | `pl.DataFrame({"i": [1, 2], "j": ["a", "b"]})` |
| `Param` | `(*dims, value)` | `pl.DataFrame({"i": [1, 2], "j": ["a", "b"], "value": [3.0, 1.0]})` |
| `add_cstr(over=…)` | columns naming the dims the constraint family is indexed over | `pl.DataFrame({"node": [...], "hour": [...]})` |

That's all the engine knows about. Anything that produces frames in
these shapes works.

## From CSV (long format)

The simplest case: a CSV with one row per cell, including a `value`
column.

```csv
i,j,value
1,a,3.0
1,b,1.0
2,a,2.0
```

```python
import polars as pl
from polar_high import Param

cost_df = pl.read_csv("cost.csv")
cost = Param(("i", "j"), cost_df)
```

## From CSV (wide format)

Spreadsheet-style CSV with rows indexed by `i` and columns indexed
by `j`:

```csv
i,a,b
1,3.0,1.0
2,2.0,
```

Reshape to long form before constructing the Param:

```python
df = pl.read_csv("cost_wide.csv")
cost_long = (
    df.unpivot(index="i", on=["a", "b"],
               variable_name="j", value_name="value")
      .drop_nulls("value")
)
cost = Param(("i", "j"), cost_long)
```

`unpivot` (formerly `melt`) flips wide to long; `drop_nulls("value")`
removes empty cells if your spreadsheet has gaps.

## From parquet

```python
demand_df = pl.read_parquet("demand.parquet")
demand = Param(("node", "hour"), demand_df.rename({"d": "value"}))
```

Use `rename` if your parquet's value column has a different name.

## From a database

```python
import polars as pl
import sqlite3

conn = sqlite3.connect("mydata.db")
edges_df = pl.read_database(
    "SELECT src, dst, capacity FROM edges", conn,
).rename({"capacity": "value"})
cap = Param(("src", "dst"), edges_df)
```

`pl.read_database` works with any DB-API connection. For complex
queries with joins and aggregations, write them in SQL and let
polars receive the result frame.

## From pandas / numpy / Excel

For sources polars doesn't read directly:

```python
import pandas as pd
import polars as pl

pd_df = pd.read_excel("mydata.xlsx", sheet_name="costs")
pl_df = pl.from_pandas(pd_df)
```

`pl.from_pandas` and `pl.from_numpy` cover the cases where another
library got the data first.

## Densification: when missing cells matter

`Param * Var`, `Param + Param`, and the joins inside `add_cstr` are
all **inner-joins** on shared dims. Rows missing in one frame are
silently dropped from the result. Two ways this matters:

1. **Sparse parameter, dense variable.** If `cost[i, j]` only has
   some `(i, j)` cells but you compute `cost * v` over every cell
   of `v`, the missing cells effectively contribute nothing. That's
   usually what you want.
2. **You explicitly need zero for missing cells.** If a missing
   cell should mean "coefficient zero" (not "this term doesn't
   apply"), densify before constructing the Param:

   ```python
   full_idx = v_idx.select("i", "j")
   cost_dense = (
       full_idx.join(cost_sparse, on=["i", "j"], how="left")
               .with_columns(value=pl.col("value").fill_null(0.0))
   )
   cost = Param(("i", "j"), cost_dense)
   ```

   Do this once at load time, not inside a constraint loop.

## SpineDB and other domain formats

polars doesn't read SpineDB natively. The pattern is the same as for
any other non-polars source:

1. Use the source's Python API to pull rows into pandas / dicts /
   lists.
2. Construct polars frames from there (`pl.from_pandas` or
   `pl.DataFrame`).
3. Reshape (rename `value`, unpivot if needed) and pass to
   polar-high.

For domain-specific loaders (energy systems, supply chains, ...) the
convention in this ecosystem is to keep the loader in the
*application* repo, not in polar-high itself. polar-high is
intentionally domain-free; the loader is yours.

## Recipes worth remembering

- **Rename a column to `value`.** Most polar-high data ends up as
  `(*dims, value)`. If your source uses a different column name
  (`cost`, `cap`, `demand`, ...), rename to `value` once at load.
- **`drop_nulls("value")`** when reading sparse spreadsheet data so
  empty cells don't silently become zeros.
- **Densify at load time, not in solve.** If the model semantics
  require explicit zeros, do the `left_join`/`fill_null(0)` step
  once, when constructing the Param.
- **One Param per parameter.** If your CSV packs multiple
  parameters into one file with a discriminator column
  (`param_name = "cost"`, `param_name = "cap"`, ...), filter and
  pivot once at load time so you end up with a separate Param
  per logical parameter.
