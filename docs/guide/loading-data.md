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

### How column names become dim names

A note on the bookkeeping. polar-high uses **polars's column names
as the dim names** — the engine doesn't track dims separately from
the frame's schema. When you write `Param(("i", "j"), cost_long)`,
the engine verifies that `cost_long` has columns named `i`, `j`,
and `value`, and stores the LazyFrame. Whenever `cost` is later
multiplied by a `Var` that also declares `("i", "j")`, polars joins
the two frames on shared column names and the result keeps those
names. Dim tracking is bookkeeping polars does for you.

The wide-format example above leans on this twice in the unpivot
call:

```python
cost_long = df.unpivot(
    index="i",
    on=["a", "b"],          # values that go into the new dim column
    variable_name="j",      # name of the new dim column
    value_name="value",     # name of the coefficient column
)
```

`variable_name="j"` is what links the wide CSV's column headers to
polar-high's `Param(("i", "j"), …)` declaration. If you wrote
`variable_name="product"` instead, the Param would carry the dim
`"product"` and the engine would only join it against `Var`s and
constraints that declare a `"product"` dim. The string is the
identity. Same for `value_name="value"` — polar-high looks for a
column literally called `value` on every `Param`, so if your source
calls it something else, rename it at load time:

```python
df = pl.read_parquet("cap.parquet").rename({"capacity": "value"})
cap = Param(("e", "t"), df)
```

That's the whole bookkeeping protocol. There's no separate type
system, no parameter registry, no name remapping at solve time —
the polars schema is the source of truth from data load through to
the COO triples that go into HiGHS.

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

## Memory: long format vs wide grids

Every polar-high `Param` frame is `(*dims, value)` long format. For
a regular grid with N nodes × T hours that's **N · T rows each
carrying both index columns plus the value**. xarray's dense layout
would store the dim coordinates *once* and a bare `(N × T)` value
array. For a 1000-node × 8760-hour grid with `int64` indices:

| storage | columns | bytes |
|---|---|---:|
| polar-high (long) | node + hour + value, all `int64`/`float64`, 8.76 M rows | ≈ 210 MB |
| xarray (wide) | 1000-long node coord + 8760-long hour coord + 1000×8760 value | ≈ 70 MB |

Three-ish times. polars does **not** auto-deduplicate the repeated
index values; what you load is what's in memory.

This overhead applies to *most* parameters in real models. Even
when entity-side topology is irregular (different processes connect
to different nodes), the **time dim is regular and dense**, and it's
usually the longest dim by a large margin. A typical hourly model
runs on 8760 time steps; entities number in the hundreds or low
thousands. So the regular-grid case isn't a corner case — it's the
common case for time-coupled parameters.

polar-high pays this overhead in exchange for two things, which you
should weigh against your own constraints:

1. **Faster build** at the matrix-assembly stage. polars's joins
   and group-bys outpace xarray's broadcast on the operations
   polar-high hits during `Problem.solve()`. The benchmark page
   shows the time numbers; for irregular topology the spread
   widens further.
2. **Simpler join semantics** for irregular relations
   (edge → node, process → commodity, period → block). These
   become explicit polars joins, which is cleaner than the
   masking-and-reindexing required to express the same thing in
   xarray. linopy can do it; the model code just gets harder.

If your model is *dominated* by time-coupled parameters on a
regular grid and memory is your tightest constraint, linopy/xarray
will likely use less RAM than polar-high. The benchmark page
confirms this on the dense `N × N` case — at full HiGHS solve,
xarray's compactness wins. Pick the layout that matches what you
care about most.

You can recover part of the constant factor with polars'
`Categorical` or `Enum` dtype on string-valued dim columns. Each
unique string is stored once and rows carry a 4-byte index instead
of a variable-length string:

```python
demand_df = pl.read_csv("demand.csv").with_columns(
    node=pl.col("node").cast(pl.Categorical),
)
demand = Param(("node", "hour"), demand_df)
```

polar-high doesn't care that `node` is categorical — joins and
group-bys behave identically. For numeric dim columns there's no
equivalent shortcut; you pay 8 bytes per row.

### Enum dtype alignment

`pl.Enum` is the stricter cousin of `pl.Categorical`: the categorical
vocabulary is fixed at construction. That's great for catching typos
but it interacts awkwardly with the typical LP-DSL workflow where
**different `Param` frames live on different subsets of an axis**.

For example, imagine a `capacity` Param defined only on the units
that *have* a finite capacity, and a `cost` Param defined on every
unit:

```text
capacity_df                       cost_df
  unit     value                    unit     value
  ─────    ─────                    ─────    ─────
  wind      120.0                   wind       5.0
  solar      60.0                   solar      3.0
                                    hydro      1.0
                                    coal      10.0

Enum vocab: {wind, solar}     Enum vocab: {wind, solar, hydro, coal}
```

```python
capacity = Param(("unit",), capacity_df)
cost     = Param(("unit",), cost_df)
```

Both columns are named `unit` and both hold strings, but their
`pl.Enum` vocabularies differ. polars 1.40 refuses to join two
`Enum` columns whose vocab strings differ — *even when one is a
strict subset of the other*. By default the first time
`capacity * cost` or any other expression brings these two frames
into a join you'd get a `SchemaError`.

polar-high aligns Enum-typed join keys at every internal `.join()`
site, transparently:

| left | right | what happens |
|---|---|---|
| same Enum | same Enum | nothing to do |
| Enum A | Enum B, where `cats(B) ⊆ cats(A)` | B is up-cast to A (the wider vocab; `strict=False`, values outside A become null and are dropped by the inner/left join) |
| Enum A | Enum B, where `cats(A) ⊆ cats(B)` | symmetric — A is up-cast to B |
| Enum | `Utf8` / `String` | the string side is cast to the Enum dtype |
| Enum A | Enum B, **disjoint vocabs** | `ValueError` with actionable guidance (cast to `Utf8` or build a union Enum) |

No try/except shims and no Utf8 fallbacks for the disjoint case — if
the two vocabs genuinely don't overlap, the model is asking for
something the LP can't represent and the error names the issue.

You don't have to do anything to opt in; it's the default behaviour
on every internal join. The contract is exercised by
`tests/test_enum_dtype_align.py`.

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
