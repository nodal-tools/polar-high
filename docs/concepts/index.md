# Concepts

polar-high's mental model is straightforward: **everything is a
polars DataFrame**. Variables, parameters, expressions, constraints —
all carry their index sets explicitly as frame columns, and the
operations on them (`*`, `+`, `Sum`, `Where`, `Lag`) are just
joins and group-bys under the hood.

If you have used Pyomo or JuMP, the data structures will feel
familiar; the semantics will not. Read this section before the API
reference. It will make better sense once the dim/join
model clicks.

## Reading order

1. [Vars and Params](vars-and-params.md) — what an indexed variable
   actually is, what `dims` mean, how parameters merge.
2. [Expressions](expressions.md) — `Sum`, `Where`, `Lag`,
   broadcasting / inner-join rules and what they compile to.
3. [Problem and Solve](problem-and-solve.md) — `Problem.add_cstr`,
   row counts, how `over=` materializes rows, the `Solution` shape.
4. [Duals and bases](duals-and-bases.md) — accessing dual values,
   reduced costs, the live `highspy.Highs` instance.

## One-liner glossary

| Term | What it is |
|---|---|
| **dim** | A named index axis (e.g. `"i"`, `"t"`). Always a column name in the underlying frame. |
| **Var** | A frame `(*dims, col_id)`: one LP column per row. |
| **Param** | A frame `(*dims, value)`: one numeric value per cell. |
| **Expr** | A list of `_Term`s, each a frame `(*dims, col_id, coef)`. |
| **`over=`** | The index frame that defines the *rows* of a constraint family. |
| **`Sum(..., over=)`** | Group-by-sum: the listed dims disappear, the rest become open. |
| **`Where(..., frame)`** | Inner-join an Expr with a frame: filters rows and can introduce new open dims. |
| **`Lag(var, frame, time, lag)`** | Reference `var` at a shifted time index, joined via `frame`. |
