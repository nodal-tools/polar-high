# Debugging models

## Inspect before solve

```python
p.cstr_names()                    # list every constraint family
p.cstr_row_count("balance")       # how many LP rows did "balance" produce?
p.cstrs_named("balance")[0]       # the CstrRecord (over_index, term protos, …)
```

Row count surprises are the most common bug source: an `over=` frame
that's bigger or smaller than you expected, or a `Where(...)` that
silently filtered out half the rows.

## Inspect after solve

```python
sol.highs.writeModel("model.mps")     # or .lp — full LP for offline diff
sol.highs.getModelStatus()
sol.highs.getInfo()
sol.constraint_dual("balance")        # dual frame, indexed by over=
```

`writeModel("model.lp")` produces human-readable LP format and is
usually the fastest route to "what is the kernel actually sending to
HiGHS" — diff against a reference build of the same model.

## Term isolation

When two builds of the same model disagree on `obj`, check the
constraint families one at a time:

1. Same `cstr_row_count` per family?
2. Same `constraint_dual` shape per family?
3. Same nonzero pattern in `model.lp`?

Because the labelled-form `add_cstr(... lhs_terms={...}, rhs_terms={...})`
preserves term names through to the LP, you can also write a
diagnostic comparator that builds two reference Exprs and compares
their resulting term frames — much sharper than "obj is off by 3.2%".

## Numerical issues

- **Infeasibility** — `sol.optimal == False` and
  `sol.highs.getModelStatus()` reports `kInfeasible`. Use HiGHS's
  irreducible-infeasible-subset support (`h.getIis(...)`) via the
  live handle if available in your `highspy` build.
- **Unboundedness** — usually a missing bound. Check that every
  `add_var` call set a finite upper bound where appropriate, and that
  no constraint family was accidentally skipped.
- **Degeneracy / cycling** — rare with default HiGHS, more common
  with hand-tuned options. Try `solver=simplex` vs `ipm` to triangulate.

## Warm-problem invariants

When using [`WarmProblem`](warm-starting.md), the first wrong-result
debugging step is always: *does a fresh `Problem.solve()` from the
same final state agree with the warm result?* If yes, the kernel is
fine and your update calls are off; if no, there's a kernel bug to
report.
