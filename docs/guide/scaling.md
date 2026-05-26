# Scaling — `polar_high.autoscale`

LP coefficient ranges that span more than six decades make HiGHS' presolve unhappy. Sometimes it prints a hint like *"Consider setting the `user_bound_scale` option to -6"* and solves anyway; other times the presolve mishandles near-zero rows and the LP comes back falsely infeasible. The `polar_high.autoscale` package detects the four-range readout, recommends `user_bound_scale` and `user_objective_scale` exponents, and (optionally) writes them back onto the `Problem` for HiGHS to apply at solve time.

This guide is for callers who:

- have an LP that spans many orders of magnitude in matrix / cost / bound / RHS values,
- see HiGHS warnings about "excessively small" or "excessively large" bound values, or
- have hit a falsely-infeasible result and suspect HiGHS' built-in scaling is the culprit.

If your LP's coefficients all sit within a few decades of 1.0, you don't need this — HiGHS' own `simplex_scale_strategy` will handle it - and polar-high will not change anything.

## Quick example

```python
from polar_high import Problem
from polar_high.autoscale import (
    ScalingConfig,
    ScalingMode,
    apply_scaling,
    detect_ranges,
    recommend_scaling,
)

p = Problem()
# … add vars, params, objective, constraints …

config = ScalingConfig(mode=ScalingMode.BASIC)

ranges = detect_ranges(p, config)          # Layer 1: read the four ranges
plan = recommend_scaling(ranges, config)   # Layer 3: pick the exponents
apply_scaling(p, plan)                     # write them onto p._solver_options

sol = p.solve()
print(plan.reasoning)
# e.g. "auto (N_obj=0, N_bnd=-4, bound_tag=clamp-large)"
```

`apply_scaling` is a one-line convenience around `Problem.set_solver_options` — equivalent to calling `p.set_solver_option("user_bound_scale", plan.user_bound_scale)` etc., but with the right merge semantics so it doesn't clobber other options the caller already set.

After `solve()`, the same `detect_ranges` works on the returned `Solution` (it reads `Solution.streamed_lp_ranges`, populated by every streaming solve at no extra allocation):

```python
sol = p.solve()
ranges_observed = detect_ranges(sol, config)
```

## `ScalingMode` and `ScalingConfig`

`ScalingMode` is an enum of four named buckets:

| mode | what the default policy enables |
|---|---|
| `OFF` | autoscaler fully disabled; no detection, no options applied |
| `SOLVER_ONLY` | leave any solver-native scaling (`simplex_scale_strategy`) to the caller, skip Layer 1 + Layer 3 |
| `BASIC` | Layer 1 detection + Layer 3 recommendation |
| `FULL` | same as `BASIC` on the polar-high side; callers can layer their own semantic scaling on top |

The library does **not** consult `mode` directly — it's there as a named bucket the orchestration layer (e.g. FlexTool) uses to decide whether to run `detect_ranges` / `recommend_scaling` at all. Two predicate helpers expose the default mapping if you want it:

```python
from polar_high.autoscale import mode_enables_layer1, mode_enables_layer3

if mode_enables_layer3(config.mode):
    plan = recommend_scaling(detect_ranges(p, config), config)
    apply_scaling(p, plan)
```

`ScalingConfig` carries the per-run knobs:

```python
@dataclass(frozen=True)
class ScalingConfig:
    mode: ScalingMode = ScalingMode.BASIC
    threshold_decades: float = 9.0
    user_bound_scale: int | None = None
    user_objective_scale: int | None = None
    report_yaml_path: Path | None = None
```

- `threshold_decades` controls the `trigger` boolean on the `RangeReport` — set to 9 by default, meaning a max/min ratio above $10^9$ triggers attention. Lower it (say, 6) if your model is range-sensitive.
- `user_bound_scale` and `user_objective_scale` are **manual overrides**. When set, the recommender uses your integer verbatim and skips its own computation for that axis.
- `report_yaml_path` is a transport for caller-side reporters — the library doesn't read or write it.

## Detection — `detect_ranges` and `RangeReport`

```python
ranges = detect_ranges(p_or_sol, config)
```

The dispatch is duck-typed:

- If the input carries `streamed_lp_ranges` (i.e. a `Solution`), Layer 1 reads it directly. Zero extra LP work.
- Otherwise the input must walk like a `Problem`: it must have `_vars`, `_cstrs`, and `_build_lp_arrays`. The function then assembles the LP arrays into the four ranges itself. Slower; only useful pre-solve.

The returned `RangeReport` is a frozen dataclass:

```python
@dataclass(frozen=True)
class RangeReport:
    matrix: tuple[float, float]              # (|min| nonzero, |max| nonzero) over A
    cost: tuple[float, float]                # same over c
    bound: tuple[float, float]               # same over finite col_lower/col_upper
    rhs: tuple[float, float]                 # same over finite row_lower/row_upper
    cross_group_max_ratio: float             # max(hi) / min(lo) across the four
    trigger: bool                            # any group hi/lo > 10**threshold_decades?
```

Groups with no finite nonzero entries (e.g. a bound-less LP) come back as `(nan, nan)` and are excluded from `cross_group_max_ratio`. A truly empty input gets `nan` ratio + `trigger=False`.

## Recommendation — `recommend_scaling` and `Layer3Plan`

```python
plan = recommend_scaling(ranges, config, problem=p)
```

The `problem=` kwarg is optional but matters: when passed, the recommender consults `Problem.get_solver_option(...)` for `user_bound_scale` and `user_objective_scale`, and **skips its own recommendation on that axis** if the caller already set the value explicitly. This is the precedence-respect rule — your `set_solver_option` always wins.

Per-axis precedence (independent for objective and bound):

1. **Caller-set on the Problem** — `Problem.get_solver_option("user_bound_scale")` returns non-`None`. The recommender preserves it. `Layer3Plan.bound_skipped_external = True`.
2. **Caller-set on the config** — `config.user_bound_scale is not None`. The recommender uses your value verbatim. Reasoning string mentions `"manual override"`.
3. **Severe-overshoot escape** — naive recommendation would crush the small end below HiGHS' `kExcessivelySmallBoundValue` (`1e-4`) AND `max_b >= 1e9`. The recommender pivots to a geometric-centring branch instead of clamping to the comfort zone (see *Why* below).
4. **Default auto** — power-of-two exponent that pulls `max(|b|)` into HiGHS' `[1e-4, 1e+6]` comfort zone. Outer-rounded `log2`. Same conservative rule HiGHS' own `suggestScaling` uses.

The resulting `Layer3Plan`:

```python
@dataclass(frozen=True)
class Layer3Plan:
    user_objective_scale: int           # exponent for user_objective_scale (cost × 2**N)
    user_bound_scale: int               # exponent for user_bound_scale (bounds + RHS × 2**N)
    simplex_scale_strategy: int         # HiGHS equilibration strategy (0..5), default 2
    reasoning: str                      # one-line audit string for logs
    objective_skipped_external: bool    # was the objective axis precedence-respected?
    bound_skipped_external: bool        # was the bound axis precedence-respected?
```

All exponents are clamped to HiGHS' option-range `[-30, 30]`.

## Apply — wiring back onto the Problem

```python
apply_scaling(p, plan)
```

This is a thin merge over `Problem.set_solver_options`:

- writes `user_objective_scale` only when the auto-recommender chose it (not when `objective_skipped_external` is True),
- writes `user_bound_scale` similarly,
- always writes `simplex_scale_strategy` (Layer 3 owns this value — callers who want a different strategy should call `p.set_solver_option("simplex_scale_strategy", N)` *after* `apply_scaling`).

If you'd rather not use `apply_scaling`, the two HiGHS options are plain integers you can pass through any of the usual routes:

```python
# equivalent — and useful when you want to record the plan without
# applying it (e.g. dry-run mode):
p.set_solver_option("user_bound_scale", plan.user_bound_scale)
p.set_solver_option("user_objective_scale", plan.user_objective_scale)
p.set_solver_option("simplex_scale_strategy", plan.simplex_scale_strategy)
```

## Why the escape branch and the min-floor guard exist

HiGHS' built-in recommendation looks at `max(bound_max, rhs_max)` and picks a delta that pulls that single value into `[1e-4, 1e+6]`. On wide-spread LPs the consequence can be brutal:

- **Example 1 model from FlexTool** has `RHS = (1.84e-3, 2.02e+8)`. HiGHS' formula picks `N=-8`, scaled RHS min becomes `7.2e-6`, which is below `_HIGHS_SMALL_BOUND` (`1e-4`). Presolve then mis-handles those near-zero rows and the LP comes back **infeasible** — even though solving with `N=0` and presolve off finds the optimum cleanly. **Min-floor guard**: when the proposed `N` would drag the scaled min below `1e-4`, `recommend_scaling` refuses to recommend that `N` and returns the current scale unchanged (reasoning tag `"refuse"`).

- **Example 2 model from FlexTool** has bounds spanning so many decades that *every* power-of-two clamp crushes one end. For those — when `max_b >= 1e+9` (a "severe overshoot") — the recommender pivots to a **geometric-centring escape** (reasoning tag `"escape"`): pick `N` such that the geometric mean of the scaled range lands at the geometric mean of HiGHS' comfort band (`sqrt(1e-4 × 1e+6) ≈ 31.6`). Worse on max, better on min, optimal on neither — but lets the LP solve.

Both behaviours are baked into `_recommend_bound_scale`. The `Layer3Plan.reasoning` string surfaces which branch fired:

- `"in-zone"` — already in `[1e-4, 1e+6]`; recommended `N=0`.
- `"clamp-large"` — pulled `max_b` down into the comfort zone.
- `"clamp-small"` — pulled `max_b` up into the comfort zone.
- `"escape"` — geometric-centring branch fired (severe overshoot).
- `"refuse"` — min-floor guard tripped; recommendation skipped.

## Migration from `auto_user_bound_scale=True`

polar-high 1.4.x exposed a constructor flag, retired in 2.0.0:

```python
# 1.4.x — retired
p = Problem(auto_user_bound_scale=True)
# … build model …
sol = p.solve()  # heuristic ran inside the streaming solve, col-bound only

# 2.0.x — equivalent (Layer 1 + Layer 3, all four ranges)
from polar_high.autoscale import (
    ScalingConfig, ScalingMode,
    apply_scaling, detect_ranges, recommend_scaling,
)

p = Problem()
# … build model …
config = ScalingConfig(mode=ScalingMode.BASIC)
ranges = detect_ranges(p, config)
plan = recommend_scaling(ranges, config, problem=p)
apply_scaling(p, plan)
sol = p.solve()
```

The new path differs in three ways from the retired flag:

- **All four ranges, not just col bounds.** The retired heuristic considered only `col_bound_lo/hi`. The new path uses cost + bound + RHS — so the same model can now get a non-trivial `user_objective_scale` too.
- **Precedence-respect is explicit.** Pass `problem=p` to `recommend_scaling` to make it consult `Problem.get_solver_option` for caller-set overrides.
- **The plan is inspectable before it's applied.** `Layer3Plan.reasoning` and the `*_skipped_external` flags let you log what would happen, dry-run, or guard on the `trigger` boolean before calling `apply_scaling`.

If you don't want the geometric-centring escape branch or the min-floor guard — e.g. you want to reproduce a 1.4.x-shaped recommendation exactly — pass a manual `user_bound_scale=N` on the `ScalingConfig` and the recommender will use your integer verbatim.

## When *not* to use autoscale

- **Coefficients are within a few decades of 1.0.** HiGHS' own `simplex_scale_strategy` handles this fine; the autoscaler would correctly recommend `N=0` everywhere and add bookkeeping noise.
- **You're tuning HiGHS by hand.** Set `user_bound_scale` / `user_objective_scale` via `Problem.set_solver_option(...)` directly. The autoscaler's precedence check will skip your axes when you ask it for a recommendation.
- **You need scaling to survive a temp-MPS roundtrip.** `save_memory=True` (see *Performance*) writes the LP via `writeModel` after scaling options are applied — HiGHS preserves the option settings across the roundtrip, so this just works, but be aware that the scaling exponents have already been baked into HiGHS' internal LP by then.

## Reference

| symbol | purpose |
|---|---|
| `ScalingMode` | `OFF` / `SOLVER_ONLY` / `BASIC` / `FULL` |
| `ScalingConfig` | per-run dataclass: mode, threshold, manual overrides |
| `mode_enables_layer1(mode)` | predicate: does default policy run detection for this mode? |
| `mode_enables_layer3(mode)` | predicate: does default policy run recommendation? |
| `detect_ranges(p_or_sol, config)` | Layer 1 — returns a `RangeReport` |
| `RangeReport` | four `(abs_min, abs_max)` tuples + cross-group ratio + trigger |
| `recommend_scaling(ranges, config, *, problem=None)` | Layer 3 — returns a `Layer3Plan` |
| `Layer3Plan` | the three HiGHS exponents + reasoning + skipped flags |
| `apply_scaling(p, plan)` | merge the plan into `p._solver_options` |
| `has_explicit_option(p, name)` | precedence helper — is this option already caller-set? |
| `get_explicit_option(p, name)` | precedence helper — what value does the caller have set? |
| `USER_SCALE_CLAMP_LO` / `_HI` | `±30`, HiGHS' option-range bounds |
