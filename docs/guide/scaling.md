# Scaling — `polar_high.autoscale`

LP coefficient ranges that span more than six decades make HiGHS' presolve unhappy. Most often it prints a hint like *"Consider setting the `user_bound_scale` option to -6"* and solves anyway. Only when a model is *badly* scaled — coefficients spanning eight or nine decades, with near-zero rows the presolve can mishandle — does it risk coming back falsely infeasible. The `polar_high.autoscale` package detects the four-range readout, recommends `user_bound_scale` and `user_objective_scale` exponents, and (optionally) writes them back onto the `Problem` for HiGHS to apply at solve time. It's worth reaching for when the automatic settings give you trouble — a scaling warning you can't shake, or a model that solves with presolve off but not on.

If your LP's coefficients all sit within a few decades of 1.0, you don't need any of this — HiGHS' own `simplex_scale_strategy` handles it and polar-high changes nothing.

## The scaling layers

polar-high thinks about scaling in three layers, which compose rather than compete. HiGHS' own `simplex_scale_strategy` (matrix equilibration, Curtis–Reid) sits underneath all of them and is always available; the `autoscale` package layers on top of it:

- **Layer 1 — detection.** Read the four coefficient ranges (matrix `A`, cost `c`, variable/row bounds, RHS) off the built LP and decide whether the spread warrants attention. No model changes. This is `detect_ranges`.
- **Layer 2 — semantic rewrites.** Per-quantity column/row factors applied to the matrix *before* it reaches HiGHS — e.g. expressing power in GW instead of MW so a whole group of coefficients shifts by a known factor. This is domain knowledge the modeller has and the solver doesn't, so it lives in the orchestration layer (FlexTool implements it); the library proper does not, and `BASIC` mode skips it. When present, Layer 2 runs first and Layers 1/3 see the already-rewritten matrix.
- **Layer 3 — HiGHS-native global scaling.** Recommend power-of-two `user_bound_scale` and `user_objective_scale` exponents that pull the worst bound/cost magnitudes into HiGHS' comfort zone, plus `simplex_scale_strategy` for the residual matrix spread. HiGHS applies these internally and unscales on output, so your `Solution` values come back in the original units. This is `recommend_scaling` + `apply_scaling`.

The three are independent dials: you can run detection alone, hand-tune the Layer 2 units, let Layer 3 recommend the solver exponents, or any combination. The rest of this guide focuses on Layers 1 and 3, the part the library owns.

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

The pre-solve readout (and `ranges_from_arrays`) also fills optional fields for the scaling check. They default to `None` (e.g. from `streamed_lp_ranges`) and are not part of equality:

| field | meaning |
|---|---|
| `bound_rhs_meaningful` | `(b_lo*, b_hi)` over the *pool*: the finite non-zero bounds of the continuous columns and the row bounds, below HiGHS' `infinite_bound` (`1e20`; larger is no limit to HiGHS). `b_lo*` leaves out *residue* — entries whose model-unit magnitude is below `ScalingConfig.residue_floor` (default 10 × the primal feasibility tolerance), such as `2e-15` from a cancellation |
| `bound_rhs_residue` | `(count, largest model-unit magnitude)` left out by the floor |
| `bound_rhs_beyond_infinite` | number of values at or above `infinite_bound` |
| `integer_bound`, `integer_matrix`, `integer_cost` | the integer columns' bounds, matrix terms and costs (HiGHS' `user_bound_scale` leaves integer bounds alone and multiplies their matrix terms and costs instead) |
| `matrix_continuous`, `cost_continuous` | the same for the continuous columns |
| `residue_floor`, `infinite_bound` | the floor and the cut used |
| `extremes` | `extremes[group][side]` → `Extreme(family, post, model, component)`: the variable / constraint family holding each end |

Model units: a row bound before the Layer 2 row factor; a column bound times the inverse column factor (`_layer2_col_factor`).

## Recommendation — `recommend_scaling` and `Layer3Plan`

```python
plan = recommend_scaling(ranges, config, problem=p)
```

The `problem=` kwarg is optional but matters: when passed, the recommender consults `Problem.get_solver_option(...)` for `user_bound_scale` and `user_objective_scale`, and **skips its own recommendation on that axis** if the caller already set the value explicitly. This is the precedence-respect rule — your `set_solver_option` always wins.

Per-axis precedence (independent for objective and bound):

1. **Caller-set on the Problem** — `Problem.get_solver_option("user_bound_scale")` returns non-`None`. The recommender preserves it. `Layer3Plan.bound_skipped_external = True`.
2. **Caller-set on the config** — `config.user_bound_scale is not None`. The recommender uses your value verbatim. Reasoning string mentions `"manual override"`.
3. **Default auto** — power-of-two exponent that pulls `max(|b|)` into HiGHS' `[1e-4, 1e+6]` comfort zone. Outer-rounded `log2`. Same conservative rule HiGHS' own `suggestScaling` uses. On a severe overshoot (`max_b >= 1e9`) whose clamp would crush the small end, the clamp is kept ("smallest-side", see *Why* below).
4. **Tolerance window** — the automatic bound exponent is then clipped into `bound_scale_window(b_lo*, b_hi, tol)`: the exponents that keep the scaled pool inside `[10·tol, 1e18·tol]` (`[1e-6, 1e11]` at HiGHS' default `tol = 1e-7`, 17 decades at any tolerance), with the small end at `100·tol` or more whenever the large end still fits. `tol` is the explicit `primal_feasibility_tolerance` on the Problem (`kkt_tolerance` replaces it when set to anything but `1e-7`, as in HiGHS), or the `primal_feasibility_tolerance=` keyword. No exponent fits → the tier exponent is kept and tagged `unscalable`. Manual and external exponents are not clipped.

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
    bound_source: str                   # "auto" / "manual" / "external"
    bound_tag: str                      # tier tag, "+raised"/"+lowered" by the window, or "unscalable"
    bound_window: ScaleWindow | None    # the window the automatic exponent was clipped into
    primal_feasibility_tolerance: float # the tolerance the window used
```

All exponents are clamped to HiGHS' option-range `[-30, 30]`.

## Assess — `assess_scaling`

```python
verdict = assess_scaling(ranges, plan, primal_tol=tol, dual_tol=tol_d, warn_decades=13.0)
```

A pure judgement of the LP HiGHS will receive (`ranges`, before `user_*_scale`) under `plan`. It returns a `ScalingVerdict` (`level` `ok` / `warn` / `error`, and `findings`); what a level means — raise, warn, carry on — is the caller's decision.

| group | warn | error |
|---|---|---|
| `bound_rhs` | spread above `warn_decades` | the scaled pool leaves `[10·tol, 1e18·tol]` (kind `spread` / `scale_option_limit` when no exponent fits, or the manual / external / unscaled exponent is outside the window) |
| `cost` | spread above `warn_decades`, or the smallest scaled cost below `10·dual_tol` | — |
| `matrix` | a scaled term at or below `small_matrix` (`1e-9`: HiGHS drops it) | one at or above `large_matrix` (`1e15`: HiGHS refuses the model) |
| `info` | residue left out, values beyond `infinite_bound`, integer bounds | — |

Integer columns are scaled as HiGHS scales them (matrix terms `× 2**N_bnd`, costs `× 2**(N_obj+N_bnd)`). `apply_user_scale=False` judges the LP at `N = 0` (a solver that gets no `user_*_scale`). `ranges_pre` (the readout before a Layer 2 rewrite) sets `Finding.caused_by_prescale` when only the rewritten LP fails. Each finding carries `numbers` (spread, scaled ends, `r_lo` = the fraction of the smallest limit HiGHS may violate, its model-unit amount, the window, and what fitting the largest value would cost the smallest limit) and the `extremes` holding each end.

`locate_extremes(p, family, group, side, k=5)` then names the entries: it walks that one family with the bounded walk and returns the `k` smallest or largest entries with their dimension values (Enum codes decoded) and the number of entries within a factor of the best one. Meant for the warning / error path.

## Checking each solve — `pre_run_check`

`Problem.solve(pre_run_check=fn)` calls `fn(ranges, highs)` once, right before `HiGHS.run()`, with `streamed_lp_ranges` plus `col_bound_meaningful` / `row_bound_meaningful` (the pool of the LP actually built, with the residue floor applied), `bound_rhs_residue`, `bound_rhs_beyond_infinite`, `residue_floor` and `infinite_bound`. It may set options on `highs` (e.g. `user_bound_scale`), return a dict of options to set, or raise (no solve runs). Streaming path only; `build_only` and `WarmProblem` solves do not call it.

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

## Tiny cost coefficients — `obj_coef_zero_threshold` / `obj_coef_zero_rel_threshold`

The objective is centred with a single exponent: `user_objective_scale` places the geometric centre of the cost band `[min|c|, max|c|]` at 1.0. One spurious near-zero cost — a float residue such as `1e-17` from an algebraic cancellation like `1/eta - slope` — stretches that band to 17 decades and pulls the centre with it (`2**25` and more), and HiGHS' dual simplex can then fail on the inflated costs ("excessive dual values"). The matrix/RHS floor `Problem.coef_zero_threshold` does not touch the cost vector, so the objective has its own cutoff:

```python
p.obj_coef_zero_rel_threshold = 1e-14   # relative to the largest |cost|
p.obj_coef_zero_threshold = 0.0         # absolute (user units); 0 = off
```

A column's summed objective coefficient whose magnitude is strictly below `max(obj_coef_zero_threshold, 2**floor(log2(obj_coef_zero_rel_threshold * max|c|)))` becomes exactly `0.0` (`max|c|` is the largest `|coefficient|` of any objective term; the relative part is rounded down to a power of two). Both default to `0.0` — off, byte-identical to earlier releases.

- The cut is judged on the **user-space** coefficient, before the Layer 2 side vectors and the Layer 3 `user_objective_scale` (both are power-of-two conditioning transforms), so the decision does not depend on how the LP happens to be scaled.
- It applies everywhere the cost vector is produced: the canonical matrix (`write_mps`, `passModel`, `WarmProblem`), the streaming solve and its `streamed_lp_ranges`, the pre-solve `detect_ranges` readout — so `recommend_scaling` centres on the costs the solver will actually see — and `WarmProblem.update_obj_coef` / `update_obj_coef_array` (with the cutoff resolved at build).
- Prefer the **relative** form: cost magnitudes legitimately span many decades (a penalty of `1e6` next to an O&M cost of `0.01`), and a model's whole objective may be scaled by a constant, so an absolute cutoff that is safe for one model can delete real costs in another. A relative `1e-14` sits about 100 machine epsilons below the largest cost — far below any real cost spread, above cancellation residue.

## Why the min-floor guard and the tolerance window exist

HiGHS' built-in recommendation looks at `max(bound_max, rhs_max)` and picks a delta that pulls that single value into `[1e-4, 1e+6]`. On wide-spread LPs the consequence can be brutal:

- **Example 1 model from FlexTool** has `RHS = (1.84e-3, 2.02e+8)`. HiGHS' formula picks `N=-8`, scaled RHS min becomes `7.2e-6`, which is below `_HIGHS_SMALL_BOUND` (`1e-4`). Presolve then mis-handles those near-zero rows and the LP comes back **infeasible** — even though solving with `N=0` and presolve off finds the optimum cleanly. **Min-floor guard**: when the proposed `N` would drag the scaled min below `1e-4`, `recommend_scaling` refuses to recommend that `N` and returns the current scale unchanged (reasoning tag `"refuse"`).

- **Example 2 model from FlexTool** has bounds spanning so many decades that *every* power-of-two clamp crushes one end. For those — when `max_b >= 1e+9` (a "severe overshoot") — refusing is worse, so the clamp is kept (reasoning tag `"smallest-side"`) and the **tolerance window** decides where the range lands: HiGHS checks `primal_feasibility_tolerance` as an absolute amount on the *scaled* bounds, so the smallest meaningful limit must stay well above it. The window lifts the exponent until the smallest limit sits at `100·tol` (or at least `10·tol`) and gives the excess to the large end, up to `1e18·tol`. (Up to 3.14 this branch centred the range geometrically on the comfort band, tag `"escape"`; with a tolerance of `1e-7` that could leave the smallest limit below the tolerance itself.)

Both behaviours are baked into `_recommend_bound_scale` + the window. The `Layer3Plan.reasoning` string (and `bound_tag`) surfaces which branch fired:

- `"in-zone"` — already in `[1e-4, 1e+6]`; recommended `N=0`.
- `"clamp-large"` — pulled `max_b` down into the comfort zone.
- `"clamp-small"` — pulled `max_b` up into the comfort zone.
- `"smallest-side"` — severe overshoot; the window lifts the small end.
- `"refuse"` — min-floor guard tripped; recommendation skipped.
- `"+raised"` / `"+lowered"` suffix — the tolerance window moved the exponent.
- `"unscalable"` — no exponent fits the window; `assess_scaling` reports an error.

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

If you don't want the tolerance window or the min-floor guard — e.g. you want to reproduce a 1.4.x-shaped recommendation exactly — pass a manual `user_bound_scale=N` on the `ScalingConfig` and the recommender will use your integer verbatim.

## When *not* to use autoscale

- **Coefficients are within a few decades of 1.0.** HiGHS' own `simplex_scale_strategy` handles this fine; the autoscaler would correctly recommend `N=0` everywhere and add bookkeeping noise.
- **You're tuning HiGHS by hand.** Set `user_bound_scale` / `user_objective_scale` via `Problem.set_solver_option(...)` directly. The autoscaler's precedence check will skip your axes when you ask it for a recommendation.
- **You need scaling to survive a temp-MPS roundtrip.** `save_memory=True` (see *Performance*) writes the LP via `writeModel` after scaling options are applied — HiGHS preserves the option settings across the roundtrip, so this just works, but be aware that the scaling exponents have already been baked into HiGHS' internal LP by then. The same applies to [`Problem.write_mps`](performance.md#writing-mps-without-highs), which emits MPS directly from the polars frames after any Layer-2 rewrites have been applied.

## Reference

| symbol | purpose |
|---|---|
| `ScalingMode` | `OFF` / `SOLVER_ONLY` / `BASIC` / `FULL` |
| `ScalingConfig` | per-run dataclass: mode, threshold, manual overrides |
| `mode_enables_layer1(mode)` | predicate: does default policy run detection for this mode? |
| `mode_enables_layer3(mode)` | predicate: does default policy run recommendation? |
| `detect_ranges(p_or_sol, config)` | Layer 1 — returns a `RangeReport` |
| `RangeReport` | four `(abs_min, abs_max)` tuples + cross-group ratio + trigger, plus the scaling-check fields |
| `recommend_scaling(ranges, config, *, problem=None, primal_feasibility_tolerance=None)` | Layer 3 — returns a `Layer3Plan` |
| `Layer3Plan` | the three HiGHS exponents + reasoning + skipped flags + source / tag / window |
| `bound_scale_window(b_lo, b_hi, primal_tol)` | the `user_bound_scale` exponents that fit the tolerance window (`ScaleWindow`) |
| `assess_scaling(ranges, plan, *, primal_tol, dual_tol, ...)` | judge a plan — returns a `ScalingVerdict` of `Finding`s |
| `locate_extremes(p, family, group, side, *, k=5)` | the `k` smallest / largest entries of one family, with their dimension values |
| `resolve_feasibility_tolerances(p)` / `resolve_infinite_bound(p)` | the tolerances / infinite bound set on `p`, as HiGHS applies them |
| `apply_scaling(p, plan)` | merge the plan into `p._solver_options` |
| `has_explicit_option(p, name)` | precedence helper — is this option already caller-set? |
| `get_explicit_option(p, name)` | precedence helper — what value does the caller have set? |
| `USER_SCALE_CLAMP_LO` / `_HI` | `±30`, HiGHS' option-range bounds |
