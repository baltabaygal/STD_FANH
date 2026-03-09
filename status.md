# Status

## Scope

Current analysis focuses on the direct `t_p` scan at fixed `H_* = 1` from:

- `ode/analysis/data/dm_tp_fitready_H1p000.txt`

Main derived quantity:

\[
Y_2(\theta_0,t_p) \equiv \frac{\xi(\theta_0,t_p)\,f_{\rm anh}^{\rm noPT}(\theta_0)}{t_p^{3/2}}.
\]

This is the quantity treated as the PT-modified anharmonic factor up to the fixed no-PT normalization.

## What Was Tried

### 1. Hilltop-shape fits at fixed `t_p`

Tested:

\[
Y_2(\theta_0,t_p) = A(t_p)\,S(\theta_0;\alpha(t_p))
\]

with three shape families:

- power: \((1-u^2)^{-\alpha}\)
- log-hilltop: \([\log(e/(1-u^2))]^\alpha\)
- pendulum-log: \([\log(e/\cos^2(\theta_0/2))]^\alpha\)

where \(u = \theta_0/\pi\).

Result:

- the log-hilltop family was the best of these three
- but the low-`\theta_0`, low-`t_p` region remained systematically mismatched
- a plateau fit for `A(t_p)` and `alpha(t_p)` was reasonable, but not the best global description

Conclusion:

- useful diagnostic
- not current best model

### 2. Effective-time model for `\xi`

Tested a global effective-time interpolation for `\xi`.

Result:

- fit quality was clearly worse than the direct `Y_2` descriptions

Conclusion:

- rejected as current working model

### 3. Option 1: multiplicative small-angle correction

Tested:

\[
Y_2 = A(t_p)\,[\log(e/(1-u^2))]^{\alpha(t_p)} (1 + c(t_p)u^2)
\]

Result:

- worsened the global fit
- did not cleanly fix the lowest-`\theta_0` residuals

Conclusion:

- rejected

### 4. Option 2: matched harmonic-to-hilltop prefactor

Tested:

\[
Y_2 = A(t_p)\,(1 + c(t_p)u^2)^\beta [\log(e/(1-u^2))]^{\alpha(t_p)}
\]

with a global `\beta`.

Result:

- only marginal improvement over the baseline hilltop fit
- not enough improvement to justify the extra structure

Conclusion:

- not adopted as current best model

### 5. Factorized multiplicative power-law correction

Tested lower-DOF forms like:

\[
Y_2 = c_0(\theta_0)\left(1 + \frac{r(\theta_0)}{t_p^{p}}\right)
\]

and a version with linear `p(\theta_0)`.

Result:

- constant-`p` version was too crude
- linear-`p` version improved, but remained worse than the additive surface-power model

Conclusion:

- not adopted

## Current Best Description

The best compact empirical model currently is:

\[
Y_2(\theta_0,t_p) = c_0(u) + \frac{c_1(u)}{t_p^{p(u)}},
\qquad
u=\theta_0/\pi.
\]

with

\[
c_0(u) = a_0 + a_1 u + a_2 u^2,
\]

\[
\log c_1(u) = b_0 + b_1 u + b_2 u^2,
\]

\[
p(u) = d_0 + d_1 u + d_2 u^2.
\]

Fitted coefficients:

- `c0(u) = 0.5439331054 - 0.0191090431 u + 0.1881281246 u^2`
- `c1(u) = exp(-1.1481913332 - 0.7155175800 u + 3.2879992014 u^2)`
- `p(u) = 1.7424922560 - 0.1146062325 u - 0.1990554148 u^2`

Global fit quality:

- log RMSE: `1.7393e-02`
- relative RMSE: `1.7498e-02`

This is better than the global plateau log-hilltop model and currently the best compact description in the repo.

## Interpretation

- strict separability in the form `f(\theta_0) g(t_p)` is not supported by the data
- the `t_p` decay itself changes with `\theta_0`
- the data prefer a plateau-plus-powerlaw-decay structure in `t_p`
- the remaining open question is whether `c_0(u)`, `c_1(u)`, and `p(u)` can be simplified further without losing fit quality

## Useful Files

Core scripts:

- `ode/analysis/analyze_xi_dm_tp_fitready.py`
- `ode/analysis/fit_y2_tp_powerlaw.py`
- `ode/analysis/fit_y2_surface_powerlaw.py`
- `ode/analysis/fit_y2_coeff_shapes.py`
- `ode/analysis/plot_y2_contour.py`

Core outputs:

- `ode/analysis/results/noPT_reference_dm_tp_fitready_H1p000.txt`
- `ode/analysis/results/fanh_noPT_vs_theta0_dm_tp_fitready_H1p000.png`
- `ode/analysis/results/fit_y2_tp_powerlaw_dm_tp_fitready_H1p000.txt`
- `ode/analysis/results/fit_y2_tp_powerlaw_dm_tp_fitready_H1p000.png`
- `ode/analysis/results/fit_y2_surface_powerlaw_dm_tp_fitready_H1p000.txt`
- `ode/analysis/results/fit_y2_surface_powerlaw_vs_tp_dm_tp_fitready_H1p000.png`
- `ode/analysis/results/fit_y2_surface_powerlaw_vs_theta_dm_tp_fitready_H1p000.png`
- `ode/analysis/results/y2_contour_dm_tp_fitready_H1p000.png`
- `ode/analysis/results/y2_log10_contour_dm_tp_fitready_H1p000.png`

## Next Reasonable Step

Try simplifying the current best model by reducing its degrees of freedom:

1. test `p(u)` as constant
2. test `p(u)` as linear in `u`
3. test whether `c_0(u)` or `\log c_1(u)` can be simplified without significantly degrading the fit
