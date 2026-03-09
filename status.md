# Status

## Scope

Current analysis focuses on the direct `t_p` scan at fixed `H_* = 1` from:

- `ode/analysis/data/dm_tp_fitready_H1p000.txt`

Main derived quantity:

\[
Y_2(\theta_0,t_p) \equiv \frac{\xi(\theta_0,t_p)\,f_{\rm anh}^{\rm noPT}(\theta_0)}{t_p^{3/2}}.
\]

This is the PT-modified anharmonic factor up to the fixed factor

\[
f_{\rm anh}^{\rm PT}(\theta_0,t_p) = (1.5)^{3/2} Y_2(\theta_0,t_p).
\]

## What Was Tried

### 1. Fixed-`t_p` angular-shape fits

The original slice-by-slice idea was

\[
Y_2(\theta_0,t_p) = A(t_p)\,S(\theta_0;\alpha(t_p)),
\qquad
u = \theta_0/\pi.
\]

The tested shape families were:

- power: \((1-u^2)^{-\alpha}\)
- log-hilltop: \([\log(e/(1-u^2))]^\alpha\)
- pendulum-log: \([\log(e/\cos^2(\theta_0/2))]^\alpha\)

Re-test on the current `Y2` table:

- raw power: global relative RMSE `4.7815e-02`
- power + `c0(t_p)`: `4.7815e-02` again, so the extra constant does not rescue the raw power family
- log-hilltop + `c0(t_p)`: `5.1788e-03`
- pendulum-log + `c0(t_p)`: `2.5100e-03`

Conclusion:

- the statement "`A(t_p)` and `alpha(t_p)` are not enough" is true for the old raw power family
- a fixed-`t_p` shape description can fit very well if we allow an additive plateau and use the pendulum-log hilltop variable
- this is a high-dimensional benchmark, not a compact final law

### 2. Stability of the fixed-`t_p` slice fits

For the two good `+ c0(t_p)` families, I ran leave-one-`\theta_0`-out checks slice by slice.

Predictive stability:

- `log_hilltop + c0(t_p)`: LOO holdout relative RMSE `1.4057e-02`, mean jackknife curve spread `3.0905e-03`
- `pendulum_log + c0(t_p)`: LOO holdout relative RMSE `1.0211e-02`, mean jackknife curve spread `2.2924e-03`

Parameter-curve smoothness:

- `log_hilltop + c0(t_p)` has a rougher `c0(t_p)` curve and more derivative sign changes
- `pendulum_log + c0(t_p)` is smoother in `c0(t_p)`, `A(t_p)`, and `alpha(t_p)`

Conclusion:

- the more stable fixed-`t_p` benchmark is

\[
Y_2(\theta_0,t_p) =
c_0(t_p) + A(t_p)\left[\log\!\left(\frac{e}{\cos^2(\theta_0/2)}\right)\right]^{\alpha(t_p)}
\]

### 3. Effective-time model for `\xi`

Tested a global effective-time interpolation for `\xi`.

Result:

- fit quality was clearly worse than the direct `Y_2` descriptions

Conclusion:

- rejected as current working model

### 4. Lower-DOF multiplicative corrections

Tested:

\[
Y_2 = A(t_p)\,[\log(e/(1-u^2))]^{\alpha(t_p)} (1 + c(t_p)u^2)
\]

and

\[
Y_2 = A(t_p)\,(1 + c(t_p)u^2)^\beta [\log(e/(1-u^2))]^{\alpha(t_p)}.
\]

Result:

- they did not beat the simpler additive slice benchmark
- they did not beat the final compact surface model either

Conclusion:

- not adopted

### 5. Older direct surface-powerlaw fit

An earlier empirical surface fit used direct low-order functions of `u = theta0/pi` in

\[
Y_2(\theta_0,t_p) = c_0(u) + \frac{c_1(u)}{t_p^{p(u)}}.
\]

It was useful as a stepping stone, but it was superseded by the current pendulum-log based compact model.

Conclusion:

- removed from the curated results set

### 6. Exponential-in-`t_p` transient benchmark

Tested:

\[
Y_2(\theta_0,t_p) = c_0(\theta_0) + c_1(\theta_0)e^{-t_p/\tau(\theta_0)}
\]

against the plateau-plus-powerlaw description.

Result:

- it was much worse than the powerlaw transient

Conclusion:

- removed from the curated results set

### 7. `eta` / `x_{\rm eff}` scaling and collapse tests

Tested whether the curves collapse better when organized by control variables such as
`eta ~ K(sin(theta0/2))/tp`, and whether normalized `Y2` curves share a common turnover.

Result:

- useful for interpretation
- not competitive as the final fit formula

Conclusion:

- removed from the curated results set

### 8. Legacy `f_anh` slice-fit variants

Before switching the focus fully to `Y2`, several direct `f_anh(theta0,tp)` slice fits were tested:

- raw power in `(1-u^2)^{-alpha}`
- `sin(theta0/2)` variant
- additive-constant version
- shifted and reparameterized singularity versions
- excess-over-plateau log-hilltop fits
- simple transition models

Result:

- these were useful diagnostics early on
- they are now superseded by the direct `Y2` analysis

Conclusion:

- removed from the curated results set

### 9. Symbolic-regression experiments

PySR runs were used to probe whether a cleaner symbolic formula could be recovered for
the coefficient laws and the full surface.

Result:

- informative as a sanity check
- not stable or clean enough to replace the current compact fit
- raw search checkpoints produced clutter in `outputs/`

Conclusion:

- raw symbolic-regression outputs and checkpoint directories were removed
- symbolic-regression attempts are not part of the curated result set

## Current Best Compact Description

The best compact physical model currently is:

\[
h(\theta_0) = \log\!\left(\frac{e}{\cos^2(\theta_0/2)}\right),
\]

\[
c_0(\theta_0) = A_0 h^{\alpha_0},
\]

\[
\log c_1(\theta_0) = B_0 + B_1 h + B_2 h^2,
\]

\[
p(\theta_0) = P_0 + P_1 c_0(\theta_0),
\]

\[
Y_2(\theta_0,t_p) = c_0(\theta_0) + \frac{c_1(\theta_0)}{t_p^{p(\theta_0)}}.
\]

Best-fit parameters:

- `A0 = 5.4099478821e-01`
- `alpha0 = 1.4633867627e-01`
- `B0 = -2.0889465029e+00`
- `B1 = 9.3879408645e-01`
- `B2 = -6.4950991313e-02`
- `P0 = 2.7048990951e+00`
- `P1 = -1.8121025642e+00`

Global fit quality:

- relative RMSE: `5.3297e-03`
- log RMSE: `5.3260e-03`

Conclusion:

- this remains the recommended final formula in the repo
- it is slightly worse than the slice-by-slice pendulum-log benchmark, but far more compact and interpretable

## Interpretation

- strict separability `f(\theta_0) g(t_p)` is not supported by the data
- the old raw power family fails even after allowing `A(t_p)` and `alpha(t_p)` freedom
- the data prefer a plateau plus algebraic transient in `t_p`
- the best angular control variable found so far is the pendulum-log hilltop variable
- if the goal is the best slice benchmark, use `pendulum_log + c0(t_p)`
- if the goal is the best compact global formula, use the 7-parameter `c0 + c1 / t_p^p` model

## Useful Files

Core scripts:

- `ode/analysis/test_y2_tp_shape_ansatz.py`
- `ode/analysis/check_y2_tp_shape_stability.py`
- `ode/analysis/refine_y2_physical_models.py`
- `ode/analysis/search_y2_final_models.py`
- `ode/analysis/fit_y2_tp_powerlaw.py`
- `ode/analysis/fit_y2_coeff_shapes.py`

Core outputs:

- `ode/analysis/results/noPT_reference_dm_tp_fitready_H1p000.txt`
- `ode/analysis/results/fanh_noPT_vs_theta0_dm_tp_fitready_H1p000.png`
- `ode/analysis/results/xi_vs_tp_dm_tp_fitready_H1p000.png`
- `ode/analysis/results/xi_panel_dm_tp_fitready_H1p000.png`
- `ode/analysis/results/xi_over_tp32_vs_tp_dm_tp_fitready_H1p000.png`
- `ode/analysis/results/xi_fanh_noPT_over_tp32_vs_tp_dm_tp_fitready_H1p000.png`
- `ode/analysis/results/y2_contour_dm_tp_fitready_H1p000.png`
- `ode/analysis/results/y2_log10_contour_dm_tp_fitready_H1p000.png`
- `ode/analysis/results/fit_y2_tp_powerlaw_dm_tp_fitready_H1p000.txt`
- `ode/analysis/results/fit_y2_tp_powerlaw_dm_tp_fitready_H1p000.png`
- `ode/analysis/results/fit_y2_tp_powerlaw_trends_dm_tp_fitready_H1p000.png`
- `ode/analysis/results/fit_y2_coeff_shapes_fit_y2_tp_powerlaw_dm_tp_fitready_H1p000.txt`
- `ode/analysis/results/fit_y2_coeff_shapes_fit_y2_tp_powerlaw_dm_tp_fitready_H1p000.png`
- `ode/analysis/results/test_y2_tp_shape_ansatz_dm_tp_fitready_H1p000.txt`
- `ode/analysis/results/test_y2_tp_shape_ansatz_log_hilltop_plus_const_vs_tp_dm_tp_fitready_H1p000.png`
- `ode/analysis/results/test_y2_tp_shape_ansatz_pendulum_log_plus_const_vs_tp_dm_tp_fitready_H1p000.png`
- `ode/analysis/results/check_y2_tp_shape_stability_dm_tp_fitready_H1p000.txt`
- `ode/analysis/results/check_y2_tp_shape_stability_dm_tp_fitready_H1p000.png`
- `ode/analysis/results/refine_y2_physical_models_dm_tp_fitready_H1p000.txt`
- `ode/analysis/results/final_fit_summary_dm_tp_fitready_H1p000.txt`

## Next Reasonable Step

If we want a better semi-analytic closure, the next target is not the old raw power law. It is:

1. fit smooth laws for `c0(t_p)`, `A(t_p)`, and `alpha(t_p)` in the stable slice benchmark
2. try to derive those laws from the first-oscillation energy-loss picture
3. keep the 7-parameter compact model as the default global fit unless a cleaner derivation appears
