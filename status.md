# Status

## Scope

Current analysis focuses on the direct `t_p` scan at fixed `H_* = 1` from:

- `ode/analysis/data/dm_tp_fitready_H1p000.txt`

Latest cross-checks also include lattice comparisons at `H_* = 1.5, 2.0` using:

- `lattice_data/data/rho_noPT_data.txt`
- `lattice_data/data/energy_ratio_by_theta_data_v9.txt`

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

### 10. Two-limit transition models

Tested compact forms that enforce both physical endpoints:

- small-`x = t_p / t_osc^noPT` should recover the no-PT limit
- large `t_p` should approach a finite plateau

The main matched-power form was

\[
Y_2(\theta_0,t_p) =
Y_\infty(\theta_0)
\left[
1 + \left(\frac{t_c(\theta_0)}{t_p}\right)^{q(\theta_0)}
\right]^{\frac{3}{2q(\theta_0)}},
\qquad
t_c(\theta_0)=\left(\frac{f_{\rm anh}^{\rm noPT}(\theta_0)}{Y_\infty(\theta_0)}\right)^{2/3}.
\]

Result:

- as a compact global model, the best matched-power version reached relative RMSE `4.9349e-03`
- using the analytic `f_anh_noPT(theta0)` fit instead of the table only shifted this to `5.0295e-03`
- as a free per-`\theta_0` slice family it was worse than the unconstrained powerlaw slices: `7.2113e-03` vs `4.2907e-03`
- the mismatch is mainly at large `\theta_0`, especially close to the hilltop

I also tested a sigmoid-in-log / logistic blending transition in `log(t_p/t_c)`.

Result:

- it failed badly
- free-slice relative RMSE was `2.0428e-01`
- best compact global relative RMSE was `2.0025e-01`
- the fitted sharpness parameter saturated at the imposed upper bound, indicating the family itself is not appropriate here

Conclusion:

- the matched-power two-limit model is a promising compact alternative
- the current 7-parameter `c0 + c1 / t_p^p` model remains the default until the matched form is reviewed more carefully
- the logistic-in-log blending attempt was rejected and removed from the working comparison script

### 11. Lattice no-PT calibration and fixed-`q` PT fits

To understand the lattice mismatch better, I separated the no-PT sector from the PT transition sector using:

- `lattice_data/data/rho_noPT_data.txt`
- `lattice_data/data/energy_ratio_by_theta_data_v9.txt`

For the `v9` lattice files, the wall velocity is fixed by the filename:

\[
v_w = 0.9.
\]

The raw lattice no-PT data factorize very well as

\[
\rho_{\rm noPT}(\theta_0,H_*) \simeq C(H_*)\,(1-\cos\theta_0)\,f_{\rm rel}(\theta_0),
\]

with small residual interaction:

- `interaction_std_log = 1.0650e-02`
- `interaction_max_log = 3.2130e-02`

The extracted relative no-PT curve agrees with the ODE no-PT reference well:

- all available `H_*`: relative RMSE `7.5127e-03`
- restricting to lattice `H_* = 1.5, 2.0`: relative RMSE `6.5976e-03`

This showed that the old no-PT ansatz

\[
f_{\rm anh}^{\rm noPT}(\theta_0)=A_f\,[\log(e/\cos^2(\theta_0/2))]^{\gamma_f}
\]

is too rigid for the lattice-facing fits. Its ODE no-PT fit quality is only:

- relative RMSE `1.1681e-01`

The better lattice-motivated no-PT parameterization is

\[
L(\theta_0) = \log\!\left(\frac{e}{1-(\theta_0/\pi)^2}\right),
\qquad
f_{\rm anh}^{\rm noPT}(\theta_0)=A_f\,L(\theta_0)^{\alpha_f},
\]

with fitted parameters

- `A_f = 3.9047143556e-01`
- `alpha_f = 1.8499066612e+00`

and ODE no-PT fit quality:

- relative RMSE `1.4800e-02`

Using this fixed no-PT law and fixing `q = 3/2`, the current working lattice fit is

\[
\xi(\theta_0,t_p)=
\frac{t_p^{3/2}}{\left[f_{\rm anh}^{\rm noPT}(\theta_0)\right]^2}
\left[
f_{\rm anh}^{\infty}(\theta_0)
+
c\,\frac{\left[f_{\rm anh}^{\rm noPT}(\theta_0)\right]^2}{t_p^{3/2}}
\right],
\]

with

\[
f_{\rm anh}^{\infty}(\theta_0)=A_\infty\,L(\theta_0)^{\gamma_\infty}.
\]

Best fixed-`q` lattice fits:

- `H_* = 1.5`: `A_inf = 1.5134072092e-01`, `gamma_inf = 2.2582802732e+00`, `c = 1.0241139892e+00`, relative RMSE `3.8546e-03`
- `H_* = 2.0`: `A_inf = 1.5399728334e-01`, `gamma_inf = 2.2167756787e+00`, `c = 1.0194792055e+00`, relative RMSE `5.7979e-03`

Important follow-up checks:

- once the new no-PT law is fixed, adding the old crossover factor in `t_p` no longer improves the fit materially; the fitted `t_c` runs large and the transition factor is effectively switched off
- replacing `t_p` by `x=t_p/t_*=2H_* t_p` in a shared-`H_*` fit only changes the combined relative RMSE from `5.3490e-03` to `5.2060e-03`, so the remaining tension is not mainly the choice of transition variable
- the remaining mismatch still grows with `theta_0`, and `H_* = 2.0` is statistically harsher because its lattice SEMs are smaller

Conclusion:

- the old lattice mismatch was driven in large part by the no-PT parameterization, not only by the PT transition shape
- the current lattice working model should keep `v_w = 0.9` fixed, keep `q = 3/2` fixed, and keep the new no-PT law fixed
- the next transition-side work should focus on the remaining `\theta_0`/`H_*` dependence in the PT sector, not on refitting the no-PT law again

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
- for lattice-facing no-PT fits, the better control variable is currently `L(\theta_0)=\log(e/(1-(\theta_0/\pi)^2))`
- if the goal is the best slice benchmark, use `pendulum_log + c0(t_p)`
- if the goal is the best compact global formula, use the 7-parameter `c0 + c1 / t_p^p` model
- if the goal is the current best lattice working fit, use fixed `v_w=0.9`, fixed `q=3/2`, and the fixed no-PT hilltop-log law from section 11

## Useful Files

Core scripts:

- `ode/analysis/test_y2_tp_shape_ansatz.py`
- `ode/analysis/check_y2_tp_shape_stability.py`
- `ode/analysis/refine_y2_physical_models.py`
- `ode/analysis/search_y2_final_models.py`
- `ode/analysis/fit_y2_tp_powerlaw.py`
- `ode/analysis/fit_y2_coeff_shapes.py`
- `ode/analysis/compare_y2_matched_limit_model.py`
- `ode/analysis/analyze_lattice_nopt_rho.py`
- `ode/analysis/fit_xi_lattice_crossover_shape.py`
- `ode/analysis/fit_lattice_shared_transition_variable.py`

Core outputs:

- `ode/analysis/results/noPT_reference_dm_tp_fitready_H1p000.txt`
- `ode/analysis/results/fanh_noPT_vs_theta0_dm_tp_fitready_H1p000.png`
- `ode/analysis/results/xi_vs_tp_dm_tp_fitready_H1p000.png`
- `ode/analysis/results/xi_panel_dm_tp_fitready_H1p000.png`
- `ode/analysis/results/xi_over_tp32_vs_tp_dm_tp_fitready_H1p000.png`
- `ode/analysis/results/xi_fanh_noPT_over_tp32_vs_tp_dm_tp_fitready_H1p000.png`
- `ode/analysis/results/xi_fit_vs_tp_dm_tp_fitready_H1p000.png`
- `ode/analysis/results/xi_compare_models_vs_tp_dm_tp_fitready_H1p000.png`
- `ode/analysis/results/xi_compare_models_vs_tp_dm_tp_fitready_H1p000.txt`
- `ode/analysis/results/y2_contour_dm_tp_fitready_H1p000.png`
- `ode/analysis/results/y2_log10_contour_dm_tp_fitready_H1p000.png`
- `ode/analysis/results/refine_y2_physical_models_dm_tp_fitready_H1p000.txt`
- `ode/analysis/results/final_fit_summary_dm_tp_fitready_H1p000.txt`
- `ode/analysis/results/lattice_fit/noPT_rho_study/lattice_nopt_summary.txt`
- `ode/analysis/results/lattice_fit/noPT_rho_study/lattice_nopt_relative_collapse.png`
- `ode/analysis/results/lattice_fit/crossover_shape_v9_fixed_nopt_hilltoplog/fit_xi_lattice_crossover_shape_summary.txt`
- `ode/analysis/results/lattice_fit/crossover_shape_v9_fixed_nopt_hilltoplog/fit_xi_lattice_crossover_shape_H1p5.png`
- `ode/analysis/results/lattice_fit/crossover_shape_v9_fixed_nopt_hilltoplog/fit_xi_lattice_crossover_shape_H2p0.png`
- `ode/analysis/results/lattice_fit/shared_transition_variable_v9_fixed_nopt_hilltoplog/fit_lattice_shared_transition_variable_summary.txt`

Old model-attempt outputs now live in:

- `ode/analysis/results/old_attempts/fit_y2_tp_powerlaw_dm_tp_fitready_H1p000.txt`
- `ode/analysis/results/old_attempts/fit_y2_tp_powerlaw_dm_tp_fitready_H1p000.png`
- `ode/analysis/results/old_attempts/fit_y2_tp_powerlaw_trends_dm_tp_fitready_H1p000.png`
- `ode/analysis/results/old_attempts/fit_y2_coeff_shapes_fit_y2_tp_powerlaw_dm_tp_fitready_H1p000.txt`
- `ode/analysis/results/old_attempts/fit_y2_coeff_shapes_fit_y2_tp_powerlaw_dm_tp_fitready_H1p000.png`
- `ode/analysis/results/old_attempts/test_y2_tp_shape_ansatz_dm_tp_fitready_H1p000.txt`
- `ode/analysis/results/old_attempts/test_y2_tp_shape_ansatz_log_hilltop_plus_const_vs_tp_dm_tp_fitready_H1p000.png`
- `ode/analysis/results/old_attempts/test_y2_tp_shape_ansatz_pendulum_log_plus_const_vs_tp_dm_tp_fitready_H1p000.png`
- `ode/analysis/results/old_attempts/check_y2_tp_shape_stability_dm_tp_fitready_H1p000.txt`
- `ode/analysis/results/old_attempts/check_y2_tp_shape_stability_dm_tp_fitready_H1p000.png`
- `ode/analysis/results/old_attempts/search_y2_final_models_dm_tp_fitready_H1p000.txt`
- `ode/analysis/results/old_attempts/search_y2_final_models_tradeoff_dm_tp_fitready_H1p000.png`
- `ode/analysis/results/old_attempts/search_y2_final_models_best_params_dm_tp_fitready_H1p000.png`
- `ode/analysis/results/old_attempts/search_y2_final_models_vs_theta_dm_tp_fitready_H1p000.png`
- `ode/analysis/results/old_attempts/compare_y2_matched_limit_model_dm_tp_fitready_H1p000.txt`
- `ode/analysis/results/old_attempts/compare_y2_matched_limit_slices_dm_tp_fitready_H1p000.png`
- `ode/analysis/results/old_attempts/compare_y2_matched_limit_global_dm_tp_fitready_H1p000.png`

## Next Reasonable Step

If we want a better semi-analytic closure, the next target is now split by use case:

1. for the ODE `H_* = 1` compact description, keep the 7-parameter `c0 + c1/t_p^p` model as the repo default unless a cleaner derivation appears
2. for the lattice comparison, keep `v_w = 0.9`, `q = 3/2`, and the fixed no-PT hilltop-log law, and focus on the remaining PT-sector transition mismatch at larger `\theta_0`
3. test whether the residual `H_* = 2.0` tension is coming from explicit `H_*` dependence in `f_{\rm anh}^{\infty}` / the PT plateau sector or from the `\beta/H_* \to t_p` mapping, rather than from the no-PT law
