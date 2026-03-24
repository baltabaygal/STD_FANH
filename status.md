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

## Conventions

For this project, the canonical form for the lattice/ODE transition fits is

\[
\xi(\theta_0,t_p) =
\left(\frac{2\,t_p}{3\,t_{\rm osc}}\right)^{3/2}
\frac{\tilde f(\theta_0,t_p)}{F_0(\theta_0)}.
\]

Use this `\tilde f` parameterization as the primary one in all discussion and fitting code.
Do not treat the additive `\xi = \text{plateau} + \text{transient}` form as the canonical model statement.

This prefactor is the scale-factor ratio

\[
\left(\frac{a_p}{a_{\rm osc}}\right)^3
\]

with

\[
a \propto \sqrt{2Ht}
\quad\text{and}\quad
3H_{\rm osc}=m_\phi,
\]

so the canonical prefactor is

\[
\left(\frac{a_p}{a_{\rm osc}}\right)^3
=
\left(\frac{2\,t_p}{3\,t_{\rm osc}}\right)^{3/2}.
\]

For the current transition kernel, the corresponding `\tilde f` ansatz is

\[
\tilde f(\theta_0,t_p) =
f_\infty(\theta_0)
+
\frac{F_0(\theta_0)}
{\left(\frac{2\,t_p}{3\,t_{\rm osc}}\right)^{3/2}\left[1+\left(\frac{t_p}{t_c}\right)^r\right]},
\]

with

\[
f_\infty(\theta_0) \equiv \frac{F_\infty(\theta_0)}{F_0(\theta_0)}.
\]

or the same expression multiplied by an explicit calibration factor only if that factor is intentionally included in the model.

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

### 12. Collapsed-coordinate fits and scalar-`\gamma` amplitude tests

I then tested whether the remaining `H_*` dependence can be absorbed by a single collapsed coordinate

\[
x = t_p\,H_*^\beta,
\]

and, if not, whether a residual amplitude factor

\[
f_{\rm anhr}(\theta_0,t_p,H_*) = H_*^\gamma\,f_{\rm univ}(\theta_0,x)
\]

is enough to describe the lattice data.

For the full `v9` dataset (`v_w = 0.9`) across `H_* = 0.5, 1.0, 1.5, 2.0`, the best collapse exponent is

- `beta = -9.7618e-02`

with a very small collapse score

- `S(beta) = 8.3268e-05`

and the collapsed global fit gives

- `t_c = 2.9846`
- `r = 4.1519`
- relative RMSE `1.6245e-02`

Important caveat:

- although the best-fit `beta` is close to zero, the one-curve collapsed fit is still worse than fitting the `H_*` slices independently (`delta AIC = -799.13`, `delta BIC = -757.67` in favor of the separate fits)

So a pure `x=t_p H_*^\beta` collapse is not enough by itself.

For the restricted lattice comparison `H_* = 1.5, 2.0`, the collapse quality depends strongly on wall velocity:

- `v_w = 0.3`: `beta = 4.8064e-01`, relative RMSE `1.8155e-02`
- `v_w = 0.5`: `beta = 3.2874e-01`, relative RMSE `8.9520e-03`
- `v_w = 0.7`: `beta = 1.4331e-01`, relative RMSE `6.1755e-03`
- `v_w = 0.9`: `beta = -2.4345e-03`, relative RMSE `4.2390e-03`

Conclusion from the collapse scan:

- as `v_w` increases, the best collapse exponent moves steadily toward `beta = 0`
- `v_w = 0.9` is clearly the cleanest lattice dataset for testing residual `H_*` dependence

Using that same collapse exponent as input, I next fit a scalar amplitude law

\[
f_{\rm anhr}(\theta_0,t_p,H_*) = H_*^\gamma\,f_{\rm univ}(\theta_0,x)
\]

with `\gamma` taken to be a single global scalar.

For the full `v9` dataset across `H_* = 0.5, 1.0, 1.5, 2.0`, the pooled amplitude estimate is

- `gamma_{\rm pooled} = -1.1735e-01`

with 95% CI

- `[-1.2113e-01,\,-1.1357e-01]`

and the scalar-`\gamma` nonlinear fit gives

- `gamma = -1.4522e-01`
- `t_c = 4.5643`
- `r = 3.4191`
- relative RMSE `1.9958e-02`

On the restricted lattice-only `H_* = 1.5, 2.0` `v9` subset, the scalar-`\gamma` fit is much cleaner:

- `gamma_{\rm pooled} = -1.1491e-01`
- scalar-fit `gamma = -1.1923e-01`
- `t_c = 1.3839`
- `r = 9.4498`
- relative RMSE `4.9204e-03`

and it is strongly preferred over the no-`\gamma` collapsed model:

- `delta AIC = -735.55`
- `delta BIC = -735.55`

Finally, repeating the scalar-`\gamma` fit for all four wall velocities shows:

- `v_w = 0.3`: `gamma = -1.6117e-02`, `t_c = 3.5663`, `r = 17.91`, relative RMSE `3.9515e-02`
- `v_w = 0.5`: `gamma = -2.1436e-02`, `t_c = 19.96`, `r = 20.00`, relative RMSE `1.8097e-02`
- `v_w = 0.7`: `gamma = -1.1090e-01`, `t_c = 15.90`, `r = 16.90`, relative RMSE `1.1125e-02`
- `v_w = 0.9`: `gamma = -1.1923e-01`, `t_c = 1.3839`, `r = 9.4498`, relative RMSE `4.9204e-03`

Interpretation:

- only the high-`v_w` runs, especially `v9`, give a clean and stable negative scalar `\gamma`
- for `v3` and `v5`, the fit compensates by pushing `r` and `t_c` to extreme values, so those scalar-`\gamma` results are not yet trustworthy
- the current best lattice-facing `H_*` extension is therefore: use `v_w = 0.9`, keep the fixed no-PT law, and include a negative scalar amplitude exponent `\gamma \simeq -0.12`

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

### 13. vw-dependence diagnosis: effective-time correction and parameter evolution

Two diagnostic studies were run to understand why the FANH collapse model degrades at lower wall velocities.

All fits used lattice data at `H_* = 0.5, 1.0, 1.5, 2.0` for each of the four wall velocities `v_w \in \{0.3, 0.5, 0.7, 0.9\}`.

#### 13a. Effective-time correction study (`study_vw_teff_correction.py`)

The reference collapse model was fitted to the `v_w = 0.9` data, giving

- `beta = -0.097`, `t_c = 2.996`, `r = 4.167`, relative RMSE `1.6\%`

Then, for each other `v_w`, two corrections to `t_p` were scanned while keeping all model parameters fixed to the `v_w = 0.9` reference:

- **multiplicative**: `t_p \to t_p \cdot s_{v_w}`
- **additive**: `t_p \to t_p + \delta_{v_w}`

Results:

| \(v_w\) | no correction | multiplicative | additive |
|---------|--------------|----------------|---------|
| 0.9 | 1.6\% | 1.6\% (s=1) | 1.6\% |
| 0.7 | 2.5\% | 2.2\% (s=1.02) | 1.8\% |
| 0.5 | 7.4\% | 5.6\% (s=1.07) | 3.1\% |
| 0.3 | 11.6\% | 8.3\% (s=1.11) | 3.7\% |

The additive shift is far more effective than the multiplicative rescaling.  At `v_w = 0.3` it recovers 3× more accuracy.  The best-fit shifts are approximately

- `delta(0.7) approx 0.033`, `delta(0.5) approx 0.131`, `delta(0.3) approx 0.225`

which grow as `v_w` decreases and are consistent with a `A * (1/v_w - 1/v_{w,ref})` scaling.

Physical interpretation: slow bubble walls add a roughly constant propagation delay before the axion field fully transitions.  This is a phase offset in `t_p`, not a time dilation.

#### 13b. Parameter evolution study (`study_vw_param_evolution.py`)

For each `v_w` independently, the best collapse exponent and FANH parameters were found:

| \(v_w\) | \(\beta\) | \(t_c\) | \(r\) | in-sample RMSE | LOO RMSE |
|---------|-----------|---------|-------|----------------|----------|
| 0.9 | \(-0.097\) | 2.996 | 4.17 | 1.6\% | 2.8\% |
| 0.7 | \(-0.084\) | 2.891 | 4.31 | 2.0\% | 2.4\% |
| 0.5 | \(-0.112\) | 2.325 | 5.64 | 5.7\% | 4.9\% |
| 0.3 | \(-0.043\) | 2.504 | 5.71 | 6.5\% | 13.6\% |

Key observations:

- `beta` is nearly constant across all `v_w` (range \(-0.04\) to \(-0.11\)), confirming that the `H_*` scaling is not the source of the degradation.
- The transient exponent `r` increases significantly at low `v_w` (from \(\sim 4\) to \(\sim 5.7\)), indicating a sharper transient shape for slow walls.
- The LOO error for `v_w = 0.3` (13.6\%) is much larger than its in-sample error (6.5\%), meaning the `v_w = 0.3` dynamics are not smoothly interpolable from the higher-`v_w` fits.

Conclusion:

- the dominant source of degradation is a `v_w`-dependent additive offset in `t_p`, not a rescaling of the time axis or a change in the `H_*` collapse exponent
- the next targeted step is to fit a single global formula `delta(v_w) = A \cdot (1/v_w - 1/v_{w,ref})` and fold this additive correction into the model
- additionally, the jump in `r` at low `v_w` may require a vw-dependent transient shape, which could be the secondary correction after the time shift is absorbed

## Interpretation

- strict separability `f(\theta_0) g(t_p)` is not supported by the data
- the old raw power family fails even after allowing `A(t_p)` and `alpha(t_p)` freedom
- the data prefer a plateau plus algebraic transient in `t_p`
- the best angular control variable found so far is the pendulum-log hilltop variable
- for lattice-facing no-PT fits, the better control variable is currently `L(\theta_0)=\log(e/(1-(\theta_0/\pi)^2))`
- if the goal is the best slice benchmark, use `pendulum_log + c0(t_p)`
- if the goal is the best compact global formula, use the 7-parameter `c0 + c1 / t_p^p` model
- if the goal is the current best lattice working fit, use fixed `v_w=0.9`, fixed `q=3/2`, and the fixed no-PT hilltop-log law from section 11
- if the goal is the current best lattice `H_*` extension, use the `v_w=0.9` collapsed-coordinate fit with a scalar negative amplitude exponent `\gamma \simeq -0.12`
- if the goal is extending the model to all `v_w`, the leading correction is an additive `t_p` shift `delta(v_w) = A*(1/v_w - 1/v_{w,ref})`; pure multiplicative rescaling is insufficient

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
- `collapse_and_fit_fanh.py`
- `measure_gamma_and_refit.py`
- `compare_vw_collapse_results.py`
- `compare_vw_gamma_results.py`
- `study_vw_teff_correction.py`
- `study_vw_param_evolution.py`

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
- `results_collapse/final_summary.json`
- `results_collapse/best_beta_plot.png`
- `results_collapse/collapse_overlay.png`
- `results_collapse_vw_compare/vw_collapse_summary.csv`
- `results_collapse_vw_compare/vw_collapse_comparison.png`
- `results_gamma/final_summary.json`
- `results_gamma/gamma_vs_theta.png`
- `results_gamma/collapse_overlay_with_gamma.png`
- `results_gamma_v9_H15H20_scalar/final_summary.json`
- `results_gamma_v9_H15H20_scalar/collapse_overlay_with_gamma.png`
- `results_gamma_vw_compare/vw_gamma_summary.csv`
- `results_gamma_vw_compare/vw_gamma_comparison.png`

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

---

### 15. ODE-anchored lattice fitting: fixing F_inf from ODE, per-(θ₀,vw) tc

**Strategy**: instead of letting F_inf(θ₀) float freely in lattice fits, extract them from the direct ODE scan and freeze them, so the lattice fit only adjusts the transient parameters (β, tc, r). This connects the lattice model directly to first-principles ODE physics.

**Step 1 — ODE direct fit** (`fit_ode_direct_transition_ansatz.py`, H* = 0.5, 1, 1.5, 2):
Model: `xi = (2tp/3t_osc)^(3/2) · F_inf(θ)/F0² + 1/(1 + (tp/tc)^r)`, shared tc and r, per-θ₀ F_inf.
Result: **tc → 50 (boundary), r = 0.56** — ODE transient is extremely slow; tc rails at the upper bound. Per-θ₀ F_inf values extracted for use in lattice fits.

**Step 2 — ODE-fixed amplitudes on lattice** (`fit_lattice_fixed_ode_amplitudes.py`, vw=0.9, H*=1,1.5,2):

| Case | β | tc | r | rel_rmse |
|------|---|----|---|----------|
| free_tc | **−0.212** | 2.23 | 3.12 | **3.4%** |
| free_tc + calib (c) | −0.136 | 2.20 | 4.09 | 3.3% |
| fixed tc=1.5 | −0.263 | 1.5 | 12.0 | 4.2% |
| fixed tc=1.5 + calib | −0.102 | 1.5 | 20(rail) | 3.5% |

Reference (free F_inf): β=−0.095, tc=2.22, r=2.13, rel_rmse=**0.7%**. Fixing ODE amplitudes costs ~5× accuracy — ODE and lattice F_inf differ by a systematic factor ~1.2 across all θ₀.

**Step 3 — Per-θ₀ tc, shared r** (`fit_lattice_fixed_ode_amplitudes_theta_tc.py`): allows tc(θ₀) to vary per angular slice.

**Step 4 — Free F_inf per (θ₀, vw), shared r** (`fit_lattice_theta_tc_shared_r_free_finf_by_vw.py`):
Most complete model: per-(θ₀,vw) tc, shared r=2.78, free F_inf per θ₀.

| vw | rel_rmse |
|----|----------|
| 0.3 | 4.0% |
| 0.5 | 2.7% |
| 0.7 | 1.9% |
| 0.9 | 2.3% |
| **global** | **2.9%** |

tc(θ₀,vw) structure: tc increases with θ₀ (hilltop takes longer to transition) and with lower vw (slow walls → delayed transient), e.g. θ₀≈0.26: tc=0.97 (vw=0.9) → 1.81 (vw=0.3).

**Step 5 — (θ₀,vw)-dependent F_inf tests** (`fit_lattice_theta_tc_shared_r_free_finf_vwtheta_tests.py`):
Benchmark (shared r, F_inf per θ₀+vw): global rel_rmse = **1.9%**, per vw: 2.6% / 2.1% / 1.3% / 1.4%.
Variant with (θ₀,vw)-dependent F_inf but fixed t_osc: global 2.7% — using F_inf(θ₀,vw) alone (without per-vw tc) does not improve over the benchmark. The tc(θ₀,vw) variation is the dominant driver.

**Interpretation**: The transient timescale tc depends on both θ₀ and vw in a physically motivated way. F_inf from ODE is systematically ~20% below lattice values (lattice/ODE ratio ~1.2–1.4, increasing toward hilltop), indicating a real lattice vs ODE amplitude discrepancy. The best current model (benchmark, step 5) achieves 1.9% global RMSE across all vw — similar in quality to the pointwise shift pipeline (1.6%) but with a more interpretable model structure.

**Key outputs**:
- `results_ode_direct_transition_ansatz/final_summary.json`
- `results_lattice_fixed_ode_amplitudes_vw0p9_H1p0H1p5H2p0/final_summary.json`
- `results_lattice_theta_tc_sharedr_free_finf_by_vw_beta0_tcmax300/final_summary.json`
- `results_lattice_theta_tc_sharedr_free_finf_vwtheta_tests_beta0_tcmax300/final_summary.json`

---

### 14. Pointwise shift pipeline: all-vw universality via s(tₚ, vw)

**What was done** (4-script pipeline):

1. **`infer_pointwise_shift_from_vw0p9_baseline.py`** — For each data point at any vw, inverts the frozen vw=0.9 FANH baseline curve ξ(x) to find `x_eff` such that `ξ_baseline(x_eff) = ξ_obs`. Defines `s = x_eff / x_base` as the pointwise multiplicative time shift needed. Inversion fraction: 100% (792/792 points). Interpolation accuracy: median |frac| residual ≈ 6×10⁻⁷ (essentially exact).

2. **`fit_pointwise_shift_powerlaw.py`** — Fits 3 compact formulas to s(tₚ, vw):
   - Best: `s = 1 + A[(0.9/vw)^m − 1]·tₚ^p`  with **A = 0.606, m = 0.281, p = −1.325** (rel_rmse = 3.3%)
   - Runner-up: anchored log-power (rel_rmse = 3.4%)
   - Plain power law (rel_rmse = 5.2%)

3. **`apply_pointwise_shift_powerlaw.py`** — Applies the fitted shift formula as `x_eff = s·tₚ·H*^β` and evaluates against all vw data:

   | vw  | Baseline rel_rmse | Shift rel_rmse | Improvement |
   |-----|-------------------|----------------|-------------|
   | 0.3 | 12.1%             | 2.1%           | 5.8×        |
   | 0.5 | 7.6%              | 1.9%           | 4.0×        |
   | 0.7 | 1.9%              | 1.4%           | 1.3×        |
   | 0.9 | 0.7%              | 0.7%           | 1.0× (ref)  |
   | **global** | **7.2%**  | **1.6%**       | **4.5×**    |

4. **`fit_shift_width_correction.py`** — Tests whether a second-order width correction `δξ ≈ ½σ²·d²ξ/dx²` with `σ²(tₚ,vw) = B[(0.9/vw)^m − 1]·tₚ^p` further improves residuals. Result: width correction provides essentially no improvement (global rel_rmse 1.61% → 1.61%). The fitted m≈0.0003 is essentially zero, meaning width is not vw-dependent at this level of precision. Curvature correlations with residuals are weak (0.11–0.19).

**Interpretation**: The single compact formula `s = 1 + 0.606·[(0.9/vw)^m − 1]·tₚ^p` with m=0.281, p=−1.325 effectively unifies all four wall velocities. The shift is largest at low vw (median s≈1.24 for vw=0.3) and decays with tₚ (negative p), consistent with slower walls needing a larger effective time stretch at early times. The width correction is negligible — the residual after shift is already ≲2% for all vw.

**Key outputs**:
- `results_pointwise_shift_from_vw0p9_baseline/pointwise_shift_table.csv`
- `results_pointwise_shift_powerlaw/final_summary.json`
- `results_pointwise_shift_powerlaw_applied/final_summary.json`
- `results_shift_width_correction/final_summary.json`

**Supervisor presentation figures** (`present_project/`):
- `fig7_s_tp_vw.png` — inferred s(tₚ, vw) scatter + fitted formula curves for each vw; right panel: s vs vw at three fixed tₚ quantiles (diamonds = binned data medians, lines = formula)
- `fig8_shift_collapse.png` — all-vw data collapsed onto the FANH baseline via x_eff = s·tₚ·H*^β; one panel per θ₀
- `fig9_residual_comparison.png` — 2-row×4-col residual comparison (baseline top, shift bottom) with bar-chart inset showing RMSE improvement by vw

---

### 15. Shared-`r` lattice fits across all `v_w` with `t_c(\theta_0; v_w)` (March 24, 2026)

Tested a joint lattice fit across `v_w = 0.3, 0.5, 0.7, 0.9` with:

- fixed `\beta = 0`, so `x = t_p`
- one shared global `r`
- free `t_c(\theta_0; v_w)`
- ODE-fixed `F_0(\theta_0)`
- transient kernel

\[
\frac{1}{1 + (t_p/t_c)^r}
\]

as the default working form

#### 15a. ODE-fixed `F_\infty(\theta_0)`, `c_{calib}=1`

Result:

- relative RMSE `3.8710e-02`
- `r = 1.9956`

Behavior:

- the hilltop `t_c` values ran to the imposed upper bound for several slices
- the worst saturation was at large `\theta_0`, especially for higher `v_w`

Interpretation:

- this was the first clear sign that the large-`\theta_0` turnover is weakly constrained if the plateau is held fixed too rigidly

Key output:

- `results_lattice_fixed_ode_amplitudes_theta_tc_sharedr_by_vw_beta0_tcmax300/final_summary.json`

#### 15b. ODE-fixed `F_\infty(\theta_0)`, free `c_{calib}(v_w)`

Result:

- relative RMSE `1.9424e-02`
- `r = 2.2287`
- `c_{calib}(v_w) = {0.3: 1.1217, 0.5: 1.0801, 0.7: 1.0022, 0.9: 0.9912}`

Behavior:

- this is the best-performing shared-`r` fit tried in this round
- hilltop `t_c` saturation was reduced but not eliminated: the largest `\theta_0` point still wanted `t_c > 300` for `v_w = 0.7, 0.9`

Interpretation:

- allowing a simple per-`v_w` normalization absorbs much of the mismatch while keeping the ODE plateau shape fixed

Key outputs:

- `results_lattice_fixed_ode_amplitudes_theta_tc_sharedr_by_vw_beta0_tcmax300/collapse_overlay_sharedr_calib.png`
- `results_lattice_fixed_ode_amplitudes_theta_tc_sharedr_by_vw_beta0_tcmax300/tc_by_vw_sharedr_calib.png`
- `results_lattice_fixed_ode_amplitudes_theta_tc_sharedr_by_vw_beta0_tcmax300/final_summary.json`

#### 15c. Failed alternative kernel

Tested the shifted kernel

\[
\frac{1}{(1 + t_p/t_c)^r}
\]

instead of

\[
\frac{1}{1 + (t_p/t_c)^r}.
\]

Result:

- calibrated relative RMSE degraded to `2.8858e-02`
- fitted `r` jumped to `19.626`
- hilltop `t_c` saturation remained

Conclusion:

- rejected
- the blow-up problem was not caused by using the wrong kernel
- the preferred kernel remains

\[
\frac{1}{1 + (t_p/t_c)^r}
\]

#### 15d. `c_{calib}=1` and free lattice-fit `F_\infty(\theta_0)`

To test whether the ODE plateau itself was too low, refit with:

- ODE-fixed `F_0(\theta_0)`
- free lattice `F_\infty(\theta_0)`
- shared `r`
- free `t_c(\theta_0; v_w)`
- no `c_{calib}`

Result:

- relative RMSE improved from `3.8710e-02` to `2.8755e-02`
- `r = 2.7787`
- all fitted `t_c(\theta_0; v_w)` values stayed finite, roughly `0.97` to `3.12`

Fitted plateau shift:

- `F_\infty^{lattice} / F_\infty^{ODE} = [1.203, 1.214, 1.233, 1.257, 1.285, 1.418]`

Interpretation:

- part of the earlier `t_c` blow-up was compensating for an ODE plateau that was systematically too low
- freeing `F_\infty(\theta_0)` removes the runaway `t_c` behavior
- however, this still does not beat the simpler ODE-fixed + free-`c_{calib}(v_w)` fit

Conclusion:

- best fit quality in this family: ODE-fixed `F_\infty` with free `c_{calib}(v_w)`
- best fit with `c_{calib}=1`: free lattice `F_\infty(\theta_0)`

Key outputs:

- `results_lattice_theta_tc_sharedr_free_finf_by_vw_beta0_tcmax300/collapse_overlay_sharedr_free_finf.png`
- `results_lattice_theta_tc_sharedr_free_finf_by_vw_beta0_tcmax300/tc_by_vw_sharedr_free_finf.png`
- `results_lattice_theta_tc_sharedr_free_finf_by_vw_beta0_tcmax300/f_infty_theta_sharedr_free_finf.png`
- `results_lattice_theta_tc_sharedr_free_finf_by_vw_beta0_tcmax300/final_summary.json`
- `results_lattice_theta_tc_sharedr_free_finf_by_vw_beta0_tcmax300/comparison_sheet.png`

#### 15e. Cleanup performed

Removed after the comparison was recorded in `comparison_sheet.png`:

- failed shifted-kernel output directory `results_lattice_fixed_ode_amplitudes_theta_tc_sharedr_by_vw_beta0_tcmax300_shifted`
- stale aggregate error file `results_lattice_fixed_ode_amplitudes_theta_tc_by_vw_beta0_tcmax100/_error.json`

#### 15f. Testing explicit `H_*` corrections in the shared-`r`, free-`f_\infty`, `c_{calib}=1` family

Starting point:

- baseline model kept the canonical form

\[
\xi = \left(\frac{x}{t_{\rm osc}}\right)^{3/2} \frac{\tilde f}{F_0},
\qquad
\tilde f = f_\infty(\theta_0) + \frac{F_0(\theta_0)}{\left(\frac{x}{t_{\rm osc}}\right)^{3/2}\left[1 + \left(\frac{x}{t_c}\right)^r\right]}
\]

with:

- shared `r`
- free `t_c(\theta_0; v_w)`
- free lattice `f_\infty(\theta_0)`
- fixed `\beta = 0`
- no `c_{calib}`

Question:

- does the remaining bad `H_* = 2.0` behavior come from a missing explicit `H_*` correction?

Baseline diagnostic:

- yes, `H_* = 2.0` remains the worst slice
- mean raw RMSE by `H_*`:
  - `H_* = 1.0`: `2.6225e-02`
  - `H_* = 1.5`: `2.0805e-02`
  - `H_* = 2.0`: `3.0864e-02`
- worst subgroup was `H_* = 2.0, v_w = 0.3` with mean raw RMSE `5.1074e-02`

Two one-parameter extensions were tested:

1. timing-only correction

\[
t_c \to t_c(\theta_0; v_w)\,H_*^{\alpha_{tc}}
\]

2. amplitude correction

\[
\tilde f \to H_*^\gamma \tilde f
\]

Results:

- baseline:
  - relative RMSE `2.8755e-02`
  - `r = 2.7787`
- timing-only `t_c H_*^{\alpha_{tc}}`:
  - relative RMSE `2.7058e-02`
  - `r = 3.2082`
  - `\alpha_{tc} = -0.2001`
- amplitude-only `H_*^\gamma`:
  - relative RMSE `2.8726e-02`
  - `r = 2.6985`
  - `\gamma = 6.33 \times 10^{-3}`

Interpretation:

- the amplitude-shift test is effectively null: `\gamma` is driven to zero and the fit quality is unchanged
- the timing-only `t_c H_*^{\alpha_{tc}}` correction helps the global fit and improves `H_* = 1.0`, but it does **not** fix the `H_* = 2.0` problem
- mean raw RMSE at `H_* = 2.0` stayed essentially flat:
  - baseline: `3.0864e-02`
  - timing-only: `3.0826e-02`
  - amplitude-only: `3.0616e-02`
- in particular, the worst subgroup `H_* = 2.0, v_w = 0.3` did not improve:
  - baseline: `5.1074e-02`
  - timing-only: `5.2718e-02`
  - amplitude-only: `4.9613e-02`

Conclusion:

- there is still a real high-`H_*` tension in the `c_{calib}=1`, free-`f_\infty`, shared-`r` model
- a global amplitude factor `H_*^\gamma` does not explain it
- a simple power-law timing correction `t_c \propto H_*^{\alpha_{tc}}` is not enough to resolve it either

Key outputs:

- `results_lattice_theta_tc_sharedr_hstar_tests_beta0_tcmax300/comparison_sheet.png`
- `results_lattice_theta_tc_sharedr_hstar_tests_beta0_tcmax300/rmse_by_h.png`
- `results_lattice_theta_tc_sharedr_hstar_tests_beta0_tcmax300/rmse_by_h_vw.png`
- `results_lattice_theta_tc_sharedr_hstar_tests_beta0_tcmax300/baseline/xi_vs_betaH_H2p0.png`
- `results_lattice_theta_tc_sharedr_hstar_tests_beta0_tcmax300/tc_hpow/xi_vs_betaH_H2p0.png`
- `results_lattice_theta_tc_sharedr_hstar_tests_beta0_tcmax300/gamma_amp/xi_vs_betaH_H2p0.png`
- `results_lattice_theta_tc_sharedr_hstar_tests_beta0_tcmax300/final_summary.json`

#### 15g. Rechecking the earlier good collapse and testing `t_{osc}` / full-time rescaling

Motivation:

- the earlier well-collapsing multi-`v_w` fit was the ODE-fixed `f_\infty(\theta_0)` model with free `c(v_w)`
- the later `c_{calib}=1`, free-lattice-`f_\infty(\theta_0)` fit was visibly worse at high `H_*`
- this raised two follow-up tests:
  1. keep the earlier calibrated ODE-fixed model, but fit a global `t_{osc}`
  2. try an alternate parametrization where `s(v_w,\theta_0) = 1/t_c(v_w,\theta_0)` is promoted to a direct time-rescaling factor and is applied to every `t_p` entering the ansatz

Canonical calibrated baseline rechecked:

- model: ODE-fixed `f_\infty(\theta_0)`, free `c(v_w)`, free `t_c(\theta_0;v_w)`, shared `r`, fixed `t_{osc}=1.5`
- result reproduced exactly:
  - relative RMSE `1.9424e-02`
  - `r = 2.2287`
  - `c(v_w) = {0.3: 1.1217, 0.5: 1.0801, 0.7: 1.0022, 0.9: 0.9912}`
- mean raw RMSE by `H_*`:
  - `H_*=1.0`: `1.7770e-02`
  - `H_*=1.5`: `1.8280e-02`
  - `H_*=2.0`: `1.6801e-02`

Conclusion:

- the earlier good collapse was real
- there was no serious residual `H_*=2` problem in this calibrated ODE-fixed family

Free-`t_{osc}` test:

- same model family, but with one extra global fit parameter `t_{osc}`
- result:
  - relative RMSE `1.9362e-02`
  - `r = 2.0568`
  - fitted `t_{osc} = 1.5430`
  - `c(v_w) = {0.3: 1.1343, 0.5: 1.0899, 0.7: 1.0101, 0.9: 0.9988}`
- mean raw RMSE by `H_*`:
  - `H_*=1.0`: `1.7586e-02`
  - `H_*=1.5`: `1.8184e-02`
  - `H_*=2.0`: `1.6516e-02`

Interpretation:

- freeing `t_{osc}` helps slightly, but only slightly
- the calibrated ODE-fixed fit was already near-optimal
- the data mildly prefers `t_{osc} \simeq 1.54` over `1.5`, but this is a refinement, not a qualitative change

Full-time `s(v_w,\theta_0)` rescaling test:

- define `s(v_w,\theta_0)` and replace every `t_p` in the canonical ansatz by `s(v_w,\theta_0)\,t_p`
- keep ODE-fixed `f_\infty(\theta_0)`, free `c(v_w)`, shared `r`
- keep `t_{osc}=1.5` fixed in this test to avoid a near-degeneracy between a global `t_{osc}` and a pointwise time-rescaling field `s`

Result:

- this fit is much worse
  - relative RMSE `8.0036e-02`
  - `r = 1.2392`
  - `c(v_w) = {0.3: 1.5007, 0.5: 1.4007, 0.7: 1.2784, 0.9: 1.2459}`
- mean raw RMSE by `H_*`:
  - `H_*=1.0`: `8.0484e-02`
  - `H_*=1.5`: `3.8595e-02`
  - `H_*=2.0`: `5.2504e-02`

Pathology:

- the fitted `s(v_w,\theta_0)` grid hit the lower bound `10^{-4}` for `\theta_0 = 2.3562` across all `v_w`
- equivalently, `t_c^{equiv} = 1/s` ran to `10^4` there

Interpretation:

- applying the same local time-rescaling factor to both the prefactor `(t_p/t_{osc})^{3/2}` and the transition denominator overconstrains the model
- the fit then tries to turn the rescaling off at large `\theta_0` by driving `s` to its floor
- this is strong evidence that the canonical ansatz does **not** want a uniform pointwise time remapping of all `t_p` entries

Conclusion:

- keep the calibrated ODE-fixed family as the preferred multi-`v_w` collapse model
- if desired, allow a free global `t_{osc}`: the best-fit value is `t_{osc} \simeq 1.54`
- reject the "apply `s` to all `t_p`" variant in its current form

Key outputs:

- `results_lattice_theta_tc_tosc_and_s_tests_beta0_tcmax300/comparison_sheet.png`
- `results_lattice_theta_tc_tosc_and_s_tests_beta0_tcmax300/rmse_by_h.png`
- `results_lattice_theta_tc_tosc_and_s_tests_beta0_tcmax300/rmse_by_h_vw.png`
- `results_lattice_theta_tc_tosc_and_s_tests_beta0_tcmax300/calib_fixed_tosc/collapse_overlay.png`
- `results_lattice_theta_tc_tosc_and_s_tests_beta0_tcmax300/calib_free_tosc/collapse_overlay.png`
- `results_lattice_theta_tc_tosc_and_s_tests_beta0_tcmax300/s_alltp/collapse_overlay.png`
- `results_lattice_theta_tc_tosc_and_s_tests_beta0_tcmax300/calib_fixed_tosc/xi_vs_betaH_H2p0.png`
- `results_lattice_theta_tc_tosc_and_s_tests_beta0_tcmax300/calib_free_tosc/xi_vs_betaH_H2p0.png`
- `results_lattice_theta_tc_tosc_and_s_tests_beta0_tcmax300/s_alltp/xi_vs_betaH_H2p0.png`
- `results_lattice_theta_tc_tosc_and_s_tests_beta0_tcmax300/final_summary.json`

#### 15h. Boundary-consistent replacement of `c(v_w)` by `f_\infty(\theta_0,v_w)`

Motivation:

- the calibrated ODE-fixed model with free `c(v_w)` fits best, but violates the exact early-time condition

\[
\xi(t_p \to 0) \to 1
\]

because it gives `\xi \to c(v_w)` instead
- to keep the boundary condition exact while still allowing `v_w`-dependent late-time behavior, replace the scalar normalization by a direct fit of the plateau

\[
\tilde f
=
f_\infty(\theta_0,v_w)
+
\frac{F_0^{ODE}(\theta_0)}
{\left(\frac{x}{t_{osc}}\right)^{3/2}\left[1+\left(\frac{x}{t_c(\theta_0,v_w)}\right)^r\right]}
\]

with `c=1`

Tested variants:

1. fixed `t_{osc}=1.5`
2. free global `t_{osc}`

Benchmark for comparison:

- calibrated ODE-fixed model with free `c(v_w)` and free `t_{osc}`
- relative RMSE `1.9362e-02`
- `r = 2.0568`
- `t_{osc} = 1.5430`

Results for the new boundary-consistent model:

- fixed `t_{osc}`:
  - relative RMSE `2.6686e-02`
  - `r = 3.0059`
- free `t_{osc}`:
  - relative RMSE `2.6686e-02`
  - `r = 3.0058`
  - `t_{osc} = 1.5215`

Interpretation:

- allowing `f_\infty(\theta_0,v_w)` does improve substantially over the older `c=1`, `f_\infty(\theta_0)`-only model
- but it does **not** recover the fit quality of the calibrated benchmark
- freeing `t_{osc}` on top of `f_\infty(\theta_0,v_w)` is nearly irrelevant: the fixed/free `t_{osc}` runs are numerically almost identical

High-`H_*` behavior:

- benchmark mean raw RMSE by `H_*`:
  - `H_*=1.0`: `1.7586e-02`
  - `H_*=1.5`: `1.8184e-02`
  - `H_*=2.0`: `1.6516e-02`
- new `f_\infty(\theta_0,v_w)` model:
  - `H_*=1.0`: `2.4291e-02`
  - `H_*=1.5`: `1.7858e-02`
  - `H_*=2.0`: `2.7041e-02`

So:

- this boundary-consistent model improves the middle `H_*=1.5` slice a bit relative to its own `H_*=1.0` and `H_*=2.0`
- but it still leaves a visible high-`H_*` tension and is worse than the calibrated benchmark at `H_*=2.0`

Fitted plateau behavior:

- `f_\infty(\theta_0,v_w)` develops substantial `v_w` dependence
- for low and moderate `\theta_0`, the fitted plateau is generally above ODE for all `v_w`
- at larger `\theta_0`, the high-`v_w` plateaus can drop back toward or even below the ODE value, while low-`v_w` remains above it

Conclusion:

- if the early-time normalization must be exact, then fitting `f_\infty(\theta_0,v_w)` is a cleaner replacement for `c(v_w)` than multiplying the whole `\tilde f`
- but empirically it still does not match the collapse quality of the calibrated ODE-fixed benchmark
- in other words: the data mismatch absorbed by `c(v_w)` is not just a pure late-time plateau correction

Key outputs:

- `results_lattice_theta_tc_sharedr_free_finf_vwtheta_tests_beta0_tcmax300/comparison_sheet.png`
- `results_lattice_theta_tc_sharedr_free_finf_vwtheta_tests_beta0_tcmax300/rmse_by_h.png`
- `results_lattice_theta_tc_sharedr_free_finf_vwtheta_tests_beta0_tcmax300/vwtheta_fixed_tosc/collapse_overlay.png`
- `results_lattice_theta_tc_sharedr_free_finf_vwtheta_tests_beta0_tcmax300/vwtheta_free_tosc/collapse_overlay.png`
- `results_lattice_theta_tc_sharedr_free_finf_vwtheta_tests_beta0_tcmax300/vwtheta_fixed_tosc/f_infty_by_vw.png`
- `results_lattice_theta_tc_sharedr_free_finf_vwtheta_tests_beta0_tcmax300/vwtheta_free_tosc/f_infty_by_vw.png`
- `results_lattice_theta_tc_sharedr_free_finf_vwtheta_tests_beta0_tcmax300/vwtheta_fixed_tosc/xi_vs_betaH_H2p0.png`
- `results_lattice_theta_tc_sharedr_free_finf_vwtheta_tests_beta0_tcmax300/vwtheta_free_tosc/xi_vs_betaH_H2p0.png`
- `results_lattice_theta_tc_sharedr_free_finf_vwtheta_tests_beta0_tcmax300/final_summary.json`

#### 15i. Prefactor consistency update: use `(2/3)` in the canonical `t_{osc}` prefactor

To match the scale-factor interpretation,

\[
\left(\frac{a_p}{a_{\rm osc}}\right)^3
=
\left(\frac{2\,t_p}{3\,t_{\rm osc}}\right)^{3/2},
\]

the canonical prefactor was updated in the active ODE/lattice scripts from

\[
\left(\frac{t_p}{t_{\rm osc}}\right)^{3/2}
\]

to

\[
\left(\frac{2\,t_p}{3\,t_{\rm osc}}\right)^{3/2}.
\]

This was implemented in:

- `fit_ode_direct_transition_ansatz.py`
- `fit_lattice_fixed_ode_amplitudes_theta_tc_shared_r_by_vw.py`
- `fit_lattice_theta_tc_tosc_and_s_tests.py`
- `fit_lattice_theta_tc_shared_r_free_finf_vwtheta_tests.py`

and string conventions were aligned in the related canonical-`\tilde f` scripts as well.

Empirical effect:

- in fits where `t_{osc}` is free, this is almost entirely absorbed by the fitted `t_{osc}` and leaves fit quality essentially unchanged
- in the direct ODE fit, `t_c` and `r` remained essentially unchanged, while the fitted `F_\infty` values rescaled upward as expected

Updated direct ODE fit:

- `t_c = 50.0`
- `r = 0.5602`
- `F_\infty = [0.7158,\ 0.8011,\ 1.0110,\ 1.4681,\ 2.5723,\ 6.5449]`

Updated calibrated ODE-fixed lattice benchmark with free global `t_{osc}`:

- relative RMSE `1.9362e-02`
- `r = 2.0570`
- `t_{osc} = 1.5430`
- `c(v_w) = {0.3: 1.1342, 0.5: 1.0899, 0.7: 1.0101, 0.9: 0.9988}`

Updated boundary-consistent `f_\infty(\theta_0,v_w)` fit with free global `t_{osc}`:

- relative RMSE `2.6686e-02`
- `r = 3.0059`
- `t_{osc} = 1.5216`

Conclusion:

- the prefactor update is the correct convention and should be used going forward
- it does not change the qualitative model ranking
- best current empirical fit still: ODE-fixed `f_\infty(\theta_0)` with free `c(v_w)` and free `t_{osc}`
- best boundary-consistent alternative still: free `f_\infty(\theta_0,v_w)` with `c=1`, but it remains worse than the calibrated benchmark

---

## Next Reasonable Step

If we want a better semi-analytic closure, the next target is now split by use case:

1. for the ODE `H_* = 1` compact description, keep the 7-parameter `c0 + c1/t_p^p` model as the repo default unless a cleaner derivation appears
2. for the lattice comparison in the current canonical `\tilde f` shared-`r` family, do **not** prioritize a global `H_*^\gamma` amplitude correction: the dedicated test drove `\gamma \to 0`
3. if an explicit `H_*` correction is still needed, it likely has to be more structured than either a single global `H_*^\gamma` factor or a single `t_c \propto H_*^{\alpha_{tc}}` power
4. keep older lower-`v_w` scalar-`\gamma` results as provisional only, because they are not supported by the newer shared-`r`, free-`f_\infty`, canonical-`\tilde f` comparison
5. ~~for multi-`v_w` universality: fit additive shift~~ → **DONE via multiplicative pointwise shift pipeline** (section 14). Global rel_rmse reduced from 7.2% → 1.6% with formula `s = 1 + 0.606·[(0.9/vw)^0.281 − 1]·tₚ^{−1.325}`
6. **ODE-anchored approach** (section 15) reaches ~1.9% global RMSE with per-(θ₀,vw) tc and shared r; next question is whether a smooth parametric formula `tc(θ₀, vw)` can be found (e.g. `tc ∝ (1/vw)^α · g(θ₀)`), which would reduce the parameter count while maintaining accuracy.
7. The systematic ~20% discrepancy between lattice and ODE F_inf values needs investigation — could be a bubble nucleation geometry effect or a lattice-spacing artefact.
