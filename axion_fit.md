# Axion PT Fit Strategies

## 1. Purpose

This document describes the strategy used to construct analytic fits for the axion relic density obtained from ODE simulations during a phase transition.

The simulations provide the quantity

\[
\rho_{PT}(\theta_0,t_p)
\]

and we compare it to the standard misalignment result

\[
\rho_{noPT}(\theta_0).
\]

The key derived observable used for fitting is

\[
\xi(\theta_0,t_p)=\frac{\rho_{PT}}{\rho_{noPT}}.
\]

---

# 2. Expected Scaling

The dominant physical effect of a phase transition is delaying the onset of oscillations.

Since

\[
\rho \propto a^{-3}
\]

and

\[
a \propto t^{1/2}
\]

in radiation domination, we expect approximately

\[
\rho \propto t_p^{3/2}.
\]

Therefore a useful diagnostic variable is

\[
Y_1(\theta_0,t_p)=\frac{\xi(\theta_0,t_p)}{t_p^{3/2}}.
\]

If this quantity becomes approximately constant or depends only weakly on parameters, it means the main PT effect is simply the delay of oscillations.

---

# 3. Recovering the PT Anharmonic Factor

If we want to study how the phase transition modifies the anharmonic structure we define

\[
Y_2(\theta_0,t_p)=\frac{\xi(\theta_0,t_p) f_{anh}^{(0)}(\theta_0)}{t_p^{3/2}}.
\]

Here

\(f_{anh}^{(0)}\) is the known anharmonic factor from the no‑PT case.

This quantity approximately isolates the **PT‑modified anharmonic factor**.

---

# 4. Diagnostic Plots

Several plots help identify the correct functional structure.

### Plot 1

\[
\xi(\theta_0,t_p)
\]

vs

\(\theta_0\)

for several values of \(t_p\).

Purpose: observe how strongly PT modifies the hilltop behavior.

### Plot 2

\[
\frac{\xi}{t_p^{3/2}}
\]

vs

\(\theta_0\).

Purpose: determine whether the dominant PT scaling is simply the delay effect.

### Plot 3

\[
\frac{\xi f_{anh}^{(0)}}{t_p^{3/2}}
\]

vs

\(\theta_0\).

Purpose: reconstruct the PT anharmonic structure.

### Plot 4

Heatmap of

\[
\log\left(\frac{\xi}{t_p^{3/2}}\right)
\]

in the \((t_p,\theta_0)\) plane.

Purpose: determine whether dependence is mostly on one variable or mixed.

---

# 5. Candidate Functional Forms

The remaining dependence on \(\theta_0\) can be tested against several candidate shapes.

Let

\[
u = \theta_0/\pi.
\]

### Power‑law hilltop divergence

\[
(1-\nu^2)^{-\alpha}
\]

Simple empirical model.

### Logarithmic enhancement

\[
\left[\ln\left(\frac{e}{1-\nu^2}\right)\right]^\alpha
\]

Motivated by the logarithmic divergence of the cosine oscillator period.

### Pendulum‑motivated form

\[
\left[\ln\left(\frac{e}{\cos^2(\theta_0/2)}\right)\right]^\alpha
\]

This corresponds more closely to the natural variable

\[
k=\sin(\theta_0/2)
\]

appearing in the elliptic integrals of the undamped oscillator.

---

# 6. Factorized Residual Models

A simple possibility is

\[
\xi(\theta_0,t_p)=t_p^{3/2} f(\theta_0).
\]

More generally

\[
\xi(\theta_0,t_p)=t_p^{3/2} f(\theta_0) g(t_p).
\]

These models assume the dependence is approximately separable.

---

# 7. Effective Oscillation Time Model

A more physical model assumes the PT modifies the effective oscillation time

\[
t_{eff}(\theta_0,t_p).
\]

The abundance then scales as

\[
\rho_{PT} \propto t_{eff}^{3/2}.
\]

One possible ansatz is

\[
t_{eff}=(t_p^\nu+(\lambda t_{osc}^{(0)}(\theta_0))^\nu)^{1/\nu}.
\]

This smoothly interpolates between

- PT‑controlled oscillations
- intrinsic hilltop delay.

---

# 8. Fitting Workflow

The recommended fitting workflow is

1. Compute \(\xi=\rho_{PT}/\rho_{noPT}\).

2. Test leading scaling using

\[
Y_1=\xi/t_p^{3/2}.
\]

3. Examine residual \(\theta_0\) dependence.

4. Test candidate hilltop functions.

5. If separability fails, move to the effective‑time model.

6. Choose the simplest model that reproduces the simulation results.

---

# 9. Goal of the Fit

The final goal is a compact analytic model for

\[
\rho_{PT}(\theta_0,t_p)
\]

that

- reproduces the ODE simulation results
- remains physically interpretable
- can be used for parameter scans and cosmological studies.

