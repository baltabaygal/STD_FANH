# Axion Dynamics During a First‑Order Phase Transition (Base Notes)

## 1. Goal of the Project

We study **axion field dynamics during a first‑order cosmological phase transition (PT)** and how the PT modifies the final axion relic abundance compared to the standard misalignment mechanism.

The objective is to construct a **simple analytic model** that reproduces results of numerical simulations while keeping the number of free parameters small and physically interpretable.

The project separates the physics into two components:

- **Delayed Misalignment (DM)** — homogeneous axion oscillations whose onset is delayed by the phase transition.
- **Bubble Misalignment (BM)** — inhomogeneous field excitations produced by bubble collisions.

At the current stage we focus mainly on **DM** and attempt to determine an accurate analytic fit for it using ODE simulations.

---

# 2. Axion Field Dynamics

## 2.1 Potential

The axion potential is

\[
V(\theta) = m^2 f_a^2 (1 - \cos\theta)
\]

with

\[
\theta = a/f_a
\]

being the dimensionless field.

The full cosine potential is used, because near the hilltop

\[
\theta_0 \sim \pi
\]

anharmonic effects become important.

---

# 3. Equation of Motion

The homogeneous axion obeys

\[
\ddot{\theta} + 3H(t)\dot{\theta} + m^2(t) \sin\theta = 0.
\]

During radiation domination

\[
H(t) = \frac{1}{2t}
\]

so the equation becomes

\[
\ddot{\theta} + \frac{3}{2t}\dot{\theta} + m^2(t)\sin\theta = 0.
\]

Initial conditions

\[
\theta(t_i) = \theta_0,
\qquad
\dot{\theta}(t_i) = 0.
\]

---

# 4. Standard Misalignment (No PT)

In the absence of a phase transition the relic density takes the schematic form

\[
\rho_{noPT}(\theta_0)
= V(\theta_0)\,f_{anh}^{(0)}(\theta_0)
\left(\frac{a_{osc}(\theta_0)}{a_{end}}\right)^3.
\]

Here

- \(V(\theta_0)=1-\cos\theta_0\)
- \(f_{anh}^{(0)}\) encodes anharmonic corrections
- \(a_{osc}\) is the scale factor when oscillations begin

Oscillations start roughly when

\[
3H \sim m.
\]

The anharmonic correction arises because the oscillation period of the cosine potential depends on amplitude.

For the undamped cosine oscillator

\[
T = \frac{4}{m} K(k),
\qquad k = \sin(\theta_{max}/2),
\]

where \(K\) is the complete elliptic integral. Near the hilltop the period diverges logarithmically.

---

# 5. Effect of a Phase Transition

During a first‑order phase transition the axion mass turns on around the **percolation time** \(t_p\).

The PT modifies the abundance in two ways:

1. It changes the **time when oscillations effectively start**.
2. It modifies the **damping history** of the field before oscillations begin.

The PT abundance can be written schematically as

\[
\rho_{PT}(\theta_0,t_p)
= V(\theta_0)\,f_{anh}^{PT}(\theta_0,t_p)
\left(\frac{a_p(t_p,\theta_0)}{a_{end}}\right)^3.
\]

---

# 6. Relation Between PT and No‑PT Results

Define

\[
\xi(\theta_0,t_p) = \frac{\rho_{PT}}{\rho_{noPT}}.
\]

Using the expressions above

\[
\xi =
\frac{f_{anh}^{PT}(\theta_0,t_p)}{f_{anh}^{(0)}(\theta_0)}
\left(
\frac{a_p(t_p,\theta_0)}{a_{osc}^{(0)}(\theta_0)}
\right)^3.
\]

Thus \(\xi\) measures how PT modifies

- anharmonic dynamics
- oscillation timing

relative to the standard misalignment case.

---

# 7. Expected Time Scaling

After oscillations begin the axion behaves as matter

\[
\rho \propto a^{-3}.
\]

During radiation domination

\[
a \propto t^{1/2}
\]

so

\[
\rho \propto t^{-3/2}.
\]

This implies that a delay in the onset of oscillations leads to an approximate scaling

\[
\rho \propto t_p^{3/2}.
\]

This scaling motivates many of the fitting strategies used later.

---

# 8. Semi‑Analytic Understanding of Damping

The axion system is equivalent to a **damped nonlinear pendulum**.

The energy of the undamped system is

\[
E = \frac12 \dot{\theta}^2 + m^2(1-\cos\theta).
\]

With damping

\[
\frac{dE}{dt} = -3H(t) \dot{\theta}^2.
\]

If damping is slow over one oscillation we can average over a cycle.

For the cosine oscillator

\[
\langle \dot{\theta}^2 \rangle = 4m^2k^2\frac{E(k)}{K(k)}
\]

where

- \(K(k)\) is the elliptic integral of the first kind
- \(E(k)\) is the elliptic integral of the second kind

This leads to an amplitude evolution equation

\[
\frac{dk}{dt} = -\Gamma(t)k\frac{E(k)}{K(k)}.
\]

Near the hilltop \(K(k)\to\infty\), so the amplitude evolves slowly. This explains the strong enhancement near \(\theta_0\sim\pi\).

---

# 9. Numerical Approach

To study the PT dynamics we perform **ODE simulations** of the homogeneous field.

Parameter scans include

- initial angle \(\theta_0\)
- percolation time \(t_p\)
- other PT parameters

From each simulation we extract the late‑time energy density and compute

\[
\xi = \rho_{PT}/\rho_{noPT}.
\]

These simulation results form the dataset used for fitting.

---

# 10. Current Objective

The current goal is to determine a **simple analytic expression** describing

\[
\rho_{PT}(\theta_0,t_p)
\]

or equivalently

\[
\xi(\theta_0,t_p)
\]

that reproduces the ODE simulation results while maintaining a clear physical interpretation.

