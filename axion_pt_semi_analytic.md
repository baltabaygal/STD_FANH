# Semi-analytic derivation: anharmonic factor \(f_{\rm anh}(\theta_0,t_p)\)

This note collects step-by-step derivations (semi-analytic) that connect the homogeneous axion ODE to the empirical fit

\[\
f_{\rm anh}(\theta_0,t_p) = \frac{a(t_p)}{(1-u^2)^{\alpha(t_p)}},\qquad u\equiv\frac{\theta_0}{\pi}.\
\]

and then explains methods to solve the driven nonlinear pendulum with *time-dependent viscous drag* that arises in our cosmological normalization.

## Update

The raw slice ansatz

\[
f_{\rm anh}(\theta_0,t_p) \sim \frac{a(t_p)}{(1-u^2)^{\alpha(t_p)}}
\]

should now be treated as a historical starting point, not the current best fit.

What the repo now finds is:

- the raw power family is not good enough on the current `Y2` table
- the best fixed-`t_p` slice benchmark is

  \[
  Y_2(\theta_0,t_p) = c_0(t_p) + A(t_p)\left[\log\!\left(\frac{e}{\cos^2(\theta_0/2)}\right)\right]^{\alpha(t_p)}
  \]

- the best compact global model is instead a plateau-plus-powerlaw form in `t_p`

\[
Y_2(\theta_0,t_p) = c_0(\theta_0) + \frac{c_1(\theta_0)}{t_p^{p(\theta_0)}}.
\]

So this note is still useful as motivation for the hilltop control variable and the first-oscillation picture, but it no longer represents the final fit formula used in the repo.

---

## 1. Starting point: equation of motion and energy

Dimensionless homogeneous axion ODE (mass on after \(t_p\), units \(M_\phi=1\)):

\[
\ddot\theta + \frac{3}{2t}\dot\theta + \sin\theta = 0, \quad t\ge t_p,
\]

with initial conditions at turn-on

\[
\theta(t_p)=\theta_0,\qquad \dot\theta(t_p)=0.
\]

Define the instantaneous mechanical energy per unit ``mass''

\[
E(t) \equiv \tfrac12\dot\theta^2 + (1-\cos\theta) = \tfrac12\dot\theta^2 + V(\theta).
\]

Differentiate to get the energy-loss equation:

\[
\frac{dE}{dt} = -\frac{3}{2t}\dot\theta^2.\tag{1}
\]

This identity is the key: losses are controlled by the kinetic energy and the damping coefficient \(\gamma(t)=3/(2t)\).

---

## 2. Instantaneous nonlinear frequency and hilltop expansion

Define the nonlinear oscillation period about an amplitude \(\theta_m\) (or initial angle \(\theta_0\) when released from rest):

\[
T(\theta_0) = 4K(k),\qquad k=\sin\frac{\theta_0}{2},
\]

where \(K(k)\) is the complete elliptic integral of the first kind. The corresponding frequency is

\[
\omega(\theta_0) = \frac{\pi}{2K(k)}.
\]

Limits of interest:

- Small angle: \(\theta_0\ll1\) gives \(\omega\approx 1\).
- Near hilltop: write \(u\equiv\theta_0/\pi\to1\). Use the standard elliptic asymptotic

  \[
  K(k) \approx \tfrac12\ln\frac{16}{1-k^2} \quad( k\to1^- ).
  \]

  With \(k=\sin(\theta_0/2)\) one finds for \(u\to1\)

  \[
  \omega(\theta_0) \approx \frac{\pi}{\ln\big(\dfrac{B}{1-u^2}\big)},\qquad B\sim 16.\tag{2}
  \]

The logarithmic vanishing of \(\omega\) near the separatrix (hilltop) is the origin of strong sensitivity to \(\theta_0\).

---

## 3. Dimensionless damping parameter at turn-on

Evaluate the damping rate at turn-on

\[
\gamma(t_p) = \frac{3}{2t_p}.
\]

Define the dimensionless ratio (control parameter)

\[
\Lambda(\theta_0,t_p) \equiv \frac{\gamma(t_p)}{\omega(\theta_0)} = \frac{3}{2t_p\,\omega(\theta_0)}.\tag{3}
\]

Physically, \(\Lambda\) is the fractional energy loss per oscillation (order of magnitude). The hypothesis is that the late-time anharmonic factor depends primarily on \(\Lambda\):

\[
f_{\rm anh}(\theta_0,t_p) = \mathcal{F}\big(\Lambda(\theta_0,t_p)\big).\tag{4}
\]

---

## 4. Minimal model for \(\mathcal{F}(\Lambda)\)

A parsimonious, physically motivated form capturing monotonic approach to limits is an exponential relaxation:

\[
\mathcal{F}(\Lambda) \simeq \mathcal{F}_\infty + (\mathcal{F}_0-\mathcal{F}_\infty)\,e^{-\kappa\Lambda},\tag{5}
\]

where:
- \(\mathcal{F}_0\) is the large-damping (early turn-on) limit (in your normalization this maps to the no-PT reference after accounting for \(t^{3/2}\) factors),
- \(\mathcal{F}_\infty\) is the underdamped (large \(t_p\)) limit, \(~\mathcal{O}(1)\),
- \(\kappa\) is an efficiency constant (order one) representing how quickly energy is lost as \(\Lambda\) increases.

Equation (5) is minimal and suffices to expose the angular dependence induced by the hilltop logarithm.

---

## 5. From \(\mathcal{F}(\Lambda)\) to the empirical power-law in \((1-u^2)^{-\alpha}\)

Use the hilltop approximation (2) to substitute into (3) and (5). We get

\[
\kappa\Lambda \approx \frac{\kappa 3}{2t_p}\frac{\ln\big(\tfrac{B}{1-u^2}\big)}{\pi}.
\]

Then the exponential term becomes

\[
e^{-\kappa\Lambda} = B^{-\frac{\kappa3}{2\pi t_p}}\;(1-u^2)^{\frac{\kappa3}{2\pi t_p}}.
\]

Thus

\[
\mathcal{F}(\Lambda)\approx \mathcal{F}_\infty + (\mathcal{F}_0-\mathcal{F}_\infty)\;B^{-\frac{\kappa3}{2\pi t_p}}\;(1-u^2)^{\frac{\kappa3}{2\pi t_p}}.\tag{6}
\]

If the second term dominates the \(u\)-dependence for the regime of interest (typical in intermediate \(t_p\) regime), then the angular dependence is well approximated by a power-law in \((1-u^2)\). Define

\[
\alpha_{\rm eff}(t_p) \equiv -\frac{\kappa 3}{2\pi t_p}\quad\text{(sign depends on empirical convention)}.
\]

So, up to multiplicative constants, one recovers the empirical structure

\[
f_{\rm anh}(\theta_0,t_p)\approx \frac{a(t_p)}{(1-u^2)^{\alpha(t_p)}},\qquad \alpha(t_p) \propto \frac{1}{t_p}.\tag{7}
\]

The proportionality constant is \(\kappa 3/(2\pi)\), which must be calibrated to ODE results.

---

## 6. Semi-analytic fit families (recommended templates)

**Exponentially relaxing template (physically motivated)**

\[
a(t_p) = a_\infty + (a_0 - a_\infty)\exp\Big[-\frac{\kappa_a 3}{2t_p}\Big],\qquad
\alpha(t_p) = \alpha_\infty + \frac{\kappa_\alpha}{t_p} \quad(\text{or saturated: }\alpha_\infty+\frac{\alpha_0-\alpha_\infty}{1+t_p/t_c}).
\]

**Rational saturating template (robust numeric fit)**

\[
a(t_p) = a_\infty + \frac{a_0-a_\infty}{1 + (t_p/t_a)^\gamma},\qquad
\alpha(t_p) = \alpha_\infty + \frac{\alpha_0-\alpha_\infty}{1 + (t_p/t_c)}.
\]

Choose initial guesses: \(t_c,t_a\sim\mathcal{O}(t_{\rm osc})\), \(a_0,a_\infty\sim O(1)\), \(\alpha_0\gtrsim\alpha_\infty\ge0\).

---

## 7. Solving the driven nonlinear pendulum with time-dependent viscous drag

The full problem: mass turns on at \(t_p\) and thereafter we have a time-dependent damping coefficient:

\[
\ddot\theta + \gamma(t)\dot\theta + \sin\theta = 0,\qquad \gamma(t)=\frac{3}{2t}.
\]

**Solution approaches and when they apply**

1. **Direct numerical integration (baseline)**
   - Use adaptive, high-accuracy ODE integrators (Dormand-Prince, DOP853) with max-step control adjusted near hilltop. Track energy (Ea^3) and use `late_time_Ea3` for plateau estimation.
   - Pros: robust, exact within numerics. Cons: expensive for large parameter scans.

2. **First-oscillation energy loss approximation (semi-analytic)**
   - Estimate energy right after turn-on: \(E(t_p)\approx V(\theta_0)\).
   - Compute fractional energy loss during first oscillation using (approximate) formula
     \(\Delta E/E \sim \dfrac{\gamma(t_p)}{\omega(\theta_0)}\).  Use one-step discrete reduction: \(E_{1}\approx E_0\exp(-C\Lambda)\) with \(C\sim O(1)\).
   - Iterate or average if multiple oscillations matter. This underpins the exponential \(\mathcal{F}(\Lambda)\) model.

3. **Method of averaging / adiabatic invariant**
   - For slowly varying \(\gamma(t)\) and slowly varying amplitude (weak damping) one can use the averaged evolution of action \(J\) (adiabatic invariant). For a nonlinear oscillator,

     \[\frac{dJ}{dt} = -\langle \gamma(t) p^2 \rangle_{\text{period}},\]

     and then integrate the averaged equation for \(J(t)\) to estimate late-time energy.
   - Good when \(\Lambda\ll1\) (weak damping).

4. **Separatrix crossing / matched asymptotics for hilltop**
   - Near the separatrix (\(\theta_0\to\pi\)) the motion is slow; matched asymptotics (inner region close to separatrix + outer oscillatory region) give scalings for energy loss that produce the power-law dependence in \((1-u^2)\).
   - Requires careful treatment of time-dependent \(\gamma(t)\) and is nontrivial; use Melnikov-like estimates for dissipative separatrix crossing.

5. **WKB / multiple-scale expansion for smooth mass turn-on**
   - If mass turns on smoothly over finite time (not instantaneous), treat the restoring force as a time-dependent frequency and apply WKB-type adiabatic analysis to compute phase and amplitude evolution across the turn-on.

6. **Global fitting (semi-empirical)**
   - Combine the above: use first-oscillation/averaging to motivate fit family, then calibrate constants to ODE results. This is computationally optimal and physically transparent.

---

## 8. Practical algorithm to extract \(a(t_p)\) and \(\alpha(t_p)\) semi-analytically

1. For each \(t_p\) of interest compute \(\Lambda(\theta_0,t_p)\) over a grid of \(\theta_0\).
2. Using the first-oscillation model set

   \[E_{\rm after}\approx V(\theta_0)\exp(-\kappa\Lambda(\theta_0,t_p)).\]

   Convert to \(f_{\rm anh}(\theta_0,t_p) = E_{\rm after}/(V(\theta_0) t_p^{3/2})\) and fit \(a,\alpha\) to the form \(a/(1-u^2)^\alpha\).
3. Use ODE to refine \(\kappa\) and \(\kappa_\alpha\) by matching numerics; iterate.

---

## 9. Summary & next steps

- The hilltop logarithm in \(\omega(\theta_0)\) combined with exponential-in-\(\Lambda\) energy loss yields an effective power-law angular dependence and predicts \(\alpha(t_p)\sim\text{const}/t_p\) in the intermediate regime.
- Use a hybrid strategy: (i) ab-initio ODE for calibrating constants, (ii) semi-analytic formulae above for physical interpretation and extrapolation.

If you want, I can now:

- derive the separatrix crossing estimate in more mathematical detail (matched asymptotics + Melnikov-type integral), or
- produce the analytic steps converting elliptic integral asymptotics to the explicit prefactor shown above (tracking constants such as \(B\) & \(\kappa\)).

Which do you prefer next?
