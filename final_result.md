# Final Result

## Target quantity

The fitted object is

```text
Y2(theta0, tp) = xi(theta0, tp) * f_anh_noPT(theta0) / tp^(3/2)
```

The PT anharmonic factor is then

```text
f_anh_PT(theta0, tp) = (1.5)^(3/2) * Y2(theta0, tp)
                     = 1.8371173070873836 * Y2(theta0, tp)
```

## Best physical compact model

Define

```text
h(theta0) = log( e / cos^2(theta0 / 2) )
```

Then use

```text
c0(theta0) = A0 * h(theta0)^alpha0

log c1(theta0) = B0 + B1*h(theta0) + B2*h(theta0)^2

p(theta0) = P0 + P1*c0(theta0)

Y2(theta0, tp) = c0(theta0) + c1(theta0) / tp^( p(theta0) )
```

Best-fit parameters:

```text
A0     =  0.54099478821
alpha0 =  0.14633867627

B0     = -2.0889465029
B1     =  0.93879408645
B2     = -0.064950991313

P0     =  2.7048990951
P1     = -1.8121025642
```

So the explicit model is

```text
h = log( e / cos^2(theta0 / 2) )

Y2(theta0, tp) =
    0.54099478821 * h^0.14633867627
    + exp(
        -2.0889465029
        + 0.93879408645*h
        - 0.064950991313*h^2
      ) / tp^(
        2.7048990951
        - 1.8121025642 * 0.54099478821 * h^0.14633867627
      )
```

And therefore

```text
f_anh_PT(theta0, tp) =
    1.8371173070873836 * Y2(theta0, tp)
```

## Interpretation

- `c0(theta0)` is the large-`tp` plateau.
- The second term is the transient correction that dies away as `tp` grows.
- The plateau is controlled by the pendulum-like hilltop variable `h(theta0)`.
- The transient amplitude also follows the same hilltop variable.
- The transient exponent is not constant; it depends on `theta0` through `c0(theta0)`.

This matches the picture we discussed:

- at large `tp`, `f_anh` is almost `tp` independent
- at smaller `tp`, friction/release effects matter
- the transient is stronger for larger `theta0`

## Fit quality

For the current corrected `H*=1` table:

- global relative RMSE: `5.3297120782e-03`
- global log RMSE: `5.3260319608e-03`
- median absolute relative error: `4.2700907982e-03`
- 95th percentile absolute relative error: `9.7236135518e-03`
- max absolute relative error: `1.6421001256e-02`

This is the best physically clean compact model found so far.

## Best fixed-`tp` slice benchmark

If we allow one independent angular fit at each `tp`, the best slice-by-slice benchmark is

```text
h(theta0) = log( e / cos^2(theta0 / 2) )

Y2(theta0, tp) = c0(tp) + A(tp) * h(theta0)^alpha(tp)
```

Its global relative RMSE on the sampled grid is

- `2.5100035751e-03`

This is better than the compact 7-parameter model above, but it uses one triplet
`{c0(tp), A(tp), alpha(tp)}` for every sampled `tp`, so it is not a compact final law.

## Stability of the slice benchmark

I checked the two viable `+ c0(tp)` slice families using leave-one-`theta0`-out tests:

- `log_hilltop + c0(tp)`: LOO holdout relative RMSE `1.4056849602e-02`, mean jackknife curve spread `3.0905030933e-03`
- `pendulum_log + c0(tp)`: LOO holdout relative RMSE `1.0210903348e-02`, mean jackknife curve spread `2.2924155812e-03`

The `pendulum_log + c0(tp)` fit is also smoother in its fitted parameter curves, so it is the more stable fixed-`tp` ansatz numerically.

## Where to see the fit

Main comparison plots:

- [refine_y2_physical_models_vs_tp_dm_tp_fitready_H1p000.png](/home/gala/Desktop/AXION_PT/CODES/STD_FANH/ode/analysis/results/refine_y2_physical_models_vs_tp_dm_tp_fitready_H1p000.png)
- [refine_y2_physical_models_vs_theta_dm_tp_fitready_H1p000.png](/home/gala/Desktop/AXION_PT/CODES/STD_FANH/ode/analysis/results/refine_y2_physical_models_vs_theta_dm_tp_fitready_H1p000.png)

Summary tables:

- [refine_y2_physical_models_dm_tp_fitready_H1p000.txt](/home/gala/Desktop/AXION_PT/CODES/STD_FANH/ode/analysis/results/refine_y2_physical_models_dm_tp_fitready_H1p000.txt)
- [final_fit_summary_dm_tp_fitready_H1p000.txt](/home/gala/Desktop/AXION_PT/CODES/STD_FANH/ode/analysis/results/final_fit_summary_dm_tp_fitready_H1p000.txt)
- [test_y2_tp_shape_ansatz_dm_tp_fitready_H1p000.txt](/home/gala/Desktop/AXION_PT/CODES/STD_FANH/ode/analysis/results/test_y2_tp_shape_ansatz_dm_tp_fitready_H1p000.txt)
- [check_y2_tp_shape_stability_dm_tp_fitready_H1p000.txt](/home/gala/Desktop/AXION_PT/CODES/STD_FANH/ode/analysis/results/check_y2_tp_shape_stability_dm_tp_fitready_H1p000.txt)

Fixed-`tp` benchmark plots:

- [test_y2_tp_shape_ansatz_log_hilltop_plus_const_vs_tp_dm_tp_fitready_H1p000.png](/home/gala/Desktop/AXION_PT/CODES/STD_FANH/ode/analysis/results/test_y2_tp_shape_ansatz_log_hilltop_plus_const_vs_tp_dm_tp_fitready_H1p000.png)
- [test_y2_tp_shape_ansatz_pendulum_log_plus_const_vs_tp_dm_tp_fitready_H1p000.png](/home/gala/Desktop/AXION_PT/CODES/STD_FANH/ode/analysis/results/test_y2_tp_shape_ansatz_pendulum_log_plus_const_vs_tp_dm_tp_fitready_H1p000.png)
- [check_y2_tp_shape_stability_dm_tp_fitready_H1p000.png](/home/gala/Desktop/AXION_PT/CODES/STD_FANH/ode/analysis/results/check_y2_tp_shape_stability_dm_tp_fitready_H1p000.png)

Model-search plots:

- [search_y2_final_models_tradeoff_dm_tp_fitready_H1p000.png](/home/gala/Desktop/AXION_PT/CODES/STD_FANH/ode/analysis/results/search_y2_final_models_tradeoff_dm_tp_fitready_H1p000.png)
- [search_y2_final_models_best_params_dm_tp_fitready_H1p000.png](/home/gala/Desktop/AXION_PT/CODES/STD_FANH/ode/analysis/results/search_y2_final_models_best_params_dm_tp_fitready_H1p000.png)

## Benchmark model

There is a slightly better but less physical 8-parameter empirical fit, and there is also the even better but high-dimensional fixed-`tp` slice benchmark described above. I am not using either as the final result because the 7-parameter model above is cleaner physically and already accurate at the `5e-3` level.
