"""Validate the seasonal-model Gibbs sampler (writes logs/sampler_validation.txt).
(a) recovery of known parameters from a simulated series with missing hours;
(b) comparison with the statsmodels SARIMAX maximum-likelihood fit on a real
    calibration series, and of the one-step predictive means."""
import os, sys, warnings
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, HERE)
warnings.filterwarnings("ignore")
import numpy as np
import forecasting as fc, calibrate
from config import BASE
from statsmodels.tsa.statespace.sarimax import SARIMAX
out = []
def say(s=""): print(s, flush=True); out.append(str(s))
names = ["phi1", "phi2", "Phi", "s2w", "V"]

# (a) simulation recovery
true = dict(phi1=1.55, phi2=-0.61, Phi=-0.50, s2w=3.0, V=1.2)
rng = np.random.default_rng(11)
c = fc.ar_coefs(true["phi1"], true["phi2"], true["Phi"])
T = 2208 + 500
y = np.zeros(T); y[:fc.NLAG] = 150 + rng.normal(0, 2, fc.NLAG)
for t in range(fc.NLAG, T):
    y[t] = c @ y[t - fc.NLAG:t][::-1] + rng.normal(0, np.sqrt(true["s2w"]))
y = y[500:]; obs = y + rng.normal(0, np.sqrt(true["V"]), len(y))
mask = rng.uniform(size=len(y)) > 0.05
d = fc.fit(obs, mask, n_chains=2, n_burn=1000, n_keep=1000, seed=[5])
say("(a) recovery from a simulated series (T=2208, 5% missing), 2 chains x (1000+1000)")
for k in names:
    v = d[k].ravel(); lo, hi = np.percentile(v, [2.5, 97.5])
    say(f"  {k:5s} true {true[k]:7.3f}  post mean {v.mean():7.3f}  95% [{lo:7.3f}, {hi:7.3f}]  {'covered' if lo <= true[k] <= hi else 'NOT COVERED'}")

# (b) real calibration series vs SARIMAX MLE
from tests.longchain_check import calib_data
for E in (2017, 2023):
    cP, cM, _ = calib_data(E, BASE)
    cal = calibrate.calibrate(E)
    yo = np.where(cM, cP, np.nan)
    m = SARIMAX(yo, order=(2, 0, 0), seasonal_order=(1, 1, 0, 24), measurement_error=True, trend="n")
    r = m.fit(disp=False, maxiter=500)
    p = dict(zip(r.model.param_names, r.params)); se = dict(zip(r.model.param_names, r.bse))
    mle = dict(phi1=p["ar.L1"], phi2=p["ar.L2"], Phi=p["ar.S.L24"], s2w=p["sigma2"], V=p["var.measurement_error"])
    mse = dict(phi1=se["ar.L1"], phi2=se["ar.L2"], Phi=se["ar.S.L24"], s2w=se["sigma2"], V=se["var.measurement_error"])
    pp = cal["port_p"]
    say(f"\n(b) {E} (calibration summer {cal['C']}): posterior (200 retained draws) vs SARIMAX MLE")
    for k in names:
        say(f"  {k:5s} post {np.mean(pp[k]):8.3f} (sd {np.std(pp[k]):6.3f})   MLE {mle[k]:8.3f} (se {mse[k]:6.3f})   diff/se {abs(np.mean(pp[k]) - mle[k]) / max(mse[k], 1e-9):5.2f}")
    # one-step predictive means on the same series
    kf = fc.KF(pp); K, Sv, _ = kf.gains(cM, kf.steady_state())
    a0 = fc.init_state(cP[:fc.NLAG], kf.D)
    mo, _ = fc.filter_means(kf, K, a0, cP[:, None], cM)
    mo = mo[:, :, 0].mean(1)
    ms = r.predict()  # one-step-ahead predictions
    sl = slice(200, None)
    ok = cM[sl]
    e_o = (cP[sl] - mo[sl])[ok]; e_s = (cP[sl] - ms[sl])[ok]
    say(f"  one-step predictions: corr {np.corrcoef(mo[sl], ms[sl])[0, 1]:.4f}, mean |diff| {np.mean(np.abs(mo[sl] - ms[sl])):.3f} MW; "
        f"RMSE ours {np.sqrt(np.mean(e_o ** 2)):.3f} vs SARIMAX {np.sqrt(np.mean(e_s ** 2)):.3f} MW")
open(os.path.join(HERE, "logs", "sampler_validation.txt"), "w").write("\n".join(out) + "\n")
