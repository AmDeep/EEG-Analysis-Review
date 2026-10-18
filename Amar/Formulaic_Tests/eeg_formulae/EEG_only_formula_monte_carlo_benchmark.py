"""EEG-only Monte Carlo scalar-formula stress test.
Synthetic data only: classes are assumed EEG patterns, not clinical pain profiles.
"""
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
FEATURES = ["alpha_suppression","beta_change","theta_change","gamma_change","peak_alpha_freq",
"frontal_alpha_asymmetry","midline_theta","theta_connectivity","alpha_connectivity","spectral_entropy",
"aperiodic_exponent","alpha_envelope_autocorr","alpha_burst_rate","erp_n2p2","erp_trial_variability","erp_habituation_slope"]
FORMULAE = {
"F01_alpha_only":["alpha_suppression"],
"F02_alpha_beta_theta":["alpha_suppression","beta_change","theta_change"],
"F03_spectral":["alpha_suppression","beta_change","theta_change","gamma_change","peak_alpha_freq"],
"F04_spectral_entropy":["alpha_suppression","beta_change","theta_change","spectral_entropy"],
"F05_connectivity":["theta_connectivity","alpha_connectivity","midline_theta"],
"F06_temporal":["alpha_envelope_autocorr","alpha_burst_rate","erp_trial_variability","erp_habituation_slope"],
"F07_ERP":["erp_n2p2","erp_trial_variability","erp_habituation_slope"],
"F08_spectral_connectivity":["alpha_suppression","beta_change","theta_change","theta_connectivity","alpha_connectivity"],
"F09_spectral_complexity":["alpha_suppression","beta_change","theta_change","gamma_change","spectral_entropy","aperiodic_exponent"],
"F10_multidomain":["alpha_suppression","beta_change","theta_change","spectral_entropy","theta_connectivity","erp_trial_variability"],
"F11_all_EEG_features":FEATURES,
"F12_reduced_multidomain":["alpha_suppression","theta_change","spectral_entropy","theta_connectivity","alpha_envelope_autocorr","erp_n2p2"]}
# This is a simplified reproducibility scaffold; consult the accompanying report and workbook
# for the full Monte Carlo results and assumptions.
rng=np.random.default_rng(20261016)
X=StandardScaler().fit_transform(rng.normal(size=(210,len(FEATURES))))
for name, cols in FORMULAE.items():
    idx=[FEATURES.index(c) for c in cols]
    Xi=X[:,idx]
    if Xi.shape[1] == 1: score=Xi[:,0]; weights=np.array([1.])
    else:
        pc=PCA(n_components=1).fit(Xi); weights=pc.components_[0].copy()
        if weights[np.argmax(np.abs(weights))] < 0: weights=-weights
        score=Xi@weights
    print(name, "features",len(cols),"scalar SD",round(float(score.std()),3))
