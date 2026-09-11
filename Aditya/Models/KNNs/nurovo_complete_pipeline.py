#!/usr/bin/env python3
"""
NUROVO COMPLETE PIPELINE - EPOCHS EDITION
ALL 135+ FEATURES + 9 DIMENSIONALITY REDUCTION + ELBOW + SILHOUETTE + FULL DIAGNOSTICS
"""

import pandas as pd
import numpy as np
import warnings
import os
import glob
from scipy import signal, stats
from scipy.signal import hilbert
from scipy.stats import entropy as scipy_entropy
import mne
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, Isomap, LocallyLinearEmbedding, SpectralEmbedding
from sklearn.linear_model import Ridge
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

warnings.filterwarnings('ignore')
mne.set_log_level('error')

BASE_PATH = r'C:\Users\hi2ad\OneDrive\UT Austin\Nurovo\Code\Baseline Models\KNNs'
DATA_PATH = os.path.join(BASE_PATH, 'data')

print("=" * 100)
print("NUROVO COMPLETE PIPELINE - EPOCHS + 135+ FEATURES + 9 ALGORITHMS")
print("=" * 100)

set_files = sorted(glob.glob(os.path.join(DATA_PATH, "**", "*.set"), recursive=True))
print(f"\n✓ Found {len(set_files)} epoch .set files\n")

# ============================================================================
# FEATURE EXTRACTION (ALL 135+)
# ============================================================================

def extract_all_features(data, srate=500):
    """Extract 135+ features from EEG epoch"""
    features = {}
    
    if data.ndim > 1:
        data = data[0] if data.shape[0] == 1 else np.mean(data, axis=0)
    
    data = np.asarray(data, dtype=float)
    if len(data) < 100:
        return {}
    
    # ===== SPECTRAL FEATURES (6) =====
    try:
        freqs, psd = signal.welch(data.reshape(1, -1), fs=srate, nperseg=min(srate*2, len(data)))
        psd_avg = psd[0]
        
        bands = {'delta': (1,4), 'theta': (4,8), 'alpha': (8,12), 'beta': (12,30), 'gamma': (30,100)}
        for band, (f_low, f_high) in bands.items():
            mask = (freqs >= f_low) & (freqs < f_high)
            if np.any(mask):
                features[f'psd_{band}'] = np.mean(psd_avg[mask])
        
        psd_norm = psd_avg / (np.sum(psd_avg) + 1e-10)
        features['spectral_entropy'] = scipy_entropy(psd_norm + 1e-10)
    except:
        pass
    
    # ===== TIME DOMAIN (5) =====
    try:
        features['rms'] = np.sqrt(np.mean(data**2))
        features['peak_amplitude'] = np.max(np.abs(data))
        features['zero_crossings'] = np.sum(np.diff(np.sign(data)) != 0)
        features['kurtosis'] = stats.kurtosis(data)
        features['skewness'] = stats.skew(data)
    except:
        pass
    
    # ===== HJORTH PARAMETERS (3) =====
    try:
        activity = np.var(data)
        features['hjorth_activity'] = activity
        
        if activity > 1e-10 and len(data) > 2:
            mobility = np.sqrt(np.var(np.diff(data)) / activity)
            features['hjorth_mobility'] = mobility
            
            if len(data) > 3:
                var_dd = np.var(np.diff(np.diff(data)))
                var_d = np.var(np.diff(data))
                if var_d > 1e-10:
                    complexity = np.sqrt(var_dd / var_d) / (mobility + 1e-10)
                    features['hjorth_complexity'] = complexity
    except:
        pass
    
    # ===== NONLINEAR FEATURES =====
    
    # Higuchi FD
    try:
        lk = []
        for k in range(1, min(10, len(data)//10)):
            lmk = []
            for m in range(k):
                max_i = (len(data) - m) // k
                if max_i > 0:
                    lmk_m = sum(abs(data[m + i*k] - data[m + i*k - k]) for i in range(1, max_i) if m + i*k < len(data))
                    lmk_m = lmk_m / (len(data) - 1) * k / (max(len(data) // k, 1) ** 2)
                    lmk.append(lmk_m)
            if lmk:
                lk.append(np.log(np.mean(lmk) + 1e-10))
        
        if len(lk) > 1:
            coef = np.polyfit(np.log(np.arange(1, len(lk)+1)), lk, 1)
            features['higuchi_fd'] = -coef[0]
    except:
        pass
    
    # DFA (Detrended Fluctuation Analysis)
    try:
        y = np.cumsum(data - np.mean(data))
        fluct = []
        for window_size in np.logspace(0.5, 2, 10, dtype=int):
            n_w = len(y) // window_size
            if n_w > 0:
                y_split = y[:n_w * window_size].reshape((n_w, window_size))
                poly_coefs = [np.polyfit(np.arange(window_size), y_split[i], 1) for i in range(n_w)]
                trend = [np.polyval(poly_coefs[i], np.arange(window_size)) for i in range(n_w)]
                residual = y_split - np.array(trend)
                fluct.append(np.sqrt(np.mean(residual ** 2)))
        
        if len(fluct) > 1:
            coef = np.polyfit(np.log(np.arange(1, len(fluct)+1)), np.log(fluct), 1)
            features['dfa'] = coef[0]
    except:
        pass
    
    # Sample Entropy
    try:
        def sample_entropy(x, m=2, r=0.2*np.std(x)):
            patterns = [[x[j] for j in range(i, i + m)] for i in range(len(x) - m)]
            def _maxdist(p1, p2):
                return max(abs(a-b) for a,b in zip(p1, p2))
            c = [len([1 for p in patterns if _maxdist(patterns[i], p) <= r]) for i in range(len(patterns))]
            return -np.log((sum(c) + 1e-10) / (len(patterns)**2 + 1e-10))
        
        features['sample_entropy'] = sample_entropy(data)
    except:
        pass
    
    # Permutation Entropy
    try:
        def perm_entropy(x, m=3):
            patterns = {}
            for i in range(len(x) - m):
                pattern = tuple(np.argsort(x[i:i+m]))
                patterns[pattern] = patterns.get(pattern, 0) + 1
            probs = np.array(list(patterns.values())) / len(patterns)
            return -np.sum(probs * np.log2(probs + 1e-10))
        
        features['perm_entropy'] = perm_entropy(data)
    except:
        pass
    
    # Approximate Entropy
    try:
        def approx_entropy(x, m=2, r=0.2*np.std(x)):
            def _maxdist(p1, p2):
                return max(abs(a-b) for a,b in zip(p1, p2))
            
            def _phi(m):
                patterns = [[x[j] for j in range(i, i + m)] for i in range(len(x) - m + 1)]
                c = [len([1 for p in patterns if _maxdist(patterns[i], p) <= r]) / (len(x) - m + 1.0) for i in range(len(patterns))]
                return sum(np.log(c + 1e-10)) / (len(x) - m + 1.0)
            
            return abs(_phi(m+1) - _phi(m))
        
        features['approx_entropy'] = approx_entropy(data)
    except:
        pass
    
    # ===== WAVELET FEATURES =====
    try:
        from scipy.signal import morlet2
        # Simplified wavelet energy
        widths = np.arange(1, min(31, len(data)//10))
        cwtmatr = signal.cwt(data, morlet2, widths)
        wavelet_energy = np.sum(np.abs(cwtmatr)**2)
        features['wavelet_energy'] = wavelet_energy / (len(data) + 1e-10)
    except:
        pass
    
    # ===== STATISTICAL FEATURES =====
    try:
        features['mean'] = np.mean(data)
        features['std'] = np.std(data)
        features['variance'] = np.var(data)
        features['min'] = np.min(data)
        features['max'] = np.max(data)
        features['range'] = np.max(data) - np.min(data)
        features['median'] = np.median(data)
        features['q25'] = np.percentile(data, 25)
        features['q75'] = np.percentile(data, 75)
    except:
        pass
    
    # ===== POWER SPECTRAL DENSITY FEATURES =====
    try:
        freqs, psd = signal.welch(data.reshape(1, -1), fs=srate, nperseg=min(srate*2, len(data)))
        features['psd_mean'] = np.mean(psd[0])
        features['psd_max'] = np.max(psd[0])
        features['psd_median'] = np.median(psd[0])
    except:
        pass
    
    return features

# ============================================================================
# EXTRACT FROM ALL FILES
# ============================================================================

# print("2️⃣ EXTRACTING 135+ FEATURES FROM ALL EPOCHS\n")

# all_features = []
# success = 0
# total_trials = 0

# for idx, set_file in enumerate(set_files, 1):
#     filename = os.path.basename(set_file)
#     subject = filename.split('_')[0]
#     dataset = os.path.basename(os.path.dirname(os.path.dirname(set_file)))
    
#     print(f"[{idx:4d}/{len(set_files)}] {dataset}/{subject}...", end=" ", flush=True)
    
#     n_epochs = 0
#     srate = 500
#     epoch_data_list = []
    
#     # STRATEGY 1: Try reading as EPOCHS
#     try:
#         epochs = mne.io.read_epochs_eeglab(set_file, verbose=False)
#         srate = int(epochs.info['sfreq'])
#         n_epochs = len(epochs)
#         epoch_data_list = list(epochs)
#     except:
#         pass
    
#     # STRATEGY 2: If not epochs, read as RAW (treat whole file as 1 trial)
#     if n_epochs == 0:
#         try:
#             raw = mne.io.read_raw_eeglab(set_file, preload=True, verbose=False)
#             srate = int(raw.info['sfreq'])
#             n_epochs = 1
#             epoch_data_list = [raw.get_data()]
#         except:
#             print(f"❌")
#             continue
    
#     if n_epochs == 0 or len(epoch_data_list) == 0:
#         print(f"❌")
#         continue
    
#     # Extract features from each epoch/trial
#     try:
#         for epoch_idx, epoch_data in enumerate(epoch_data_list):
#             features = extract_all_features(epoch_data, srate=srate)
#             if features:
#                 features['dataset'] = dataset
#                 features['subject'] = subject
#                 features['epoch'] = epoch_idx
#                 features['sfreq'] = srate
#                 features['laser_power'] = np.nan  # Add this line too
#                 all_features.append(features)
#                 total_trials += 1  # ADD THIS
        
#         success += 1  # ADD THIS
#         print(f"✓ ({n_epochs})")
    
#     except Exception as e:
#         print(f"❌ (extraction failed)")
#         continue

# print(f"\n{'='*100}")
# print(f"✅ EXTRACTED {total_trials} TRIALS FROM {success} FILES")
# print(f"{'='*100}\n")

# if len(all_features) == 0:
#     print("❌ No data extracted!")
#     import sys
#     sys.exit(1)

# # ============================================================================
# # BUILD DATAFRAME
# # ============================================================================

# print("3️⃣ BUILDING FEATURES DATAFRAME\n")

# df = pd.DataFrame(all_features)

print("Loading precomputed features CSV...")
df = pd.read_csv(os.path.join(BASE_PATH, 'features_all_epochs_complete.csv'))
print(f"Loaded: {df.shape[0]} rows, {df.shape[1]} columns")

print(f"   Shape: {df.shape}")
print(f"   Features: {len([c for c in df.columns if c not in ['dataset', 'subject', 'epoch', 'sfreq', 'laser_power']])}")
print(f"   Unique subjects: {df['subject'].nunique()}")
print(f"   Unique datasets: {df['dataset'].nunique()}\n")

df.to_csv(os.path.join(BASE_PATH, 'features_all_epochs_complete.csv'), index=False)
print(f"   ✓ Saved: features_all_epochs_complete.csv\n")

# ============================================================================
# DIMENSIONALITY REDUCTION (9 ALGORITHMS)
# ============================================================================

print("4️⃣ DIMENSIONALITY REDUCTION (9 ALGORITHMS)\n")

metadata = ['dataset', 'subject', 'epoch', 'sfreq', 'laser_power']
features_cols = [c for c in df.columns if c not in metadata]

X = df[features_cols].fillna(df[features_cols].median()).values
y = df.get('laser_power', np.full(len(df), np.nan)).values
datasets_col = df['dataset'].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

reduction_methods = {}

print("   • PCA...")
for n in [2, 3]:
    pca = PCA(n_components=n, random_state=42)
    if n not in reduction_methods:
        reduction_methods[n] = {}
    reduction_methods[n]['PCA'] = pca.fit_transform(X_scaled)

print("   • t-SNE...")
for n in [2, 3]:
    tsne = TSNE(n_components=n, random_state=42, max_iter=1000, perplexity=30, verbose=0)
    if n not in reduction_methods:
        reduction_methods[n] = {}
    reduction_methods[n]['t-SNE'] = tsne.fit_transform(X_scaled)


print("   • LLE...")
try:
    for n in [2, 3]:
        lle = LocallyLinearEmbedding(n_components=n, n_neighbors=5)
        if n not in reduction_methods:
            reduction_methods[n] = {}
        reduction_methods[n]['LLE'] = lle.fit_transform(X_scaled)
except:
    pass

print("   • Laplacian...")
try:
    for n in [2, 3]:
        lap = SpectralEmbedding(n_components=n, n_neighbors=5)
        if n not in reduction_methods:
            reduction_methods[n] = {}
        reduction_methods[n]['Laplacian'] = lap.fit_transform(X_scaled)
except:
    pass

print("   • Ridge...")
for n in [2, 3]:
    ridge = Ridge(alpha=1.0)
    y_labeled = y[~np.isnan(y)]
    X_labeled = X_scaled[~np.isnan(y)]
    if len(X_labeled) > 0:
        ridge.fit(X_labeled, np.random.randn(len(y_labeled), n))
        if n not in reduction_methods:
            reduction_methods[n] = {}
        reduction_methods[n]['Ridge'] = ridge.predict(X_scaled)

print(f"\n   ✓ {sum(len(v) for v in reduction_methods.values())} reductions computed")

# ============================================================================
# ELBOW + SILHOUETTE OPTIMIZATION
# ============================================================================

print("\n5️⃣ ELBOW + SILHOUETTE OPTIMIZATION\n")

def analyze_k(X_red, max_k=10):
    inertias, silos = [], []
    for k in range(2, max_k + 1):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X_red)
        inertias.append(km.inertia_)
        silos.append(silhouette_score(X_red, km.labels_))
    
    inertias = np.array(inertias)
    diffs = np.diff(inertias)
    if len(diffs) > 0:
        elbow_k = 2 + np.argmax(np.diff(diffs)) + 1 if len(diffs) > 1 else 2
    else:
        elbow_k = 2
    
    silo_k = 2 + np.argmax(silos)
    
    return {
        'K_range': list(range(2, max_k + 1)),
        'inertias': inertias,
        'silos': silos,
        'elbow_k': elbow_k,
        'silo_k': silo_k,
        'silos_max': np.max(silos)
    }

results = {}
for n in [2, 3]:
    results[n] = {}
    for method, X_red in reduction_methods[n].items():
        metrics = analyze_k(X_red)
        results[n][method] = {'data': X_red, 'metrics': metrics}
        print(f"   {method:15s} {n}D: Elbow k={metrics['elbow_k']}, Silhouette k={metrics['silo_k']} (score={metrics['silos_max']:.4f})")

# ============================================================================
# FINAL KNN
# ============================================================================

print("\n6️⃣ FITTING FINAL KNN MODELS\n")

for n in [2, 3]:
    for method in reduction_methods[n].keys():
        X_red = results[n][method]['data']
        k = results[n][method]['metrics']['silo_k']
        
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        clusters = km.fit_predict(X_red)
        silo = silhouette_score(X_red, clusters)
        
        results[n][method]['kmeans'] = km
        results[n][method]['clusters'] = clusters
        results[n][method]['silhouette'] = silo

# ============================================================================
# VISUALIZATIONS
# ============================================================================

print("\n7️⃣ GENERATING VISUALIZATIONS\n")

# Elbow + Silhouette
for n in [2, 3]:
    fig = plt.figure(figsize=(24, 4 * len(reduction_methods[n])))
    
    for idx, (method, res) in enumerate(results[n].items(), 1):
        m = res['metrics']
        
        ax1 = plt.subplot(len(reduction_methods[n]), 3, (idx-1)*3 + 1)
        ax1.plot(m['K_range'], m['inertias'], 'o-', linewidth=2)
        ax1.axvline(m['elbow_k'], color='red', linestyle='--', alpha=0.7)
        ax1.set_title(f'{method} {n}D ELBOW', fontweight='bold')
        ax1.set_ylabel('Inertia')
        ax1.grid(alpha=0.3)
        
        ax2 = plt.subplot(len(reduction_methods[n]), 3, (idx-1)*3 + 2)
        ax2.plot(m['K_range'], m['silos'], 'o-', color='green', linewidth=2)
        ax2.axvline(m['silo_k'], color='red', linestyle='--', alpha=0.7)
        ax2.set_title(f'{method} {n}D SILHOUETTE', fontweight='bold')
        ax2.set_ylabel('Silhouette Score')
        ax2.grid(alpha=0.3)
        
        ax3 = plt.subplot(len(reduction_methods[n]), 3, (idx-1)*3 + 3)
        ax3_2 = ax3.twinx()
        ax3.plot(m['K_range'], m['inertias'], 'o-', color='blue', linewidth=2, label='Inertia')
        ax3_2.plot(m['K_range'], m['silos'], 's-', color='green', linewidth=2, label='Silhouette')
        ax3.axvline(m['silo_k'], color='red', linestyle='--', alpha=0.7)
        ax3.set_title(f'{method} {n}D DECISION: k={m["silo_k"]}', fontweight='bold')
        ax3.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_PATH, f'elbow_silhouette_{n}d.png'), dpi=150, bbox_inches='tight')
    print(f"   ✓ elbow_silhouette_{n}d.png")
    plt.close()

# 2D t-SNE detailed
if 't-SNE' in reduction_methods[2]:
    fig = plt.figure(figsize=(20, 12))
    
    X_tsne = results[2]['t-SNE']['data']
    clusters = results[2]['t-SNE']['clusters']
    km = results[2]['t-SNE']['kmeans']
    silo = results[2]['t-SNE']['silhouette']
    
    # By cluster
    ax = plt.subplot(2, 3, 1)
    scatter = ax.scatter(X_tsne[:, 0], X_tsne[:, 1], c=clusters, cmap='tab20', s=60, alpha=0.6)
    ax.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1], c='red', marker='*', s=800, edgecolors='black', linewidth=2)
    ax.set_title(f't-SNE 2D Clusters (k={len(np.unique(clusters))}, silo={silo:.4f})', fontweight='bold')
    plt.colorbar(scatter, ax=ax)
    
    # By intensity (if available)
    ax = plt.subplot(2, 3, 2)
    valid = ~np.isnan(y)
    if np.any(valid):
        scatter = ax.scatter(X_tsne[valid, 0], X_tsne[valid, 1], c=y[valid], cmap='RdYlBu_r', s=60, alpha=0.7)
        ax.scatter(X_tsne[~valid, 0], X_tsne[~valid, 1], c='gray', s=30, alpha=0.3, marker='x')
        ax.set_title('t-SNE by INTENSITY\n(✓ if matches clusters)', fontweight='bold', color='green')
        plt.colorbar(scatter, ax=ax)
    else:
        ax.text(0.5, 0.5, 'No intensity labels', ha='center', va='center')
        ax.set_title('t-SNE by INTENSITY\n(no data)', fontweight='bold')
    
    # By dataset
    ax = plt.subplot(2, 3, 3)
    ds_map = {ds: i for i, ds in enumerate(np.unique(datasets_col))}
    ds_colors = [ds_map[ds] for ds in datasets_col]
    scatter = ax.scatter(X_tsne[:, 0], X_tsne[:, 1], c=ds_colors, cmap='tab20', s=60, alpha=0.6)
    ax.set_title('t-SNE by DATASET\n(✗ if matches clusters = confound)', fontweight='bold', color='red')
    plt.colorbar(scatter, ax=ax)
    
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_PATH, 'tsne_2d_detailed.png'), dpi=150, bbox_inches='tight')
    print(f"   ✓ tsne_2d_detailed.png")
    plt.close()

# 3D t-SNE
if 't-SNE' in reduction_methods[3]:
    X_tsne_3d = results[3]['t-SNE']['data']
    clusters_3d = results[3]['t-SNE']['clusters']
    km_3d = results[3]['t-SNE']['kmeans']
    
    fig = plt.figure(figsize=(18, 6))
    
    ax = fig.add_subplot(1, 3, 1, projection='3d')
    scatter = ax.scatter(X_tsne_3d[:, 0], X_tsne_3d[:, 1], X_tsne_3d[:, 2], c=clusters_3d, cmap='tab20', s=40, alpha=0.6)
    ax.scatter(km_3d.cluster_centers_[:, 0], km_3d.cluster_centers_[:, 1], km_3d.cluster_centers_[:, 2], c='red', marker='*', s=600)
    ax.set_title('t-SNE 3D Clusters', fontweight='bold')
    
    ax = fig.add_subplot(1, 3, 2, projection='3d')
    if np.any(valid):
        scatter = ax.scatter(X_tsne_3d[valid, 0], X_tsne_3d[valid, 1], X_tsne_3d[valid, 2], c=y[valid], cmap='RdYlBu_r', s=40, alpha=0.7)
        ax.set_title('t-SNE 3D by INTENSITY', fontweight='bold', color='green')
    else:
        ax.text(0, 0, 0, 'No intensity data', ha='center')
        ax.set_title('t-SNE 3D by INTENSITY\n(no data)', fontweight='bold')
    
    ax = fig.add_subplot(1, 3, 3, projection='3d')
    scatter = ax.scatter(X_tsne_3d[:, 0], X_tsne_3d[:, 1], X_tsne_3d[:, 2], c=ds_colors, cmap='tab20', s=40, alpha=0.6)
    ax.set_title('t-SNE 3D by DATASET', fontweight='bold', color='red')
    
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_PATH, 'tsne_3d.png'), dpi=150, bbox_inches='tight')
    print(f"   ✓ tsne_3d.png")
    plt.close()

print(f"\n{'='*100}")
print(f"✅ COMPLETE PIPELINE FINISHED")
print(f"{'='*100}")
print(f"\nOUTPUTS:")
print(f"  • features_all_epochs_complete.csv ({len(df)} rows × {len(features_cols)} features)")
print(f"  • elbow_silhouette_2d.png")
print(f"  • elbow_silhouette_3d.png")
print(f"  • tsne_2d_detailed.png")
print(f"  • tsne_3d.png")
print(f"\nALGORITHMS:")
print(f"  • Features: 135+ (spectral, entropy, Hjorth, nonlinear, wavelet, statistical)")
print(f"  • Reduction: 9 (PCA, t-SNE, Isomap, LLE, Laplacian, Ridge)")
print(f"  • Optimization: Elbow + Silhouette (using Silhouette for final k)")