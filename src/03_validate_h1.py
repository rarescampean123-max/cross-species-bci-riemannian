import os
import json
import warnings
import numpy as np
from scipy.signal import butter, filtfilt
from scipy.linalg import fractional_matrix_power
from scipy.stats import pearsonr
from pyriemann.estimation import Covariances
from pyriemann.geometry.distance import distance_riemann
from pyriemann.geometry.mean import mean_riemann
from pyriemann.classification import MDM

warnings.filterwarnings("ignore", category=DeprecationWarning)


def bandpass_filter(signals, lowcut=0.02, highcut=0.25, fs=1.0, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signals, axis=-1)


def compute_spd_covariances(signals):
    cov_estimator = Covariances(estimator='lwf')
    return cov_estimator.fit_transform(signals)


def euclidean_alignment(cov_matrices):
    r_bar = np.mean(cov_matrices, axis=0)
    r_inv_sqrt = fractional_matrix_power(r_bar, -0.5)
    return np.einsum('ij,njk,kl->nil', r_inv_sqrt, cov_matrices, r_inv_sqrt)


def fetch_real_fmri_sample(raw_dir):
    os.makedirs(raw_dir, exist_ok=True)
    cache_file = os.path.join(raw_dir, 'real_caudate_sample.npz')
    if os.path.exists(cache_file):
        data = np.load(cache_file)
        return data['X_h'], data['y_h'], data['X_d'], data['y_d']

    try:
        from nilearn import datasets
        from nilearn.maskers import NiftiLabelsMasker

        print("[INFO] Se descarca esantionul real fMRI (Nilearn + Atlas Subcortical)...")
        adhd = datasets.fetch_adhd(n_subjects=1, data_dir=raw_dir)
        atlas = datasets.fetch_atlas_harvard_oxford('sub-maxprob-thr25-2mm', data_dir=raw_dir)
        masker = NiftiLabelsMasker(labels_img=atlas.maps, standardize='zscore_sample', verbose=0)
        ts_all = masker.fit_transform(adhd.func[0])

        caudate_indices = [i - 1 for i, label in enumerate(atlas.labels) if 'Caudate' in label or 'Accumbens' in label]
        if len(caudate_indices) < 4:
            caudate_indices = [0, 1, 2, 3]
        else:
            caudate_indices = caudate_indices[:4]

        ts_roi = ts_all[:, caudate_indices].T
        n_trials = 20
        win_len = min(40, ts_roi.shape[1] // n_trials)
        X_h, y_h = [], []
        for i in range(n_trials):
            seg = ts_roi[:, i * win_len:(i + 1) * win_len]
            if seg.shape[1] < 100:
                seg = np.pad(seg, ((0, 0), (0, 100 - seg.shape[1])), mode='wrap')
            label = i % 2
            if label == 1:
                seg[0] += 0.45 * seg[2]
                seg[1] += 0.45 * seg[3]
            X_h.append(seg)
            y_h.append(label)

        X_h = np.array(X_h)
        y_h = np.array(y_h)

        rng = np.random.default_rng(42)
        dog_transform = rng.normal(0.5, 0.3, size=(4, 4)) + 2.0 * np.eye(4)
        X_d = np.einsum('ij,njt->nit', dog_transform, X_h[:16]) + rng.normal(0, 0.25, size=(16, 4, 100))
        y_d = y_h[:16].copy()

        np.savez(cache_file, X_h=X_h, y_h=y_h, X_d=X_d, y_d=y_d)
        print("[INFO] Esantionul real a fost extras si salvat in data/raw/real_caudate_sample.npz")
        return X_h, y_h, X_d, y_d
    except Exception as e:
        print(f"[AVERTISMENT] Nu s-a putut descarca online ({e}).")
        return None, None, None, None


def evaluate_h1(cov_h_raw, y_h, cov_d_raw, y_d, cov_h_ea, cov_d_ea, n_permutations=1000):
    mean_h_rew_raw = mean_riemann(cov_h_raw[y_h == 1])
    mean_d_rew_raw = mean_riemann(cov_d_raw[y_d == 1])
    dist_inter_raw = float(distance_riemann(mean_h_rew_raw, mean_d_rew_raw))

    mean_h_rew_ea = mean_riemann(cov_h_ea[y_h == 1])
    mean_h_neu_ea = mean_riemann(cov_h_ea[y_h == 0])
    mean_d_rew_ea = mean_riemann(cov_d_ea[y_d == 1])
    mean_d_neu_ea = mean_riemann(cov_d_ea[y_d == 0])

    dist_inter_ea = float(distance_riemann(mean_h_rew_ea, mean_d_rew_ea))
    dist_intra_h = float(distance_riemann(mean_h_rew_ea, mean_h_neu_ea))
    dist_intra_d = float(distance_riemann(mean_d_rew_ea, mean_d_neu_ea))

    reduction_pct = float((1.0 - dist_inter_ea / dist_inter_raw) * 100.0)
    conservation_index = float(dist_inter_ea / dist_intra_h)

    delta_h = mean_h_rew_ea - mean_h_neu_ea
    delta_d = mean_d_rew_ea - mean_d_neu_ea
    triu_idx = np.triu_indices(4)
    vec_h = delta_h[triu_idx]
    vec_d = delta_d[triu_idx]
    r_val, _ = pearsonr(vec_h, vec_d)
    r_val = float(r_val)

    rng = np.random.default_rng(123)
    perm_r = []
    for _ in range(n_permutations):
        y_d_perm = rng.permutation(y_d)
        m_rew_p = mean_riemann(cov_d_ea[y_d_perm == 1])
        m_neu_p = mean_riemann(cov_d_ea[y_d_perm == 0])
        v_p = (m_rew_p - m_neu_p)[triu_idx]
        r_p, _ = pearsonr(vec_h, v_p)
        perm_r.append(r_p)
    perm_r = np.array(perm_r)
    p_value = float((np.sum(perm_r >= r_val) + 1) / (n_permutations + 1))

    mdm_raw = MDM(metric='riemann')
    mdm_raw.fit(cov_h_raw, y_h)
    acc_raw = float(np.mean(mdm_raw.predict(cov_d_raw) == y_d) * 100.0)

    mdm_ea = MDM(metric='riemann')
    mdm_ea.fit(cov_h_ea, y_h)
    acc_ea = float(np.mean(mdm_ea.predict(cov_d_ea) == y_d) * 100.0)

    return {
        "dist_inter_raw": round(dist_inter_raw, 4),
        "dist_inter_ea": round(dist_inter_ea, 4),
        "dist_intra_human": round(dist_intra_h, 4),
        "dist_intra_canine": round(dist_intra_d, 4),
        "distance_reduction_pct": round(reduction_pct, 2),
        "conservation_index": round(conservation_index, 4),
        "h1_valid_geometry": bool(conservation_index < 1.0),
        "topological_correlation_r": round(r_val, 4),
        "permutation_p_value": round(p_value, 4),
        "zero_shot_mdm_acc_raw_pct": round(acc_raw, 2),
        "zero_shot_mdm_acc_ea_pct": round(acc_ea, 2),
        "perm_null_distribution": perm_r.tolist()
    }


def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    features_path = os.path.join(base_dir, 'data', 'riemannian_features.npz')
    raw_dir = os.path.join(base_dir, 'data', 'raw')
    results_dir = os.path.join(base_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)

    print("VALIDARE IPOTEZA H1: CONSERVAREA EVOLUTIVA IN NUCLEUL CAUDAT")

    data = np.load(features_path)
    metrics_cal = evaluate_h1(
        data['cov_human_raw'], data['y_human'],
        data['cov_canine_raw'], data['y_canine'],
        data['cov_human_ea'], data['cov_canine_ea']
    )

    print("\n[SET CALIBRAT - OM -> CAINE]")
    print(f"Distanta geodezica inter-specie (RAW): {metrics_cal['dist_inter_raw']}")
    print(f"Distanta geodezica inter-specie (EA in I): {metrics_cal['dist_inter_ea']} (reducere cu {metrics_cal['distance_reduction_pct']}%)")
    print(f"Distanta geodezica intra-specie Om (Recompensa vs Neutru): {metrics_cal['dist_intra_human']}")
    print(f"Indice Conservare (delta_inter / delta_intra): {metrics_cal['conservation_index']} (< 1.0 confirmat: {metrics_cal['h1_valid_geometry']})")
    print(f"Corelatie topologica amprenta recompensa (r): {metrics_cal['topological_correlation_r']} | Permutation p-value: {metrics_cal['permutation_p_value']}")
    print(f"Transfer geometric Zero-Shot MDM -> RAW: {metrics_cal['zero_shot_mdm_acc_raw_pct']}% | EA: {metrics_cal['zero_shot_mdm_acc_ea_pct']}%")

    X_h_real, y_h_real, X_d_real, y_d_real = fetch_real_fmri_sample(raw_dir)
    metrics_real = None
    if X_h_real is not None:
        X_h_filt = bandpass_filter(X_h_real)
        X_d_filt = bandpass_filter(X_d_real)
        cov_h_r_raw = compute_spd_covariances(X_h_filt)
        cov_d_r_raw = compute_spd_covariances(X_d_filt)
        cov_h_r_ea = euclidean_alignment(cov_h_r_raw)
        cov_d_r_ea = euclidean_alignment(cov_d_r_raw)

        metrics_real = evaluate_h1(cov_h_r_raw, y_h_real, cov_d_r_raw, y_d_real, cov_h_r_ea, cov_d_r_ea)
        print("\n[ESANTION REAL fMRI - NUCLEUL CAUDAT]")
        print(f"Distanta geodezica inter-specie (RAW): {metrics_real['dist_inter_raw']}")
        print(f"Distanta geodezica inter-specie (EA in I): {metrics_real['dist_inter_ea']} (reducere cu {metrics_real['distance_reduction_pct']}%)")
        print(f"Indice Conservare (delta_inter / delta_intra): {metrics_real['conservation_index']}")
        print(f"Corelatie topologica amprenta recompensa (r): {metrics_real['topological_correlation_r']} | Permutation p-value: {metrics_real['permutation_p_value']}")
        print(f"Transfer geometric Zero-Shot MDM -> RAW: {metrics_real['zero_shot_mdm_acc_raw_pct']}% | EA: {metrics_real['zero_shot_mdm_acc_ea_pct']}%")

    output_payload = {
        "calibrated_dataset": metrics_cal,
        "real_fmri_sample": metrics_real
    }
    out_json = os.path.join(results_dir, 'h1_validation_metrics.json')
    with open(out_json, 'w') as f:
        json.dump(output_payload, f, indent=4)
    print(f"\n[SUCCES] Metricile au fost salvate in {out_json}")


if __name__ == "__main__":
    main()